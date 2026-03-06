import torch
import torch.nn as nn
import math

from model.components.rms_norm import RMSNorm
from model.components.attention import MultiHeadAttention, build_causal_mask
from model.components.swi_glu import SwiGLUFFN
from model.components.rope import precompute_rope_freqs


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int = None):
        super().__init__()

        # Two RMSNorms - one before attention, one before FFN
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # Attention and FFN sublayers
        self.attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:

        # sublayer 1: multi-head attention
        x = x + self.attn(self.norm1(x), cos, sin, mask)

        # sublayer 2: FFN
        x = x + self.ffn(self.norm2(x))

        return x


class TransformerLM(nn.Module):

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_seq_len: int, d_ff: int = None):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.head_dim = d_model // n_heads  # fix: was `-` instead of `=`

        # Token embedding table
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff)
            for _ in range(n_layers)
        ])

        # Final norm before LM head
        self.final_norm = RMSNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: LM head and embedding share the exact same weight matrix
        self.lm_head.weight = self.embedding.weight

        # Precompute RoPE tables once
        cos, sin = precompute_rope_freqs(head_dim=self.head_dim, seq_len=max_seq_len)

        # Register as buffers - move with .to(device), not parameters
        self.register_buffer('cos', cos)   # fix: was `cas` typo
        self.register_buffer('sin', sin)   # fix: was `registers_buffer`

        # Precompute causal mask once
        mask = build_causal_mask(seq_len=max_seq_len, device=torch.device('cpu'))  # fix: `seq_len` was undefined
        self.register_buffer('mask', mask)

        self._init_weights()

    def _init_weights(self):
        """
        Xavier-style initialization to keep activation variance stable.
        Without this, deep networks explode or vanish at init.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(self.d_model))
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,  # (batch, seq_len) integers
    ) -> torch.Tensor:            # (batch, seq_len, vocab_size) logits
        B, T = token_ids.shape
        assert T <= self.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        # Step 1: embed token ids to vectors
        x = self.embedding(token_ids)  # (B, T, d_model)

        # Step 2: slice RoPE tables to current sequence length
        cos = self.cos[:T]  # (T, head_dim)
        sin = self.sin[:T]

        # Step 3: slice causal mask to current sequence length
        mask = self.mask[:T, :T]  # (T, T)

        # Step 4: pass through all transformer blocks
        for block in self.blocks:
            x = block(x, cos, sin, mask=mask)  # fix: was `mask=None`

        # Step 5: Final normalization
        x = self.final_norm(x)

        # Step 6: project to vocabulary logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits
