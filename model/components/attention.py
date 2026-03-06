import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math 

from model.components import apply_rope

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):

        super().__init__() 
        assert d_model % n_heads == 0 

        self.d_model = d_model 
        self.n_heads = n_heads 
        self.head_dim = d_model // n_heads 

        # single projection matrices (no biases)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False) 

        # Precompute the RoPE tables (I need to pass max_seq_len at init)
        # I will inject cos/sin from outside during forward for flexibility.
    
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:

        B, T, D = x.shape # (batch, seq_len, d_model)

        # Step 1: Project to Q, K, V 
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x) 

        # Step 2: reshape into heads 

        def split_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V) 


        # Step 3: Apply RoPE to Q & K only
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        # Step 4: scaled dot-product attention 
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale # (B, n_heads, T, T)

        # Step 5: causal mask 
        # each token can only attend to itself and past tokens (not future)
        if mask is not None:
            scores = scores + mask # mask will have -inf at illegal positions 
        
        # Step 6: softmax over last dim  (over keys)
        weights = F.softmax(scores, dim=-1) # (B, n_heads, T, T)

        # Step 7: weighted sum of V
        out = torch.matmul(weights, V) # (B, n_heads, T, head_dim)

        # Step 8: Merge the heads back 
        # (B, n_heads, T, head_dim)  -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, D) 

        
        # Step 9: Output projection
        return self.W_o(out) # (B, T, d_model)


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Upper triangular matrix of -inf above he diagonal. 
    Token at position i cannot attend to position j > i.
    """

    mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1) # keep diagonal, -inf above it 
    return mask # (seq_len, seq_len)





