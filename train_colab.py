"""
train_colab.py  —  single-file miniLLM trainer for Google Colab
----------------------------------------------------------------
Before running:
  1. Upload tokenizer.json       -> /content/tokenizer.json
  2. Upload tokenizer_train.txt  -> /content/tokenizer_train.txt
  3. Runtime -> Change runtime type -> T4 GPU
  4. Run:  !python train_colab.py
"""

import os
import re
import json
import math
import time
import shutil
import regex
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

class Tokenizer:
    def __init__(self):
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: Dict[Tuple[int, int], int] = {}
        self.special_tokens: Dict[str, int] = {}
        self._special_id_to_str: Dict[int, str] = {}
        self._pre_token_pattern = regex.compile(
            r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        )

    def _pre_tokenize(self, text: str) -> List[str]:
        return self._pre_token_pattern.findall(text)

    def _encode_chunk(self, text: str) -> List[int]:
        chunks = self._pre_tokenize(text)
        merge_rank = {pair: rank for rank, pair in enumerate(self.merges.keys())}
        result: List[int] = []
        for chunk in chunks:
            if not chunk:
                continue
            ids: List[int] = list(chunk.encode("utf-8"))
            while len(ids) >= 2:
                best_rank = float("inf")
                best_idx = -1
                for i in range(len(ids) - 1):
                    rank = merge_rank.get((ids[i], ids[i + 1]), float("inf"))
                    if rank < best_rank:
                        best_rank = rank
                        best_idx = i
                if best_idx == -1 or best_rank == float("inf"):
                    break
                pair = (ids[best_idx], ids[best_idx + 1])
                ids = ids[:best_idx] + [self.merges[pair]] + ids[best_idx + 2:]
            result.extend(ids)
        return result

    def _encode_with_special(self, text: str) -> List[int]:
        pattern = "(" + "|".join(
            re.escape(tok) for tok in sorted(self.special_tokens.keys(), key=len, reverse=True)
        ) + ")"
        result = []
        for part in re.split(pattern, text):
            if part in self.special_tokens:
                result.append(self.special_tokens[part])
            elif part:
                result.extend(self._encode_chunk(part))
        return result

    def encode(self, text: str) -> List[int]:
        if self.special_tokens:
            return self._encode_with_special(text)
        return self._encode_chunk(text)

    def decode(self, ids: List[int]) -> str:
        byte_pieces = []
        for tid in ids:
            if tid in self._special_id_to_str:
                byte_pieces.append(self._special_id_to_str[tid].encode("utf-8"))
            else:
                byte_pieces.append(self.vocab.get(tid, bytes([tid & 0xFF])))
        return b"".join(byte_pieces).decode("utf-8", errors="replace")

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        tok = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for pair_list, new_id in data["merges"]:
            pair = (pair_list[0], pair_list[1])
            tok.merges[pair] = new_id
            tok.vocab[new_id] = tok.vocab[pair[0]] + tok.vocab[pair[1]]
        for token, tid in data["special_tokens"].items():
            tok.special_tokens[token] = tid
            tok._special_id_to_str[tid] = token
            tok.vocab[tid] = token.encode("utf-8")
        print(f"Tokenizer loaded  ({len(tok.merges)} merges, vocab={len(tok.vocab)})")
        return tok


# ─────────────────────────────────────────────────────────────────────────────
# Model components
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return self.weight * x / (rms + self.eps)


def precompute_rope_freqs(head_dim: int, seq_len: int, base: float = 10_000.0):
    i = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (base ** (i / head_dim))
    positions = torch.arange(seq_len).float()
    freqs = torch.outer(positions, theta)
    freqs = torch.cat([freqs, freqs], dim=1)
    return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
    return torch.triu(mask, diagonal=1)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff = (d_ff + 64) // 64 * 64
        self.W_1   = nn.Linear(d_model, d_ff,    bias=False)
        self.W_2   = nn.Linear(d_model, d_ff,    bias=False)
        self.W_out = nn.Linear(d_ff,    d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_out(F.silu(self.W_1(x)) * self.W_2(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape

        def split_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        Q = apply_rope(split_heads(self.W_q(x)), cos, sin)
        K = apply_rope(split_heads(self.W_k(x)), cos, sin)
        V = split_heads(self.W_v(x))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int = None):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn  = MultiHeadAttention(d_model, n_heads)
        self.ffn   = SwiGLUFFN(d_model, d_ff)

    def forward(self, x, cos, sin, mask=None):
        x = x + self.attn(self.norm1(x), cos, sin, mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, max_seq_len: int, d_ff: int = None):
        super().__init__()
        self.d_model     = d_model
        self.head_dim    = d_model // n_heads
        self.max_seq_len = max_seq_len

        self.embedding  = nn.Embedding(vocab_size, d_model)
        self.blocks      = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head    = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying

        cos, sin = precompute_rope_freqs(self.head_dim, max_seq_len)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        mask = build_causal_mask(max_seq_len, torch.device('cpu'))
        self.register_buffer('mask', mask)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02 / math.sqrt(self.d_model))
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape
        assert T <= self.max_seq_len
        x    = self.embedding(token_ids)
        cos  = self.cos[:T]
        sin  = self.sin[:T]
        mask = self.mask[:T, :T]
        for block in self.blocks:
            x = block(x, cos, sin, mask=mask)
        return self.lm_head(self.final_norm(x))


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────────────────────

class AdamW:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.lr           = lr
        self.beta1        = betas[0]
        self.beta2        = betas[1]
        self.eps          = eps
        self.weight_decay = weight_decay
        self.t            = 0
        self.params       = [p for p in params if p.requires_grad]
        self.m            = [torch.zeros_like(p) for p in self.params]
        self.v            = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            self.v[i].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)
            p.mul_(1.0 - self.lr * self.weight_decay)
            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_lm_loss(model: TransformerLM, token_ids: torch.Tensor) -> torch.Tensor:
    inputs  = token_ids[:, :-1]
    targets = token_ids[:, 1:]
    logits  = model(inputs)
    B, T, V = logits.shape
    return F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step, total_steps, max_lr, min_lr, warmup_steps):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + (max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def clip_grad_norm(params, max_norm=1.0):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads)).item()
    clip_coef  = max_norm / max(total_norm, max_norm)
    for g in grads:
        g.mul_(clip_coef)
    return total_norm


def save_checkpoint(model, optimizer, step, loss, path, drive_dir=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'step': step,
        'loss': loss,
        'model_state': model.state_dict(),
        'optimizer_state': {
            't': optimizer.t,
            'm': [m.clone() for m in optimizer.m],
            'v': [v.clone() for v in optimizer.v],
        }
    }, path)
    print(f"[ckpt] Saved -> {path}")
    if drive_dir:
        os.makedirs(drive_dir, exist_ok=True)
        dest = os.path.join(drive_dir, os.path.basename(path))
        shutil.copy2(path, dest)
        print(f"[ckpt] Mirrored -> {dest}")


def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    optimizer.t = ckpt['optimizer_state']['t']
    for i, (m, v) in enumerate(zip(
        ckpt['optimizer_state']['m'], ckpt['optimizer_state']['v']
    )):
        optimizer.m[i].copy_(m)
        optimizer.v[i].copy_(v)
    print(f"[ckpt] Resumed from step {ckpt['step']}")
    return ckpt['step']


def train(model, optimizer, data, total_steps, batch_size, seq_len,
          max_lr=3e-4, min_lr=3e-5, warmup_steps=200, max_grad_norm=1.0,
          ckpt_every=500, ckpt_dir="checkpoints", drive_dir=None,
          device="cpu", start_step=0):

    model.to(device)
    model.train()
    run_loss = 0.0

    print(f"Training for {total_steps} steps | batch={batch_size} | seq={seq_len} | device={device}\n")

    for step in range(start_step, total_steps):
        t0 = time.time()

        optimizer.lr = get_lr(step, total_steps, max_lr, min_lr, warmup_steps)

        max_start = len(data) - seq_len - 1
        starts = torch.randint(0, max_start, (batch_size,))
        batch  = torch.stack([data[s: s + seq_len + 1] for s in starts]).to(device)

        optimizer.zero_grad()
        loss = compute_lm_loss(model, batch)
        loss.backward()
        grad_norm = clip_grad_norm(optimizer.params, max_norm=max_grad_norm)
        optimizer.step()

        step_time = time.time() - t0
        run_loss += loss.item()

        if step % 10 == 0:
            avg_loss = run_loss / (10 if step > 0 else 1)
            run_loss = 0.0
            tok_per_sec = (batch_size * seq_len) / step_time
            print(
                f"step {step:5d} | loss {avg_loss:.4f} | "
                f"lr {optimizer.lr:.2e} | grad_norm {grad_norm:.3f} | "
                f"{tok_per_sec:.0f} tok/s"
            )

        if step > 0 and step % ckpt_every == 0:
            save_checkpoint(model, optimizer, step, loss.item(),
                            path=f"{ckpt_dir}/step_{step:06d}.pt",
                            drive_dir=drive_dir)

    print("\nTraining complete")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Google Drive (Colab only) ─────────────────────────────────────────
    DRIVE_CKPT_DIR = None
    try:
        from google.colab import drive as _drive
        _drive.mount('/content/drive', force_remount=False)
        DRIVE_CKPT_DIR = '/content/drive/MyDrive/miniLLM_checkpoints'
        os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)
        print(f"[drive] Checkpoints will be mirrored to {DRIVE_CKPT_DIR}")
    except Exception:
        pass

    # ── Load tokenizer ────────────────────────────────────────────────────
    tok = Tokenizer.load("tokenizer.json")
    vocab_size = len(tok.vocab)

    # ── Tokenize corpus ───────────────────────────────────────────────────
    print("Tokenizing corpus...")
    with open("tokenizer_train.txt", "r", encoding="utf-8") as f:
        text = f.read()
    token_ids = tok.encode(text)
    data = torch.tensor(token_ids, dtype=torch.long)
    print(f"Dataset: {len(data):,} tokens")

    # ── Build model ───────────────────────────────────────────────────────
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=6,
        max_seq_len=512,
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=3e-4)

    # ── Resume from latest checkpoint ─────────────────────────────────────
    ckpt_dir   = "checkpoints"
    start_step = 0

    def _latest_ckpt(directory):
        if directory and os.path.isdir(directory):
            pts = sorted(f for f in os.listdir(directory) if f.endswith('.pt'))
            return os.path.join(directory, pts[-1]) if pts else None
        return None

    resume_path = _latest_ckpt(DRIVE_CKPT_DIR) or _latest_ckpt(ckpt_dir)
    if resume_path:
        if DRIVE_CKPT_DIR and resume_path.startswith(DRIVE_CKPT_DIR):
            os.makedirs(ckpt_dir, exist_ok=True)
            local_copy = os.path.join(ckpt_dir, os.path.basename(resume_path))
            shutil.copy2(resume_path, local_copy)
            resume_path = local_copy
        start_step = load_checkpoint(model, optimizer, resume_path)

    # ── Train ─────────────────────────────────────────────────────────────
    train(
        model=model,
        optimizer=optimizer,
        data=data,
        total_steps=5000,
        batch_size=16,
        seq_len=512,
        max_lr=3e-4,
        min_lr=3e-5,
        warmup_steps=200,
        max_grad_norm=1.0,
        ckpt_every=500,
        ckpt_dir=ckpt_dir,
        drive_dir=DRIVE_CKPT_DIR,
        device=device,
        start_step=start_step,
    )
