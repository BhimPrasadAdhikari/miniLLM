import torch
from model.components import MultiHeadAttention, precompute_rope_freqs, build_causal_mask

torch.manual_seed(42)

batch, seq_len, d_model, n_heads = 2, 16, 512, 8 

head_dim = d_model // n_heads 

attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

cos, sin = precompute_rope_freqs(head_dim=head_dim, seq_len=seq_len)

mask = build_causal_mask(seq_len=seq_len, device=torch.device("cpu"))

# forward pass 
x = torch.randn(batch, seq_len, d_model)
out = attn(x, cos, sin, mask=mask)

print(f"Input  shape : {x.shape}")
print(f"Output shape : {out.shape}")    # must match input



# casuality test 
# if I change token at position 8, positions 0-7 must not change 

x2 = x.clone()
x2[:, 8] = torch.randn(d_model)

out2 = attn(x2, cos, sin, mask=mask)

affected = (out2 - out).abs().max(dim=-1).values # (batch, seq)
print("Casuality test - Max change per position: ")
print(f"Positions 0-7: {affected[0, :8].max().item():.6f}")
print(f"Positions 8-15: {affected[0, 8:].max().item():.6f}")

# parameter count
params = sum(p.numel() for p in attn.parameters())
print(f"Total attention params: {params:,}")
