import torch


def precompute_rope_freqs(
    head_dim: int,
    seq_len: int,
    base: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the (cos, sin) rotation matrices for RoPE.

    Only needs to be called once and then cached / moved to the right device.

    Args:
        head_dim: Dimension of a single attention head (must be even).
        seq_len:  Maximum sequence length to precompute for.
        base:     Base frequency (default 10 000, matches the original paper).

    Returns:
        cos, sin  – both shaped (seq_len, head_dim)
    """
    # Step 1: theta_i = 1 / base^(2i / head_dim)  for i = 0 … head_dim/2 - 1
    i = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (base ** (i / head_dim))

    # Step 2: positions  m = 0 … seq_len - 1
    positions = torch.arange(seq_len).float()

    # Step 3: outer product  →  (seq_len, head_dim/2)
    freqs = torch.outer(positions, theta)

    # Step 4: duplicate each angle for the pair  →  (seq_len, head_dim)
    freqs = torch.cat([freqs, freqs], dim=1)

    return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the last dimension of x by 90°:
        [x1 | x2]  →  [-x2 | x1]

    This is the rotation complement used in apply_rope.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE to a query or key tensor.

    Args:
        x:   (batch, n_heads, seq_len, head_dim)
        cos: (seq_len, head_dim)  – from precompute_rope_freqs
        sin: (seq_len, head_dim)  – from precompute_rope_freqs

    Returns:
        Rotated tensor with same shape as x.
    """
    # Unsqueeze for broadcasting over batch and heads
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    return (x * cos) + (rotate_half(x) * sin)
