import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm omits the mean-centering step and has no bias,
    making it cheaper while empirically matching performance (used in LLaMA).

    Args:
        d_model: Feature dimension to normalize over.
        eps:     Small constant for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        # Learnable per-feature scale (gamma), initialised to 1
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)

        # Step 1: RMS over the last (feature) dimension
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()

        # Step 2: Normalize
        x_normed = x / (rms + self.eps)

        # Step 3: Scale by learned weight
        return self.weight * x_normed
