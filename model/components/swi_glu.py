import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network (used in LLaMA / PaLM).

    Formula:  FFN(x) = W_out( SiLU(W1(x)) ⊙ W2(x) )

    The hidden dimension defaults to the value recommended in the SwiGLU paper:
        d_ff = ⌈8·d_model/3⌉ rounded up to the nearest multiple of 64.

    Args:
        d_model: Input / output dimension.
        d_ff:    Hidden dimension. If None, computed automatically.
    """

    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()

        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff = (d_ff + 64) // 64 * 64  # round up to nearest 64

        self.d_ff = d_ff

        self.W_1   = nn.Linear(d_model, d_ff, bias=False)  # value projection
        self.W_2   = nn.Linear(d_model, d_ff, bias=False)  # gate projection
        self.W_out = nn.Linear(d_ff,   d_model, bias=False)  # output projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: value and gate streams in parallel
        value = self.W_1(x)
        gate  = self.W_2(x)

        # Step 2: gated activation
        hidden = F.silu(value) * gate

        # Step 3: project back to d_model
        return self.W_out(hidden)
