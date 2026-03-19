"""
BitLinear: Ternary quantization-aware linear layer for Parameter Golf.

Drop-in replacement for CastedLinear. Maintains latent fp32 weights;
forward pass uses ternary {-1, 0, +1} when qat_enabled=True, with
straight-through estimator for gradients.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class BitLinear(nn.Linear):
    """
    Ternary quantization-aware linear layer.
    Maintains latent fp32 weights; forward pass uses ternary {-1, 0, +1}
    when qat_enabled=True, with straight-through estimator for gradients.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        # Per-output-channel scale factor for ternary quantization
        self.register_buffer("weight_scale", torch.ones(out_features))
        self.qat_enabled: bool = False
        self.l1_lambda: float = 1e-5  # L1 regularization strength

    @staticmethod
    def quantize_ternary(w: Tensor) -> tuple[Tensor, Tensor]:
        """AbsMedian quantization: scale = mean(|w|) per row, then round to {-1,0,+1}."""
        scale = w.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        w_scaled = w / scale
        w_ternary = torch.clamp(torch.round(w_scaled), -1, 1)
        return w_ternary, scale.squeeze(-1)

    def forward(self, x: Tensor) -> Tensor:
        if self.qat_enabled:
            w_ternary, scale = self.quantize_ternary(self.weight)
            # STE: use ternary in forward, gradient flows to self.weight
            w_q = self.weight + (w_ternary * scale.unsqueeze(-1) - self.weight).detach()
            return F.linear(x, w_q.to(x.dtype), self.bias)
        else:
            return F.linear(x, self.weight.to(x.dtype), self.bias)


def compute_l1_reg(model: nn.Module, l1_lambda: float) -> Tensor:
    """L1 regularization on BitLinear latent weights to encourage ternary zeros."""
    reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for m in model.modules():
        if isinstance(m, BitLinear) and m.qat_enabled:
            reg = reg + m.weight.abs().mean()
    return l1_lambda * reg
