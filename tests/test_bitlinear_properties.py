"""Property-based tests for BitLinear ternary quantization."""

import sys
from pathlib import Path

# Add parent directory so we can import bitlinear directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from hypothesis import given, settings, strategies as st, HealthCheck

from bitlinear import BitLinear, compute_l1_reg


@given(
    rows=st.integers(min_value=1, max_value=64),
    cols=st.integers(min_value=1, max_value=128),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100)
def test_ternary_quantization_roundtrip(rows, cols, seed):
    """
    Property 1: Ternary quantization round-trip.

    For any float32 weight matrix, quantize_ternary produces values only
    in {-1, 0, +1} with positive scales.

    **Validates: Requirements 1.1, 1.6**
    """
    torch.manual_seed(seed)
    w = torch.randn(rows, cols)
    w_ternary, scale = BitLinear.quantize_ternary(w)

    # All values must be in {-1, 0, +1}
    assert torch.all((w_ternary == -1) | (w_ternary == 0) | (w_ternary == 1)), (
        f"Found values outside {{-1, 0, +1}}: {w_ternary.unique().tolist()}"
    )

    # All scale values must be positive
    assert torch.all(scale > 0), (
        f"Found non-positive scale values: {scale[scale <= 0].tolist()}"
    )

    # Scale shape must match number of rows
    assert scale.shape == (rows,), (
        f"Expected scale shape ({rows},), got {scale.shape}"
    )


@given(
    in_features=st.integers(min_value=4, max_value=64),
    out_features=st.integers(min_value=4, max_value=64),
    batch_size=st.integers(min_value=1, max_value=8),
    seq_len=st.integers(min_value=1, max_value=16),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100)
def test_ste_gradient_flow(in_features, out_features, batch_size, seq_len, seed):
    """
    Property 2: STE gradient flow.

    For any BitLinear layer with qat_enabled=True and any random input tensor,
    backward produces non-None finite gradients on weight with the same shape.

    **Validates: Requirements 1.2**
    """
    torch.manual_seed(seed)
    layer = BitLinear(in_features, out_features)
    layer.qat_enabled = True
    x = torch.randn(batch_size, seq_len, in_features)
    out = layer(x)
    out.sum().backward()
    assert layer.weight.grad is not None, "weight.grad should not be None after backward"
    assert layer.weight.grad.shape == layer.weight.shape, (
        f"Expected grad shape {layer.weight.shape}, got {layer.weight.grad.shape}"
    )
    assert torch.all(torch.isfinite(layer.weight.grad)), (
        "All gradient values must be finite (no NaN or Inf)"
    )


@given(
    in_features=st.integers(min_value=4, max_value=32),
    out_features=st.integers(min_value=4, max_value=32),
    scale_factor=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100)
def test_l1_regularization_proportionality(in_features, out_features, scale_factor, seed):
    """Property 3: L1 regularization proportionality. Validates: Requirements 1.3"""
    torch.manual_seed(seed)

    # Create a simple model with a BitLinear layer
    model = torch.nn.Sequential(BitLinear(in_features, out_features))
    for m in model.modules():
        if isinstance(m, BitLinear):
            m.qat_enabled = True

    l1_lambda = 1e-4
    reg_original = compute_l1_reg(model, l1_lambda)

    # Non-negative
    assert reg_original >= 0, f"L1 reg should be non-negative, got {reg_original.item()}"

    # Scale weights by factor c
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, BitLinear):
                m.weight.mul_(scale_factor)

    reg_scaled = compute_l1_reg(model, l1_lambda)

    # Should scale linearly
    expected = reg_original * scale_factor
    assert torch.allclose(reg_scaled, expected, rtol=1e-4, atol=1e-7), (
        f"Expected {expected.item()}, got {reg_scaled.item()} (scale_factor={scale_factor})"
    )


# Minimal CastedLinear for shape comparison (mirrors train_gpt.py baseline)
class CastedLinear(torch.nn.Linear):
    def forward(self, x):
        return torch.nn.functional.linear(
            x,
            self.weight.to(x.dtype),
            self.bias.to(x.dtype) if self.bias is not None else None,
        )


@given(
    in_features=st.integers(min_value=4, max_value=64),
    out_features=st.integers(min_value=4, max_value=64),
    batch_size=st.integers(min_value=1, max_value=8),
    seq_len=st.integers(min_value=1, max_value=32),
    qat_on=st.booleans(),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_bitlinear_drop_in_shape_compatibility(in_features, out_features, batch_size, seq_len, qat_on, seed):
    """Property 4: BitLinear drop-in shape compatibility.

    BitLinear produces identical output shapes to CastedLinear for any valid input.

    **Validates: Requirements 1.4**
    """
    torch.manual_seed(seed)

    bit_layer = BitLinear(in_features, out_features)
    bit_layer.qat_enabled = qat_on

    casted_layer = CastedLinear(in_features, out_features, bias=False)

    x = torch.randn(batch_size, seq_len, in_features)

    bit_out = bit_layer(x)
    casted_out = casted_layer(x)

    assert bit_out.shape == casted_out.shape, (
        f"Shape mismatch: BitLinear {bit_out.shape} vs CastedLinear {casted_out.shape}"
    )


@given(
    in_features=st.integers(min_value=4, max_value=64),
    out_features=st.integers(min_value=4, max_value=64),
    batch_size=st.integers(min_value=1, max_value=8),
    seq_len=st.integers(min_value=1, max_value=16),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100)
def test_latent_weight_preservation_during_qat(in_features, out_features, batch_size, seq_len, seed):
    """Property 7: Latent weight preservation during QAT. Validates: Requirements 2.3"""
    torch.manual_seed(seed)
    layer = BitLinear(in_features, out_features)
    layer.qat_enabled = True

    # Weight should be float32 before forward
    assert layer.weight.dtype == torch.float32

    x = torch.randn(batch_size, seq_len, in_features)
    out = layer(x)

    # Weight should still be float32 after forward
    assert layer.weight.dtype == torch.float32, (
        f"Weight dtype changed to {layer.weight.dtype} after forward pass"
    )

    # Also check after backward
    out.sum().backward()
    assert layer.weight.dtype == torch.float32, (
        f"Weight dtype changed to {layer.weight.dtype} after backward pass"
    )
