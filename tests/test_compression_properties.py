"""Property-based tests for the ternary compression pipeline."""

import math
import sys
import zlib
from pathlib import Path

# Add parent directory so we can import from train_gpt and bitlinear
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from hypothesis import given, settings, strategies as st, HealthCheck

from train_gpt import (
    pack_ternary,
    unpack_ternary,
    quantize_state_dict_ternary,
    dequantize_state_dict_ternary,
)
from bitlinear import BitLinear


# ---------------------------------------------------------------------------
# Property 14: Ternary packing efficiency
# ---------------------------------------------------------------------------

@given(
    rows=st.integers(min_value=1, max_value=128),
    cols=st.integers(min_value=1, max_value=256),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_ternary_packing_efficiency(rows, cols, seed):
    """
    Property 14: Ternary packing efficiency.

    pack_ternary produces at most ceil(N/4) bytes for N values.

    **Validates: Requirements 5.2**
    """
    torch.manual_seed(seed)
    w = torch.randn(rows, cols)
    w_ternary, _ = BitLinear.quantize_ternary(w)

    packed_bytes, shape = pack_ternary(w_ternary)
    n = rows * cols
    max_expected_bytes = math.ceil(n / 4)

    assert len(packed_bytes) <= max_expected_bytes, (
        f"pack_ternary produced {len(packed_bytes)} bytes for {n} values, "
        f"expected at most ceil({n}/4) = {max_expected_bytes}"
    )

    # Also verify round-trip: unpack must recover the original ternary values
    recovered = unpack_ternary(packed_bytes, shape)
    assert torch.equal(w_ternary, recovered.float()), (
        f"Round-trip failed: max diff = {(w_ternary - recovered.float()).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Property 15: Compression pipeline round-trip
# ---------------------------------------------------------------------------

@given(
    dim=st.sampled_from([64, 128]),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_compression_pipeline_roundtrip(dim, seed):
    """
    Property 15: Compression pipeline round-trip.

    Compress then decompress yields identical model outputs within
    floating-point tolerance.

    **Validates: Requirements 5.5**
    """
    torch.manual_seed(seed)

    # Build a minimal state dict with both ternary-eligible and passthrough tensors
    state_dict = {
        # Large 2D tensor -> ternary treatment (numel > 65536 requires dim >= 256x256,
        # so we use a threshold-aware size)
        "big_weight": torch.randn(512, 256),   # 131072 > 65536 -> ternary
        "small_weight": torch.randn(dim, dim),  # <= 65536 -> passthrough as fp16
        "bias": torch.randn(dim),               # 1D -> passthrough
        "int_param": torch.tensor([1, 2, 3]),    # non-float -> passthrough
    }

    # Compress
    compressed = quantize_state_dict_ternary(state_dict)

    # Decompress
    restored = dequantize_state_dict_ternary(compressed)

    # Verify all keys present
    assert set(restored.keys()) == set(state_dict.keys()), (
        f"Key mismatch: {set(restored.keys())} vs {set(state_dict.keys())}"
    )

    # For the big ternary-compressed weight: the restored values should be the
    # ternary-quantized version reconstructed through fp16 scales (as the pipeline
    # stores scales in fp16), so we replicate that path for the expected value
    w_ternary, scale = BitLinear.quantize_ternary(state_dict["big_weight"])
    scale_fp16 = scale.to(torch.float16).to(torch.float32)  # round-trip through fp16
    expected_big = (w_ternary * scale_fp16.unsqueeze(-1)).to(torch.bfloat16)
    actual_big = restored["big_weight"]
    assert actual_big.shape == expected_big.shape
    assert torch.allclose(actual_big.float(), expected_big.float(), atol=1e-3), (
        f"Ternary round-trip max diff: {(actual_big.float() - expected_big.float()).abs().max().item()}"
    )

    # For passthrough tensors: should match within fp16 tolerance
    for name in ["small_weight", "bias"]:
        orig_fp16 = state_dict[name].to(torch.float16).float()
        rest_float = restored[name].float()
        assert torch.allclose(rest_float, orig_fp16, atol=1e-3), (
            f"Passthrough round-trip failed for '{name}': "
            f"max diff = {(rest_float - orig_fp16).abs().max().item()}"
        )

    # Non-float passthrough should be exact
    assert torch.equal(restored["int_param"], state_dict["int_param"])


# ---------------------------------------------------------------------------
# Property 5: Ternary compression achieves target bits-per-weight
# ---------------------------------------------------------------------------

@given(
    rows=st.integers(min_value=64, max_value=512),
    cols=st.integers(min_value=64, max_value=512),
    zero_frac=st.floats(min_value=0.5, max_value=0.95),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100, deadline=None)
def test_ternary_compression_bits_per_weight(rows, cols, zero_frac, seed):
    """
    Property 5: Ternary compression achieves target bits-per-weight.

    With >=50% zeros, achieves <=1.7 bits per weight after zlib.

    **Validates: Requirements 1.5**
    """
    torch.manual_seed(seed)
    n = rows * cols

    # Generate a ternary tensor with at least zero_frac zeros
    # Start with all zeros, then fill (1 - zero_frac) with random {-1, +1}
    w_ternary = torch.zeros(rows, cols)
    num_nonzero = int(n * (1 - zero_frac))
    if num_nonzero > 0:
        indices = torch.randperm(n)[:num_nonzero]
        signs = torch.randint(0, 2, (num_nonzero,)) * 2 - 1  # {-1, +1}
        w_ternary.view(-1)[indices] = signs.float()

    # Verify we actually have >= 50% zeros
    actual_zero_frac = (w_ternary == 0).float().mean().item()
    assert actual_zero_frac >= 0.5, f"Zero fraction {actual_zero_frac} < 0.5"

    # Pack and compress
    packed_bytes, _ = pack_ternary(w_ternary)
    compressed = zlib.compress(packed_bytes, level=9)

    bits_per_weight = (len(compressed) * 8) / n
    assert bits_per_weight <= 1.7, (
        f"bits_per_weight = {bits_per_weight:.4f} > 1.7 "
        f"(zeros={actual_zero_frac:.2%}, n={n}, compressed={len(compressed)} bytes)"
    )
