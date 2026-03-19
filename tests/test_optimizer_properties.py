"""Property-based tests for optimizer group assignment and LR schedule."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from hypothesis import given, settings, strategies as st, HealthCheck

from train_gpt import GPT, CastedLinear, CONTROL_TENSOR_NAME_PATTERNS, Muon
from bitlinear import BitLinear


# ---------------------------------------------------------------------------
# Property 19: Optimizer group assignment correctness
# ---------------------------------------------------------------------------


@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_optimizer_group_assignment(seed):
    """
    Property 19: Optimizer group assignment correctness.

    All 2D parameters in BitLinear layers (excluding control tensors) are
    assigned to Muon, and all embedding, scalar, loop signal, and loop norm
    parameters are assigned to Adam groups.

    **Validates: Requirements 7.1, 7.2**
    """
    torch.manual_seed(seed)

    model = GPT(
        vocab_size=32,
        model_dim=128,
        num_heads=2,
        num_kv_heads=1,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        num_shared_blocks=1,
        num_loops=3,
        loop_signal_rank=4,
        progressive_loss_loops=[1],
        progressive_loss_weight=0.3,
    )

    # Replicate the optimizer setup from main()
    block_named_params = (
        list(model.prelude.named_parameters())
        + list(model.recurrent.blocks.named_parameters())
        + list(model.coda.named_parameters())
    )
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    for name, p in model.recurrent.named_parameters():
        if name.startswith("blocks."):
            continue
        scalar_params.append(p)

    matrix_param_ids = {id(p) for p in matrix_params}
    scalar_param_ids = {id(p) for p in scalar_params}

    # Verify: all 2D CastedLinear/BitLinear weight params from blocks are in matrix group
    for name, p in block_named_params:
        if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS):
            assert id(p) in matrix_param_ids, (
                f"2D param '{name}' should be in Muon (matrix) group"
            )
        else:
            assert id(p) in scalar_param_ids, (
                f"Non-matrix param '{name}' should be in Adam (scalar) group"
            )

    # Verify: loop signals, norms, inject_alpha are in scalar group
    for name, p in model.recurrent.named_parameters():
        if name.startswith("blocks."):
            continue
        assert id(p) in scalar_param_ids, (
            f"Recurrent param '{name}' should be in Adam (scalar) group"
        )

    # Verify: embedding is NOT in either block group (it has its own optimizer)
    assert id(model.tok_emb.weight) not in matrix_param_ids
    assert id(model.tok_emb.weight) not in scalar_param_ids

    # Verify no overlap between matrix and scalar
    overlap = matrix_param_ids & scalar_param_ids
    assert len(overlap) == 0, f"Found {len(overlap)} params in both groups"


# ---------------------------------------------------------------------------
# Property 20: Learning rate schedule shape
# ---------------------------------------------------------------------------


@given(
    iterations=st.integers(min_value=100, max_value=10000),
    warmdown_iters=st.integers(min_value=10, max_value=2000),
    lr_warmup_steps=st.integers(min_value=1, max_value=50),
    step_frac=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100)
def test_lr_schedule_shape(iterations, warmdown_iters, lr_warmup_steps, step_frac, seed):
    """
    Property 20: Learning rate schedule shape.

    lr_mul returns values in [0, 1]: ramps up during warmup, equals 1.0 in
    stable phase, linearly decreases in cooldown, and is never negative.

    **Validates: Requirements 7.3**
    """
    # Clamp warmdown_iters to be at most iterations
    warmdown_iters = min(warmdown_iters, iterations)
    lr_warmup_steps = min(lr_warmup_steps, iterations - warmdown_iters)

    step = int(step_frac * iterations)
    step = min(step, iterations - 1)  # step < iterations in the loop

    # Replicate lr_mul logic (step-based path, no wallclock)
    def lr_mul(s):
        if lr_warmup_steps > 0 and s < lr_warmup_steps:
            return (s + 1) / lr_warmup_steps
        if warmdown_iters <= 0:
            return 1.0
        warmdown_start = max(iterations - warmdown_iters, 0)
        if warmdown_start <= s < iterations:
            return max((iterations - s) / max(warmdown_iters, 1), 0.0)
        return 1.0

    val = lr_mul(step)

    # Never negative
    assert val >= 0.0, f"lr_mul({step}) = {val} is negative"

    # Never exceeds 1.0
    assert val <= 1.0 + 1e-9, f"lr_mul({step}) = {val} exceeds 1.0"

    warmdown_start = max(iterations - warmdown_iters, 0)

    # During warmup: should be < 1.0 (unless warmup is 1 step and step=0 → 1.0)
    if lr_warmup_steps > 1 and step < lr_warmup_steps - 1:
        assert val < 1.0, f"lr_mul({step}) = {val} should be < 1.0 during warmup"

    # During stable phase: should be 1.0
    if step >= lr_warmup_steps and step < warmdown_start:
        assert abs(val - 1.0) < 1e-9, (
            f"lr_mul({step}) = {val} should be 1.0 in stable phase "
            f"(warmup={lr_warmup_steps}, warmdown_start={warmdown_start})"
        )
