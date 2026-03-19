"""Property-based tests for depth recurrence (RecurrentBlockGroup + Prelude-Recurrent-Coda)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add parent directory so we can import train_gpt directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from hypothesis import given, settings, strategies as st, HealthCheck

from train_gpt import Block, RecurrentBlockGroup


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Keep dimensions small for fast CPU tests but large enough for valid GQA configs.
# model_dim must be divisible by num_heads, and head_dim must be even.
# We fix num_heads=2, num_kv_heads=1, so model_dim must be divisible by 2
# and head_dim = model_dim / num_heads must be even → model_dim divisible by 4.
FIXED_NUM_HEADS = 2
FIXED_NUM_KV_HEADS = 1
FIXED_MLP_MULT = 2
FIXED_ROPE_BASE = 10000.0
FIXED_QK_GAIN_INIT = 1.5


def _make_recurrent_group(
    num_shared_blocks: int,
    num_loops: int,
    model_dim: int,
    loop_signal_rank: int,
    progressive_loss_loops: list[int],
) -> RecurrentBlockGroup:
    return RecurrentBlockGroup(
        num_shared_blocks=num_shared_blocks,
        num_loops=num_loops,
        model_dim=model_dim,
        num_heads=FIXED_NUM_HEADS,
        num_kv_heads=FIXED_NUM_KV_HEADS,
        mlp_mult=FIXED_MLP_MULT,
        rope_base=FIXED_ROPE_BASE,
        qk_gain_init=FIXED_QK_GAIN_INIT,
        loop_signal_rank=loop_signal_rank,
        progressive_loss_loops=progressive_loss_loops,
    )


# ---------------------------------------------------------------------------
# Property 8: Recurrent depth multiplication
# ---------------------------------------------------------------------------


@given(
    num_shared_blocks=st.integers(min_value=1, max_value=3),
    num_loops=st.integers(min_value=2, max_value=5),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_recurrent_depth_multiplication(num_shared_blocks, num_loops, seed):
    """
    Property 8: Recurrent depth multiplication.

    Forward pass executes each shared block exactly M times (K×M effective depth).

    **Validates: Requirements 3.2**
    """
    torch.manual_seed(seed)
    model_dim = 16  # small for speed; divisible by 4 for head_dim=8 (even)
    group = _make_recurrent_group(
        num_shared_blocks=num_shared_blocks,
        num_loops=num_loops,
        model_dim=model_dim,
        loop_signal_rank=4,
        progressive_loss_loops=[],
    )

    # Attach call counters to each shared block's forward
    call_counts = [0] * num_shared_blocks
    original_forwards = [block.forward for block in group.blocks]

    def make_counting_forward(idx, orig_fn):
        def counting_forward(x, x0):
            nonlocal call_counts
            call_counts[idx] += 1
            return orig_fn(x, x0)
        return counting_forward

    for i, block in enumerate(group.blocks):
        block.forward = make_counting_forward(i, original_forwards[i])

    x = torch.randn(1, 4, model_dim)
    x0 = torch.randn(1, 4, model_dim)
    group(x, x0)

    for i in range(num_shared_blocks):
        assert call_counts[i] == num_loops, (
            f"Block {i} called {call_counts[i]} times, expected {num_loops}"
        )


# ---------------------------------------------------------------------------
# Property 9: Per-loop differentiation via unique signals and norms
# ---------------------------------------------------------------------------


@given(
    num_loops=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100)
def test_per_loop_differentiation(num_loops, seed):
    """
    Property 9: Per-loop differentiation via unique signals and norms.

    Module contains exactly M distinct LayerNorm instances and M distinct
    loop signal slices.

    **Validates: Requirements 3.3, 3.4**
    """
    torch.manual_seed(seed)
    model_dim = 16
    group = _make_recurrent_group(
        num_shared_blocks=2,
        num_loops=num_loops,
        model_dim=model_dim,
        loop_signal_rank=4,
        progressive_loss_loops=[],
    )

    # Exactly M LayerNorm instances
    assert len(group.loop_norms) == num_loops, (
        f"Expected {num_loops} loop norms, got {len(group.loop_norms)}"
    )

    # All LayerNorm instances are distinct objects
    norm_ids = [id(norm) for norm in group.loop_norms]
    assert len(set(norm_ids)) == num_loops, (
        f"Expected {num_loops} distinct LayerNorm objects, got {len(set(norm_ids))}"
    )

    # Loop signal parameters have M slices
    assert group.loop_signal_down.shape[0] == num_loops, (
        f"loop_signal_down has {group.loop_signal_down.shape[0]} slices, expected {num_loops}"
    )
    assert group.loop_signal_up.shape[0] == num_loops, (
        f"loop_signal_up has {group.loop_signal_up.shape[0]} slices, expected {num_loops}"
    )

    # For any two different loop indices, the LayerNorm params differ in identity
    for i in range(num_loops):
        for j in range(i + 1, num_loops):
            assert group.loop_norms[i] is not group.loop_norms[j], (
                f"loop_norms[{i}] and loop_norms[{j}] are the same object"
            )


# ---------------------------------------------------------------------------
# Property 10: Input injection x0 influence
# ---------------------------------------------------------------------------


@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_input_injection_x0_influence(seed):
    """
    Property 10: Input injection x0 influence.

    Different x0 tensors produce different outputs from the recurrent group.

    **Validates: Requirements 3.5**
    """
    torch.manual_seed(seed)
    model_dim = 16
    group = _make_recurrent_group(
        num_shared_blocks=1,
        num_loops=3,
        model_dim=model_dim,
        loop_signal_rank=4,
        progressive_loss_loops=[],
    )
    group.eval()

    x = torch.randn(1, 4, model_dim)
    x0_a = torch.randn(1, 4, model_dim)
    x0_b = torch.randn(1, 4, model_dim)

    with torch.no_grad():
        out_a, _ = group(x.clone(), x0_a)
        out_b, _ = group(x.clone(), x0_b)

    assert not torch.allclose(out_a, out_b, atol=1e-6), (
        "Outputs should differ when x0 differs, but they are equal"
    )


# ---------------------------------------------------------------------------
# Property 11: Progressive loss generation at designated loops
# ---------------------------------------------------------------------------


@given(
    num_loops=st.integers(min_value=3, max_value=8),
    num_prog_loops=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_progressive_loss_generation(num_loops, num_prog_loops, seed):
    """
    Property 11: Progressive loss generation at designated loops.

    Returns exactly `len(progressive_loss_loops)` finite positive auxiliary losses.

    **Validates: Requirements 3.6**
    """
    torch.manual_seed(seed)
    model_dim = 16
    vocab_size = 32

    # Pick distinct loop indices within range
    all_candidates = list(range(1, num_loops))
    prog_loops = sorted(all_candidates[:num_prog_loops])

    group = _make_recurrent_group(
        num_shared_blocks=1,
        num_loops=num_loops,
        model_dim=model_dim,
        loop_signal_rank=4,
        progressive_loss_loops=prog_loops,
    )

    # Create a simple lm_head_fn that computes cross-entropy
    lm_head = nn.Linear(model_dim, vocab_size, bias=False)

    def lm_head_fn(x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = lm_head(x.reshape(-1, model_dim))
        return torch.nn.functional.cross_entropy(
            logits, targets.reshape(-1), reduction="mean"
        )

    batch_size, seq_len = 1, 4
    x = torch.randn(batch_size, seq_len, model_dim)
    x0 = torch.randn(batch_size, seq_len, model_dim)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    _, aux_losses = group(x, x0, lm_head_fn=lm_head_fn, targets=targets)

    assert len(aux_losses) == len(prog_loops), (
        f"Expected {len(prog_loops)} aux losses, got {len(aux_losses)}"
    )

    for i, loss in enumerate(aux_losses):
        assert torch.isfinite(loss), f"aux_loss[{i}] is not finite: {loss.item()}"
        assert loss.item() > 0, f"aux_loss[{i}] is not positive: {loss.item()}"
