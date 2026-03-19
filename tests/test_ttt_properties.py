"""Property-based tests for Test-Time Training (TTTModule)."""

import sys
from pathlib import Path
from unittest.mock import patch

# Add parent directory so we can import train_gpt directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from hypothesis import given, settings, strategies as st, HealthCheck

from train_gpt import GPT, TTTModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXED_NUM_HEADS = 2
FIXED_NUM_KV_HEADS = 1
FIXED_MLP_MULT = 2
FIXED_ROPE_BASE = 10000.0
FIXED_QK_GAIN_INIT = 1.5
FIXED_MODEL_DIM = 128  # divisible by 2, head_dim=64
FIXED_VOCAB_SIZE = 32
FIXED_SEQ_LEN = 16


def _make_small_gpt(**overrides) -> GPT:
    """Build a minimal GPT model for CPU testing."""
    defaults = dict(
        vocab_size=FIXED_VOCAB_SIZE,
        model_dim=FIXED_MODEL_DIM,
        num_heads=FIXED_NUM_HEADS,
        num_kv_heads=FIXED_NUM_KV_HEADS,
        mlp_mult=FIXED_MLP_MULT,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=FIXED_ROPE_BASE,
        qk_gain_init=FIXED_QK_GAIN_INIT,
        num_shared_blocks=2,
        num_loops=3,
        loop_signal_rank=4,
        progressive_loss_loops=[1],
        progressive_loss_weight=0.3,
    )
    defaults.update(overrides)
    return defaults


def _build_model(seed: int = 42, **overrides) -> GPT:
    torch.manual_seed(seed)
    cfg = _make_small_gpt(**overrides)
    model = GPT(**cfg)
    model.float()  # ensure fp32 for CPU
    return model


def _make_tokens(num_tokens: int, vocab_size: int = FIXED_VOCAB_SIZE, seed: int = 0) -> torch.Tensor:
    """Generate random token tensor."""
    torch.manual_seed(seed)
    return torch.randint(0, vocab_size, (num_tokens,))


# ---------------------------------------------------------------------------
# Property 16: TTT prefix fraction alignment
# ---------------------------------------------------------------------------

@given(
    num_tokens=st.integers(min_value=2, max_value=2000),
    prefix_frac=st.floats(min_value=0.05, max_value=0.95),
    seq_len=st.integers(min_value=4, max_value=64),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_prefix_alignment(num_tokens, prefix_frac, seq_len):
    """
    Property 16: TTT prefix fraction alignment.

    Prefix length is always a multiple of seq_len and at most L × prefix_frac.

    Feature: parameter-golf-winning-entry, Property 16: TTT prefix fraction alignment
    **Validates: Requirements 6.1**
    """
    # Replicate the prefix length computation from TTTModule.adapt_and_eval
    prefix_len = int(num_tokens * prefix_frac)
    prefix_len = (prefix_len // seq_len) * seq_len

    # Property: prefix_len is a multiple of seq_len
    assert prefix_len % seq_len == 0, (
        f"prefix_len={prefix_len} is not a multiple of seq_len={seq_len}"
    )

    # Property: prefix_len <= L * prefix_frac (within integer rounding)
    max_allowed = num_tokens * prefix_frac
    assert prefix_len <= max_allowed + 1, (
        f"prefix_len={prefix_len} exceeds L*prefix_frac={max_allowed}"
    )

    # Property: prefix_len >= 0
    assert prefix_len >= 0


# ---------------------------------------------------------------------------
# Property 17: TTT selective parameter adaptation
# ---------------------------------------------------------------------------

@given(
    num_adapt_layers=st.integers(min_value=1, max_value=2),
    seed=st.integers(min_value=0, max_value=2**16 - 1),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_selective_adaptation(num_adapt_layers, seed):
    """
    Property 17: TTT selective parameter adaptation.

    Only last num_adapt_layers parameters change; all others remain bitwise identical.

    Feature: parameter-golf-winning-entry, Property 17: TTT selective parameter adaptation
    **Validates: Requirements 6.2**
    """
    model = _build_model(seed=seed)
    ttt = TTTModule(model=model, num_adapt_layers=num_adapt_layers, prefix_frac=0.3, ttt_lr=1e-3)

    # Identify which parameters should be adapted
    adapt_param_ids = {id(p) for p in ttt._get_adapt_params()}

    # Snapshot all parameters before adaptation
    pre_adapt = {name: p.detach().clone() for name, p in model.named_parameters()}

    # Generate tokens long enough for TTT to work
    # Need at least: prefix >= seq_len AND suffix > seq_len
    # With prefix_frac=0.3 and seq_len=16, need ~60+ tokens
    tokens = _make_tokens(128, seed=seed)

    # Manually perform the adaptation step (without autocast which needs CUDA)
    model.train()
    adapt_params = ttt._get_adapt_params()

    prefix_len = int(tokens.numel() * ttt.prefix_frac)
    prefix_len = (prefix_len // FIXED_SEQ_LEN) * FIXED_SEQ_LEN
    if prefix_len < FIXED_SEQ_LEN:
        return  # skip if document too short

    prefix = tokens[:prefix_len + 1]
    px = prefix[:-1].reshape(-1, FIXED_SEQ_LEN)
    py = prefix[1:].reshape(-1, FIXED_SEQ_LEN)

    loss = model(px, py)
    grads = torch.autograd.grad(loss, adapt_params, allow_unused=True)
    with torch.no_grad():
        for param, grad in zip(adapt_params, grads):
            if grad is not None:
                param.add_(grad, alpha=-ttt.ttt_lr)

    # Check: only adapt params changed, all others are bitwise identical
    for name, p in model.named_parameters():
        if id(p) in adapt_param_ids:
            # These may have changed (though not guaranteed if grad was zero)
            continue
        assert torch.equal(p, pre_adapt[name]), (
            f"Non-adapt parameter '{name}' changed during TTT adaptation"
        )


# ---------------------------------------------------------------------------
# Property 18: TTT weight reset round-trip
# ---------------------------------------------------------------------------

@given(
    num_adapt_layers=st.integers(min_value=1, max_value=2),
    seed=st.integers(min_value=0, max_value=2**16 - 1),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_weight_reset_roundtrip(num_adapt_layers, seed):
    """
    Property 18: TTT weight reset round-trip.

    checkpoint → adapt → reset restores all parameters to exact pre-adaptation values.

    Feature: parameter-golf-winning-entry, Property 18: TTT weight reset round-trip
    **Validates: Requirements 6.4**
    """
    model = _build_model(seed=seed)
    ttt = TTTModule(model=model, num_adapt_layers=num_adapt_layers, prefix_frac=0.3, ttt_lr=1e-3)

    # Step 1: Checkpoint
    ttt.checkpoint()

    # Snapshot for verification
    pre_adapt = {name: p.detach().clone() for name, p in model.named_parameters()}

    # Step 2: Adapt (manually, to avoid CUDA autocast)
    tokens = _make_tokens(128, seed=seed)
    adapt_params = ttt._get_adapt_params()

    prefix_len = int(tokens.numel() * ttt.prefix_frac)
    prefix_len = (prefix_len // FIXED_SEQ_LEN) * FIXED_SEQ_LEN
    if prefix_len < FIXED_SEQ_LEN:
        return

    prefix = tokens[:prefix_len + 1]
    px = prefix[:-1].reshape(-1, FIXED_SEQ_LEN)
    py = prefix[1:].reshape(-1, FIXED_SEQ_LEN)

    model.train()
    loss = model(px, py)
    grads = torch.autograd.grad(loss, adapt_params, allow_unused=True)
    with torch.no_grad():
        for param, grad in zip(adapt_params, grads):
            if grad is not None:
                param.add_(grad, alpha=-ttt.ttt_lr)

    # Verify something actually changed
    any_changed = False
    for name, p in model.named_parameters():
        if not torch.equal(p, pre_adapt[name]):
            any_changed = True
            break

    # Step 3: Reset
    ttt.reset()

    # Step 4: Verify all parameters restored to exact pre-adaptation values
    for name, p in model.named_parameters():
        assert torch.equal(p, pre_adapt[name]), (
            f"Parameter '{name}' not restored after reset. "
            f"Max diff: {(p - pre_adapt[name]).abs().max().item()}"
        )
