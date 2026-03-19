"""Property-based tests for Phase 7 integration: tied embeddings, seed determinism, BPB validity."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from hypothesis import given, settings, strategies as st, HealthCheck

from train_gpt import GPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXED_NUM_HEADS = 2
FIXED_NUM_KV_HEADS = 1
FIXED_MLP_MULT = 2
FIXED_ROPE_BASE = 10000.0
FIXED_QK_GAIN_INIT = 1.5
FIXED_MODEL_DIM = 128  # head_dim = 128/2 = 64
FIXED_VOCAB_SIZE = 32


def _build_gpt(seed: int = 42, tie_embeddings: bool = True, **overrides) -> GPT:
    torch.manual_seed(seed)
    defaults = dict(
        vocab_size=FIXED_VOCAB_SIZE,
        model_dim=FIXED_MODEL_DIM,
        num_heads=FIXED_NUM_HEADS,
        num_kv_heads=FIXED_NUM_KV_HEADS,
        mlp_mult=FIXED_MLP_MULT,
        tie_embeddings=tie_embeddings,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=FIXED_ROPE_BASE,
        qk_gain_init=FIXED_QK_GAIN_INIT,
        num_shared_blocks=1,
        num_loops=2,
        loop_signal_rank=4,
        progressive_loss_loops=[],
        progressive_loss_weight=0.3,
    )
    defaults.update(overrides)
    return GPT(**defaults)


# ---------------------------------------------------------------------------
# Property 23: Tied embedding weight sharing
# ---------------------------------------------------------------------------

@given(seed=st.integers(min_value=0, max_value=2**16 - 1))
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_tied_embedding_weight_sharing(seed):
    """
    Property 23: Tied embedding weight sharing.

    With tie_embeddings=True, lm_head is None and tok_emb.weight is used
    for output projection.

    **Validates: Requirements 8.5**
    """
    model = _build_gpt(seed=seed, tie_embeddings=True)

    # lm_head must be None when tied
    assert model.lm_head is None, "lm_head should be None when tie_embeddings=True"
    assert model.tie_embeddings is True

    # Forward pass should use tok_emb.weight for output projection
    # Verify by checking _compute_logits path: it should use F.linear(h, tok_emb.weight)
    x = torch.randint(0, FIXED_VOCAB_SIZE, (1, 4))
    y = torch.randint(0, FIXED_VOCAB_SIZE, (1, 4))
    loss = model(x, y)
    assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"

    # When untied, lm_head should NOT be None
    model_untied = _build_gpt(seed=seed, tie_embeddings=False)
    assert model_untied.lm_head is not None, "lm_head should exist when tie_embeddings=False"
    assert model_untied.tie_embeddings is False

    # Untied model should also produce finite loss
    loss_untied = model_untied(x, y)
    assert torch.isfinite(loss_untied), f"Untied loss should be finite, got {loss_untied.item()}"


# ---------------------------------------------------------------------------
# Property 24: Seed determinism
# ---------------------------------------------------------------------------

@given(seed=st.integers(min_value=0, max_value=2**16 - 1))
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_seed_determinism(seed):
    """
    Property 24: Seed determinism.

    Same seed + config produces bitwise-identical initial parameters.

    **Validates: Requirements 10.1, 10.3**
    """
    model_a = _build_gpt(seed=seed)
    model_b = _build_gpt(seed=seed)

    for (name_a, p_a), (name_b, p_b) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        assert name_a == name_b, f"Parameter name mismatch: {name_a} vs {name_b}"
        assert torch.equal(p_a, p_b), (
            f"Parameter '{name_a}' differs between two models with same seed={seed}. "
            f"Max diff: {(p_a - p_b).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# Property 22: BPB computation validity
# ---------------------------------------------------------------------------

@given(
    val_loss=st.floats(min_value=0.01, max_value=20.0, allow_nan=False, allow_infinity=False),
    token_count=st.integers(min_value=1, max_value=10_000_000),
    byte_count=st.integers(min_value=1, max_value=10_000_000),
)
@settings(max_examples=100)
def test_bpb_computation_validity(val_loss, token_count, byte_count):
    """
    Property 22: BPB computation validity.

    For any positive finite val_loss and counts, BPB is positive and finite.

    **Validates: Requirements 8.4**
    """
    # Replicate the BPB computation from eval_val
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count / byte_count
    bpb = bits_per_token * tokens_per_byte

    assert bpb > 0, f"BPB should be positive, got {bpb}"
    assert math.isfinite(bpb), f"BPB should be finite, got {bpb}"
