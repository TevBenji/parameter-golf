"""Property-based tests for model width expansion and hyperparameter validation."""

import sys
from pathlib import Path

# Add parent directory so we can import train_gpt directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from hypothesis import given, settings, strategies as st, HealthCheck

from train_gpt import MLP, GPT, CastedLinear


# ---------------------------------------------------------------------------
# Property 12: MLP hidden dimension equals model_dim × mlp_mult
# ---------------------------------------------------------------------------


@given(
    model_dim=st.sampled_from([64, 128, 256, 512, 768]),
    mlp_mult=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100)
def test_mlp_hidden_dimension(model_dim, mlp_mult, seed):
    """
    Property 12: MLP hidden dimension equals model_dim × mlp_mult.

    **Validates: Requirements 4.4**
    """
    torch.manual_seed(seed)
    mlp = MLP(model_dim, mlp_mult)
    expected_hidden = model_dim * mlp_mult

    # Check fc (input -> hidden) weight shape
    assert mlp.fc.weight.shape == (expected_hidden, model_dim), (
        f"fc weight shape {mlp.fc.weight.shape} != expected ({expected_hidden}, {model_dim})"
    )

    # Check proj (hidden -> output) weight shape
    assert mlp.proj.weight.shape == (model_dim, expected_hidden), (
        f"proj weight shape {mlp.proj.weight.shape} != expected ({model_dim}, {expected_hidden})"
    )

    # Verify output shape matches input dim
    x = torch.randn(1, 4, model_dim)
    out = mlp(x)
    assert out.shape == (1, 4, model_dim), (
        f"MLP output shape {out.shape} != expected (1, 4, {model_dim})"
    )


# ---------------------------------------------------------------------------
# Property 13: Head configuration validation
# ---------------------------------------------------------------------------


@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100)
def test_head_config_valid_combos(seed):
    """
    Property 13 (positive): Valid model_dim/num_heads combos with head_dim=64 succeed.

    **Validates: Requirements 4.2, 4.3, 4.5**
    """
    torch.manual_seed(seed)
    # model_dim=768, num_heads=12 → head_dim=64 ✓
    model = GPT(
        vocab_size=32,
        model_dim=768,
        num_heads=12,
        num_kv_heads=3,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        num_shared_blocks=1,
        num_loops=2,
        loop_signal_rank=4,
        progressive_loss_loops=[],
        progressive_loss_weight=0.3,
    )
    assert model.tok_emb.weight.shape[1] == 768


# Invalid combos: head_dim != 64
_INVALID_CONFIGS = [
    # model_dim not divisible by num_heads
    (768, 7, 1),
    # head_dim != 64 (768/12=64 is valid, but 768/8=96 is not)
    (768, 8, 2),
    # head_dim = 32 (512/16=32)
    (512, 16, 4),
    # head_dim = 128 (1024/8=128)
    (1024, 8, 2),
]


@pytest.mark.parametrize("model_dim,num_heads,num_kv_heads", _INVALID_CONFIGS)
def test_head_config_invalid_combos_raise(model_dim, num_heads, num_kv_heads):
    """
    Property 13 (negative): Invalid model_dim/num_heads combos raise ValueError.

    **Validates: Requirements 4.2, 4.3, 4.5**
    """
    with pytest.raises(ValueError):
        GPT(
            vocab_size=32,
            model_dim=model_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.005,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.5,
            num_shared_blocks=1,
            num_loops=2,
            loop_signal_rank=4,
            progressive_loss_loops=[],
            progressive_loss_weight=0.3,
        )
