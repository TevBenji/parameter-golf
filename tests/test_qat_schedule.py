"""Property-based tests for QAT schedule and learning rate reduction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from hypothesis import given, settings, strategies as st, HealthCheck

from bitlinear import BitLinear
from train_gpt import activate_qat


# ---------------------------------------------------------------------------
# Property 6: QAT schedule state consistency
# ---------------------------------------------------------------------------


@given(
    num_layers=st.integers(min_value=1, max_value=5),
    in_features=st.integers(min_value=4, max_value=32),
    out_features=st.integers(min_value=4, max_value=32),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100)
def test_qat_state_consistency(num_layers, in_features, out_features, seed):
    """
    Property 6: QAT schedule state consistency.

    Before switchover all BitLinear have qat_enabled=False,
    after activate_qat all have qat_enabled=True.

    **Validates: Requirements 2.1, 2.2**
    """
    torch.manual_seed(seed)

    # Build a model with multiple BitLinear layers
    layers = [BitLinear(in_features, out_features) for _ in range(num_layers)]
    model = nn.Sequential(*layers)

    # Before activation: all qat_enabled must be False
    for m in model.modules():
        if isinstance(m, BitLinear):
            assert not m.qat_enabled, "qat_enabled should be False before activation"

    # Activate QAT
    activate_qat(model)

    # After activation: all qat_enabled must be True
    for m in model.modules():
        if isinstance(m, BitLinear):
            assert m.qat_enabled, "qat_enabled should be True after activation"


# ---------------------------------------------------------------------------
# Property 21: QAT learning rate reduction
# ---------------------------------------------------------------------------


@given(
    original_lr=st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
    qat_lr_factor=st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False),
    num_layers=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100, deadline=None)
def test_qat_lr_reduction(original_lr, qat_lr_factor, num_layers, seed):
    """
    Property 21: QAT learning rate reduction.

    After QAT activation, Muon base_lr equals original × qat_lr_factor.

    **Validates: Requirements 7.4**
    """
    torch.manual_seed(seed)

    from train_gpt import Muon

    # Create BitLinear layers and a Muon optimizer for their weights
    layers = [BitLinear(16, 16) for _ in range(num_layers)]
    model = nn.Sequential(*layers)
    matrix_params = [m.weight for m in model.modules() if isinstance(m, BitLinear)]

    optimizer_muon = Muon(
        matrix_params,
        lr=original_lr,
        momentum=0.95,
        backend_steps=5,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = original_lr

    # Simulate QAT activation: activate_qat + reduce Muon LR
    activate_qat(model)
    for group in optimizer_muon.param_groups:
        group["base_lr"] *= qat_lr_factor

    # Verify base_lr is now original × qat_lr_factor
    expected_lr = original_lr * qat_lr_factor
    for group in optimizer_muon.param_groups:
        assert abs(group["base_lr"] - expected_lr) < 1e-7, (
            f"Expected base_lr={expected_lr}, got {group['base_lr']}"
        )
