import pytest

from scmepls_studio.models.energy import EnergyInput, calculate_energy_metrics
from scmepls_studio.models.rollout_sim import RolloutParameters, simulate_rollout


def test_rollout_rejects_incomplete_sequence():
    with pytest.raises(ValueError):
        simulate_rollout(RolloutParameters(end_time_s=30.0))


def test_rollout_rejects_out_of_range_gap():
    with pytest.raises(ValueError):
        simulate_rollout(RolloutParameters(target_gap_m=0.03))


def test_rollout_rejects_invalid_fault_index():
    with pytest.raises(ValueError):
        simulate_rollout(RolloutParameters(coil_fault_index=9))


def test_energy_recovery_cannot_exceed_gross_input():
    with pytest.raises(ValueError):
        calculate_energy_metrics(EnergyInput(1000.0, 50.0, 1.0, recovered_kwh=1.1))
