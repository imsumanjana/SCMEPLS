from scmepls_studio.models.energy import EnergyInput, calculate_energy_metrics, compare_energy


def test_energy_balance():
    m = calculate_energy_metrics(EnergyInput(1000, 50, 8, 1, 1, 2))
    assert m.gross_energy_kwh == 10
    assert m.net_energy_kwh == 8
    assert abs(m.specific_energy_kwh_per_tonne_km - 0.16) < 1e-12


def test_energy_comparison():
    df = compare_energy({"A": EnergyInput(10, 100, 1), "B": EnergyInput(10, 100, 0.5)})
    assert len(df) == 2
    assert df.loc[df.system == "B", "normalized_energy_performance"].iloc[0] > df.loc[df.system == "A", "normalized_energy_performance"].iloc[0]
