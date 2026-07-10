from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EnergyInput:
    mass_tonnes: float
    distance_m: float
    traction_kwh: float
    levitation_kwh: float = 0.0
    auxiliaries_kwh: float = 0.0
    recovered_kwh: float = 0.0


@dataclass(frozen=True)
class EnergyMetrics:
    gross_energy_kwh: float
    net_energy_kwh: float
    specific_energy_kwh_per_tonne_km: float
    recovery_fraction: float


def calculate_energy_metrics(values: EnergyInput) -> EnergyMetrics:
    if values.mass_tonnes <= 0 or values.distance_m <= 0:
        raise ValueError("Mass and distance must be positive.")
    components = [values.traction_kwh, values.levitation_kwh, values.auxiliaries_kwh, values.recovered_kwh]
    if any(v < 0 or not np.isfinite(v) for v in components):
        raise ValueError("Energy components must be finite and non-negative.")
    gross = values.traction_kwh + values.levitation_kwh + values.auxiliaries_kwh
    net = max(0.0, gross - values.recovered_kwh)
    tonne_km = values.mass_tonnes * values.distance_m / 1000.0
    sec = net / tonne_km
    recovery = values.recovered_kwh / gross if gross > 0 else 0.0
    return EnergyMetrics(gross, net, sec, recovery)


def compare_energy(systems: Mapping[str, EnergyInput]) -> pd.DataFrame:
    rows = []
    for name, values in systems.items():
        metrics = calculate_energy_metrics(values)
        rows.append({"system": name, **metrics.__dict__})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("At least one system is required.")
    worst = float(df["specific_energy_kwh_per_tonne_km"].max())
    if worst <= 0:
        df["normalized_energy_performance"] = 1.0
    else:
        df["normalized_energy_performance"] = 1.0 - df["specific_energy_kwh_per_tonne_km"] / worst
        df["normalized_energy_performance"] = 0.1 + 0.9 * df["normalized_energy_performance"]
    return df
