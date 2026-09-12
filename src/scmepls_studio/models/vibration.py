from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.signal import welch

MAX_RELATIVE_SAMPLE_JITTER = 0.02


@dataclass(frozen=True)
class VibrationMetrics:
    rms: float
    peak: float
    peak_to_peak: float
    crest_factor: float
    dominant_frequency_hz: float


def rms_acceleration(values: np.ndarray) -> float:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        raise ValueError("Acceleration array is empty.")
    if not np.isfinite(a).all():
        raise ValueError("Acceleration array contains non-finite values.")
    return float(np.sqrt(np.mean(np.square(a))))


def calculate_metrics(time_s: np.ndarray, acceleration_mps2: np.ndarray) -> VibrationMetrics:
    t = np.asarray(time_s, dtype=float)
    a = np.asarray(acceleration_mps2, dtype=float)
    if t.size != a.size or t.size < 4:
        raise ValueError("Time and acceleration arrays must have equal length and at least four samples.")
    if not np.isfinite(t).all() or not np.isfinite(a).all():
        raise ValueError("Time and acceleration arrays must contain only finite values.")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("Time values must be strictly increasing.")
    median_dt = float(np.median(dt))
    relative_jitter = float(np.max(np.abs(dt - median_dt)) / max(median_dt, 1e-15))
    if relative_jitter > MAX_RELATIVE_SAMPLE_JITTER:
        raise ValueError(
            "Welch PSD requires approximately uniform sampling. Resample the time history before import "
            f"(maximum relative sample-interval jitter {relative_jitter:.3%} exceeds {MAX_RELATIVE_SAMPLE_JITTER:.1%})."
        )
    rms = rms_acceleration(a)
    peak = float(np.max(np.abs(a)))
    crest = float(peak / rms) if rms > 0 else 0.0
    fs = float(1.0 / median_dt)
    frequencies, psd = welch(a - np.mean(a), fs=fs, nperseg=min(1024, a.size))
    dominant = float(frequencies[int(np.argmax(psd))]) if psd.size else 0.0
    return VibrationMetrics(
        rms=rms,
        peak=peak,
        peak_to_peak=float(np.ptp(a)),
        crest_factor=crest,
        dominant_frequency_hz=dominant,
    )


def normalize_rms(rms_values: Mapping[str, float], reference: str | None = None) -> dict[str, float]:
    if not rms_values:
        raise ValueError("At least one RMS value is required.")
    clean = {str(k): float(v) for k, v in rms_values.items()}
    if any(v < 0 or not np.isfinite(v) for v in clean.values()):
        raise ValueError("RMS values must be finite and non-negative.")
    if reference is None:
        reference_value = max(clean.values())
    else:
        if reference not in clean:
            raise KeyError(f"Unknown reference system: {reference}")
        reference_value = clean[reference]
    if reference_value <= 0:
        raise ValueError("Reference RMS value must be positive.")
    return {name: value / reference_value for name, value in clean.items()}


def generate_synthetic_timeseries(
    duration_s: float,
    sample_rate_hz: float,
    target_rms: Mapping[str, float],
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic illustrative acceleration histories scaled to target RMS.

    The signal contains low-frequency transport motion, structural harmonics, broadband noise,
    and sparse transient events. It is intended for software demonstration only.
    """
    if duration_s <= 0 or sample_rate_hz <= 1:
        raise ValueError("Duration and sample rate must be positive.")
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, 1.0 / sample_rate_hz)
    data: dict[str, np.ndarray] = {"time_s": t}
    base_freqs = {
        "Crawler Transporter": (1.2, 5.5, 11.0),
        "Rail Platform": (2.0, 8.0, 15.0),
        "Maglev Platform": (0.8, 3.5, 12.0),
    }
    for idx, (name, target) in enumerate(target_rms.items()):
        f1, f2, f3 = base_freqs.get(name, (1.0 + idx, 5.0 + idx, 10.0 + idx))
        signal = (
            0.65 * np.sin(2 * np.pi * f1 * t)
            + 0.25 * np.sin(2 * np.pi * f2 * t + 0.3)
            + 0.10 * np.sin(2 * np.pi * f3 * t + 1.1)
            + 0.12 * rng.normal(size=t.size)
        )
        if "Crawler" in name:
            for center in np.linspace(duration_s * 0.15, duration_s * 0.9, 7):
                signal += 0.55 * np.exp(-((t - center) / 0.025) ** 2)
        elif "Rail" in name:
            for center in np.linspace(duration_s * 0.2, duration_s * 0.85, 5):
                signal += 0.28 * np.exp(-((t - center) / 0.018) ** 2)
        current = rms_acceleration(signal)
        scaled = signal * (float(target) / current if current > 0 else 0.0)
        data[name] = scaled
    return pd.DataFrame(data)


def metrics_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "time_s" not in df.columns:
        raise ValueError("CSV must contain a 'time_s' column.")
    rows = []
    for column in df.columns:
        if column == "time_s":
            continue
        metrics = calculate_metrics(df["time_s"].to_numpy(), df[column].to_numpy())
        rows.append({"system": column, **metrics.__dict__})
    if not rows:
        raise ValueError("CSV must contain at least one acceleration column.")
    result = pd.DataFrame(rows)
    normalized = normalize_rms(dict(zip(result["system"], result["rms"])))
    result["normalized_rms"] = result["system"].map(normalized)
    return result
