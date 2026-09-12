from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Literal

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import welch

MAX_RELATIVE_SAMPLE_JITTER = 0.02
PreprocessingMode = Literal["raw", "remove_mean", "detrend", "highpass"]


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


def preprocess_acceleration(
    time_s: np.ndarray,
    acceleration_mps2: np.ndarray,
    mode: PreprocessingMode = "raw",
    *,
    highpass_cutoff_hz: float = 0.5,
) -> np.ndarray:
    """Apply an explicit, reproducible preprocessing choice before vibration metrics.

    Experimental accelerometer data can contain gravity projection, DC sensor bias,
    or slow drift. The previous implementation always included those components in
    RMS while subtracting the mean only for PSD. This function makes the scientific
    choice explicit and applies it consistently to RMS, peak, crest factor and PSD.
    """
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
    mode = str(mode).strip().lower()
    if mode == "raw":
        return a.copy()
    if mode == "remove_mean":
        return a - float(np.mean(a))
    if mode == "detrend":
        return np.asarray(signal.detrend(a, type="linear"), dtype=float)
    if mode == "highpass":
        fs = 1.0 / median_dt
        cutoff = float(highpass_cutoff_hz)
        if not np.isfinite(cutoff) or cutoff <= 0.0 or cutoff >= 0.45 * fs:
            raise ValueError(f"High-pass cutoff must be positive and below 45% of sample rate ({0.45*fs:.6g} Hz).")
        sos = signal.butter(4, cutoff, btype="highpass", fs=fs, output="sos")
        # filtfilt requires sufficient samples. The explicit error is preferable to
        # silently changing filter behavior for a short dataset.
        try:
            return np.asarray(signal.sosfiltfilt(sos, a), dtype=float)
        except ValueError as exc:
            raise ValueError("Time history is too short for zero-phase high-pass filtering.") from exc
    raise ValueError("Preprocessing mode must be raw, remove_mean, detrend, or highpass.")


def calculate_metrics(
    time_s: np.ndarray,
    acceleration_mps2: np.ndarray,
    preprocessing: PreprocessingMode = "raw",
    *,
    highpass_cutoff_hz: float = 0.5,
) -> VibrationMetrics:
    t = np.asarray(time_s, dtype=float)
    a = preprocess_acceleration(t, acceleration_mps2, preprocessing, highpass_cutoff_hz=highpass_cutoff_hz)
    dt = np.diff(t)
    median_dt = float(np.median(dt))
    rms = rms_acceleration(a)
    peak = float(np.max(np.abs(a)))
    crest = float(peak / rms) if rms > 0 else 0.0
    fs = float(1.0 / median_dt)
    frequencies, psd = welch(a - np.mean(a), fs=fs, nperseg=min(1024, a.size))
    dominant = float(frequencies[int(np.argmax(psd))]) if psd.size else 0.0
    return VibrationMetrics(rms=rms,peak=peak,peak_to_peak=float(np.ptp(a)),crest_factor=crest,dominant_frequency_hz=dominant)


def normalize_rms(rms_values: Mapping[str, float], reference: str | None = None) -> dict[str, float]:
    if not rms_values:
        raise ValueError("At least one RMS value is required.")
    clean = {str(k): float(v) for k, v in rms_values.items()}
    if any(v < 0 or not np.isfinite(v) for v in clean.values()):
        raise ValueError("RMS values must be finite and non-negative.")
    if reference is None:
        reference_value = max(clean.values())
    else:
        if reference not in clean: raise KeyError(f"Unknown reference system: {reference}")
        reference_value = clean[reference]
    if reference_value <= 0: raise ValueError("Reference RMS value must be positive.")
    return {name: value / reference_value for name, value in clean.items()}


def generate_synthetic_timeseries(duration_s: float,sample_rate_hz: float,target_rms: Mapping[str, float],seed: int = 42) -> pd.DataFrame:
    """Generate deterministic illustrative acceleration histories scaled to target RMS."""
    if duration_s <= 0 or sample_rate_hz <= 1: raise ValueError("Duration and sample rate must be positive.")
    rng = np.random.default_rng(seed); t = np.arange(0.0, duration_s, 1.0 / sample_rate_hz); data: dict[str, np.ndarray] = {"time_s": t}
    base_freqs = {"Crawler Transporter": (1.2,5.5,11.0),"Rail Platform": (2.0,8.0,15.0),"Maglev Platform": (0.8,3.5,12.0)}
    for idx,(name,target) in enumerate(target_rms.items()):
        f1,f2,f3 = base_freqs.get(name,(1.0+idx,5.0+idx,10.0+idx))
        wave = 0.65*np.sin(2*np.pi*f1*t)+0.25*np.sin(2*np.pi*f2*t+0.3)+0.10*np.sin(2*np.pi*f3*t+1.1)+0.12*rng.normal(size=t.size)
        if "Crawler" in name:
            for center in np.linspace(duration_s*0.15,duration_s*0.9,7): wave += 0.55*np.exp(-((t-center)/0.025)**2)
        elif "Rail" in name:
            for center in np.linspace(duration_s*0.2,duration_s*0.85,5): wave += 0.28*np.exp(-((t-center)/0.018)**2)
        current=rms_acceleration(wave); data[name]=wave*(float(target)/current if current>0 else 0.0)
    return pd.DataFrame(data)


def metrics_from_dataframe(
    df: pd.DataFrame,
    preprocessing: PreprocessingMode = "raw",
    *,
    highpass_cutoff_hz: float = 0.5,
) -> pd.DataFrame:
    if "time_s" not in df.columns: raise ValueError("CSV must contain a 'time_s' column.")
    rows=[]
    for column in df.columns:
        if column=="time_s": continue
        metrics=calculate_metrics(df["time_s"].to_numpy(),df[column].to_numpy(),preprocessing,highpass_cutoff_hz=highpass_cutoff_hz)
        rows.append({"system":column,**metrics.__dict__})
    if not rows: raise ValueError("CSV must contain at least one acceleration column.")
    result=pd.DataFrame(rows); normalized=normalize_rms(dict(zip(result["system"],result["rms"]))); result["normalized_rms"]=result["system"].map(normalized)
    result["preprocessing"] = preprocessing
    return result
