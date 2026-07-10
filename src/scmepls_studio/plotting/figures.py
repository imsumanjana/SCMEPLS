from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .style import configure_matplotlib


def vibration_figure(metrics: pd.DataFrame, grayscale: bool = False, font_size: int = 12) -> Figure:
    configure_matplotlib(font_size)
    fig = Figure(figsize=(7.2, 4.5), constrained_layout=True)
    ax = fig.add_subplot(111)
    labels = metrics["system"].tolist()
    values = metrics["normalized_rms"].to_numpy(dtype=float)
    errors = metrics["uncertainty"].to_numpy(dtype=float) if "uncertainty" in metrics else None
    colors = ["0.20", "0.50", "0.78"] if grayscale else ["C0", "C1", "C2"]
    bars = ax.bar(labels, values, yerr=errors, capsize=4, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Normalized RMS Acceleration Index")
    ax.set_ylim(0, max(1.12, float(np.max(values + (errors if errors is not None else 0))) * 1.18))
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025, f"{value:.2f}", ha="center", va="bottom")
    ax.text(0.5, -0.18, "Lower value indicates lower transport-induced vibration.", transform=ax.transAxes, ha="center")
    return fig


def vibration_timeseries_figure(df: pd.DataFrame, font_size: int = 11) -> Figure:
    configure_matplotlib(font_size)
    fig = Figure(figsize=(7.2, 4.6), constrained_layout=True)
    ax = fig.add_subplot(111)
    for column in df.columns:
        if column != "time_s":
            ax.plot(df["time_s"], df[column], linewidth=1.0, label=column)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Acceleration (m/s$^2$)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=True)
    return fig


def step_response_figure(responses: Mapping[str, tuple[np.ndarray, np.ndarray]], font_size: int = 12) -> Figure:
    configure_matplotlib(font_size)
    fig = Figure(figsize=(7.2, 4.6), constrained_layout=True)
    ax = fig.add_subplot(111)
    for name, (t, y) in responses.items():
        ax.plot(t, y, linewidth=2.0, label=name)
    ax.axhline(1.0, linestyle="--", linewidth=0.8, color="black", label="Reference")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Position Response")
    ax.set_ylim(min(-0.05, ax.get_ylim()[0]), max(1.08, ax.get_ylim()[1]))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.55)
    ax.legend(frameon=True)
    return fig


def energy_figure(comparison: pd.DataFrame, mode: str = "specific", grayscale: bool = False, font_size: int = 12) -> Figure:
    configure_matplotlib(font_size)
    fig = Figure(figsize=(7.2, 4.5), constrained_layout=True)
    ax = fig.add_subplot(111)
    labels = comparison["system"].tolist()
    if mode == "performance":
        values = comparison["normalized_energy_performance"].to_numpy(dtype=float)
        ylabel = "Normalized Energy-Performance Index"
        footer = "Higher value indicates lower net specific energy consumption."
    else:
        values = comparison["specific_energy_kwh_per_tonne_km"].to_numpy(dtype=float)
        ylabel = r"Net Specific Energy (kWh tonne$^{-1}$ km$^{-1}$)"
        footer = "Lower value indicates lower net transport energy demand."
    # Keep the same categorical color order used in the vibration time-history plots:
    # Crawler = blue, Rail = orange, Maglev = green. Grayscale remains available for print-only use.
    colors = ["0.25", "0.52", "0.78"] if grayscale else ["C0", "C1", "C2"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    ymax = max(values) if len(values) else 1.0
    ax.set_ylim(0, ymax * 1.20 if ymax > 0 else 1.0)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.025, f"{value:.3g}", ha="center", va="bottom")
    ax.text(0.5, -0.18, footer, transform=ax.transAxes, ha="center")
    return fig


def radar_figure(scores: pd.DataFrame, criteria: list[str], font_size: int = 12) -> Figure:
    configure_matplotlib(font_size)
    n = len(criteria)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    fig = Figure(figsize=(8.6, 8.4), constrained_layout=True)
    ax = fig.add_subplot(111, polar=True)
    for _, row in scores.iterrows():
        values = [float(row[c]) for c in criteria]
        values += values[:1]
        ax.plot(angles, values, linewidth=2.0, label=str(row["system"]))
        ax.fill(angles, values, alpha=0.08)

    # Keep the actual radar circle at r=5; labels are placed outside with clip disabled.
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontweight="bold", fontsize=font_size + 1)
    ax.set_rlabel_position(10)

    # Place outer annotations manually beyond the r=5 circle so they do not overlap the radar grid.
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([""] * n)
    label_radius = 5.55
    for angle, label in zip(angles[:-1], criteria):
        # Keep labels outside the circle and align according to quadrant.
        c = np.cos(angle)
        if c > 0.25:
            ha = "left"
        elif c < -0.25:
            ha = "right"
        else:
            ha = "center"
        ax.text(
            angle,
            label_radius,
            label.replace(" ", "\n"),
            ha=ha,
            va="center",
            fontsize=font_size + 3,
            fontweight="bold",
            clip_on=False,
        )

    for tick in ax.yaxis.get_ticklabels():
        tick.set_fontweight("bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True, prop={"weight": "bold", "size": font_size})
    return fig


def rollout_dashboard_figure(df: pd.DataFrame, design_weight_n: float, font_size: int = 10) -> Figure:
    configure_matplotlib(font_size)
    fig = Figure(figsize=(10.5, 7.2), constrained_layout=True)
    axes = fig.subplots(3, 2)
    ax = axes[0, 0]
    ax.plot(df["time_s"], df["x_m"], label="Position")
    ax.plot(df["time_s"], df["x_ref_m"], "--", label="Reference")
    ax.set_ylabel("Position (m)"); ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.4); ax.legend()
    ax = axes[0, 1]
    ax.plot(df["time_s"], df["mean_gap_m"] * 1000, label="Mean gap")
    ax.fill_between(df["time_s"], df["min_gap_m"] * 1000, df["max_gap_m"] * 1000, alpha=0.15, label="Module range")
    ax.plot(df["time_s"], df["gap_ref_m"] * 1000, "--", label="Reference")
    ax.set_ylabel("Gap (mm)"); ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.4); ax.legend()
    ax = axes[1, 0]
    ax.plot(df["time_s"], np.rad2deg(df["roll_rad"]), label="Roll")
    ax.plot(df["time_s"], np.rad2deg(df["pitch_rad"]), label="Pitch")
    ax.plot(df["time_s"], np.rad2deg(df["yaw_rad"]), label="Yaw")
    ax.set_ylabel("Angle (deg)"); ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.4); ax.legend()
    ax = axes[1, 1]
    ax.plot(df["time_s"], df["em_force_n"], label="Electromagnetic")
    ax.plot(df["time_s"], df["pneumatic_force_n"], label="Pneumatic/mechanical")
    ax.plot(df["time_s"], df["total_support_force_n"], "--", label="Total support")
    ax.axhline(design_weight_n, linestyle=":", color="black", label="Weight")
    ax.set_ylabel("Force (N)"); ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.4); ax.legend(loc="center left", bbox_to_anchor=(0.05, 0.42), ncol=1, frameon=True, framealpha=0.94, facecolor="white", edgecolor="black")
    ax = axes[2, 0]
    ax.plot(df["time_s"], df["vibration_rms_proxy"], label="RMS proxy")
    ax.plot(df["time_s"], df["fault_severity"], label="Fault severity")
    ax.set_ylabel("Normalized magnitude"); ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.4); ax.legend()
    ax = axes[2, 1]
    ax.plot(df["time_s"], df["lock_fraction"], label="Lock fraction")
    ax.step(df["time_s"], df["unsafe_flag"], where="post", label="Unsafe flag")
    ax.set_ylabel("State") ; ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.4); ax.legend()
    return fig


def feature_importance_figure(importances: Mapping[str, float], font_size: int = 11) -> Figure:
    configure_matplotlib(font_size)
    items = sorted(importances.items(), key=lambda item: item[1])
    fig = Figure(figsize=(7.0, 4.5), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.barh([x[0] for x in items], [x[1] for x in items], edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Feature importance")
    ax.grid(axis="x", linestyle="--", alpha=0.45)
    return fig
