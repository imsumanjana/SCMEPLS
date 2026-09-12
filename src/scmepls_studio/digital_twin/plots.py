from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from ..plotting.style import configure_matplotlib
from .binding import SceneBinding
from .physics import SimulationTimeline
from .results import ResultMetric

_METRIC_COLUMN = {
    "gap": "gap_{n}_m",
    "current": "coil_current_{n}_a",
    "em_force": "em_force_{n}_n",
    "pressure": "pressure_{n}_pa",
    "pneumatic_force": "pneumatic_force_{n}_n",
    "health": "health_{n}",
}
_METRIC_AXIS = {
    "gap": "Air gap (m)",
    "current": "Coil current (A)",
    "em_force": "EM force (N)",
    "pressure": "Pressure (Pa)",
    "pneumatic_force": "Pneumatic force (N)",
    "health": "Health (p.u.)",
}


def digital_twin_history_figure(
    timeline: SimulationTimeline,
    binding: SceneBinding | None,
    metric: ResultMetric = "em_force",
    current_time_s: float | None = None,
    font_size: int = 10,
) -> Figure:
    """Create a compact linked history plot for the 3D digital-twin tab."""
    configure_matplotlib(font_size)
    fig = Figure(figsize=(7.2, 4.2), constrained_layout=True)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    history = timeline.history
    time_s = history["time_s"].to_numpy(dtype=float)

    if binding is not None and binding.simulation_module is not None:
        module = binding.simulation_module
        column = _METRIC_COLUMN[metric].format(n=module)
        if column in history.columns:
            ax1.plot(time_s, history[column].to_numpy(dtype=float), linewidth=1.5, label=f"M{module} {metric.replace('_', ' ')}")
        ax1.set_ylabel(_METRIC_AXIS[metric])
        ax1.legend(frameon=True, loc="best")

        gap_columns = [
            (f"gap_{module}_m", "True gap"),
            (f"sensor_gap_{module}_m", "Sensor gap"),
            (f"estimated_gap_{module}_m", "Estimated gap"),
        ]
        for name, label in gap_columns:
            if name in history.columns:
                ax2.plot(time_s, history[name].to_numpy(dtype=float) * 1000.0, linewidth=1.1, label=label)
        ax2.set_ylabel("Gap (mm)")
        ax2.legend(frameon=True, loc="best")
    else:
        for name, label in (("x_m", "X"), ("y_m", "Y"), ("z_m", "Z")):
            if name in history.columns:
                ax1.plot(time_s, history[name].to_numpy(dtype=float), linewidth=1.3, label=label)
        ax1.set_ylabel("Position (m)")
        ax1.legend(frameon=True, loc="best")
        for name, label in (("roll_rad", "Roll"), ("pitch_rad", "Pitch"), ("yaw_rad", "Yaw")):
            if name in history.columns:
                ax2.plot(time_s, np.rad2deg(history[name].to_numpy(dtype=float)), linewidth=1.2, label=label)
        ax2.set_ylabel("Attitude (deg)")
        ax2.legend(frameon=True, loc="best")

    if current_time_s is not None and np.isfinite(current_time_s):
        for axis in (ax1, ax2):
            axis.axvline(float(current_time_s), linestyle="--", linewidth=0.9, color="black", label="Current time" if axis is ax1 else None)
    for axis in (ax1, ax2):
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.set_xlabel("Time (s)")
    return fig
