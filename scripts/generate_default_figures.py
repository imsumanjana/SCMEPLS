from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from scmepls_studio.io.export import export_figure
from scmepls_studio.models.control_response import ResponseParameters, calculate_response_metrics, simulate_step
from scmepls_studio.models.energy import EnergyInput, compare_energy
from scmepls_studio.models.mcda import Criterion, weighted_scores
from scmepls_studio.models.rollout_sim import RolloutParameters, simulate_rollout
from scmepls_studio.models.vibration import generate_synthetic_timeseries, metrics_from_dataframe
from scmepls_studio.plotting.figures import energy_figure, radar_figure, rollout_dashboard_figure, step_response_figure, vibration_figure


def main() -> None:
    out = ROOT / "outputs" / "default_article_figures"
    out.mkdir(parents=True, exist_ok=True)

    vib = generate_synthetic_timeseries(20, 200, {"Crawler Transporter": 0.9, "Rail Platform": 0.6, "Maglev Platform": 0.1})
    vib.to_csv(ROOT / "data" / "sample_vibration_timeseries.csv", index=False)
    vm = metrics_from_dataframe(vib); vm["uncertainty"] = 0.0
    export_figure(vibration_figure(vm), out / "Fig_vibration_normalized_RMS.png", 600, {"provenance": "Illustrative"})
    vm.to_csv(out / "vibration_metrics.csv", index=False)

    responses = {}
    control_rows = []
    for name, params in {
        "Maglev System": ResponseParameters(model_type="first_order", time_constant_s=0.8),
        "Conventional System": ResponseParameters(model_type="first_order", time_constant_s=2.8),
    }.items():
        t, y = simulate_step(params, 10, 2000)
        responses[name] = (t, y)
        control_rows.append({"system": name, **calculate_response_metrics(t, y).__dict__})
    export_figure(step_response_figure(responses), out / "Fig_closed_loop_step_response.png", 600, {"provenance": "Simulation"})
    pd.DataFrame(control_rows).to_csv(out / "control_metrics.csv", index=False)

    energy = compare_energy({
        "Crawler Transporter": EnergyInput(1000, 50, 8.5, 0, 1.5, 0.2),
        "Rail Platform": EnergyInput(1000, 50, 6.5, 0, 1.0, 0.5),
        "Maglev Platform": EnergyInput(1000, 50, 3.8, 1.8, 0.7, 0.8),
    })
    export_figure(energy_figure(energy, "specific"), out / "Fig_energy_specific.png", 600, {"provenance": "Illustrative"})
    energy.to_csv(out / "energy_metrics.csv", index=False)

    criteria = [
        "Vibration reduction", "Positioning accuracy", "Maintainability", "Energy performance",
        "Cost advantage", "Low complexity", "Load capability", "Fail-safe robustness", "Environmental robustness",
    ]
    scores = pd.DataFrame([
        {"system": "Crawler Transporter", **dict(zip(criteria, [2,2,2,2.5,4,4,5,5,4.5]))},
        {"system": "Rail Platform", **dict(zip(criteria, [3,3,3,3.5,3.5,3.5,4.5,4.5,4]))},
        {"system": "Maglev Platform", **dict(zip(criteria, [5,5,4,4.2,2,2,3.5,3,2.8]))},
    ])
    export_figure(radar_figure(scores, criteria), out / "Fig_multicriteria_radar.png", 600, {"provenance": "Illustrative"})
    ranked = weighted_scores(scores, [Criterion(c, 1, True) for c in criteria])
    ranked.to_csv(out / "radar_ranking.csv", index=False)

    history, metrics = simulate_rollout(RolloutParameters())
    export_figure(rollout_dashboard_figure(history, metrics["design_weight_n"]), out / "Fig_SCMEPLS_rollout_dashboard.png", 600, {"provenance": "Simulation"})
    history.to_csv(out / "scmepls_time_history.csv", index=False)
    pd.DataFrame([{k: v for k, v in metrics.items() if k != "parameters"}]).to_csv(out / "scmepls_metrics.csv", index=False)


if __name__ == "__main__":
    main()
