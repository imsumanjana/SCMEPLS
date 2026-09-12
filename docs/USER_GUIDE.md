# User guide

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Run `pip install -r requirements.txt` for a source checkout, or `pip install .` for an installed package.
4. Start the application using `python main.py` from a source checkout or `scmepls-studio` after package installation.

## Recommended workflow

1. **Overview:** review the current software version, scientific scope, and validation limitations.
2. **Vibration:** import accelerometer/simulation data or generate illustrative data. Imported data require an explicit provenance choice. Time stamps must be approximately uniformly sampled for Welch PSD; resample irregular data before import. Imported uncertainty is treated as unspecified unless separately supplied.
3. **Control Response:** choose an equivalent or mass–damper PID model. Use identified parameters where available. Review rise time, settling time, overshoot, IAE, and ITAE.
4. **Energy:** enter a complete energy balance for each transport method. Recovered energy cannot exceed gross supplied energy in the current model. Prefer the physical specific-energy metric for scientific reporting.
5. **Radar / MCDA:** replace default scores with a documented evidence-based rubric. Keep every axis oriented so higher is better. Criterion weights must be finite and non-negative and are retained in export metadata.
6. **SC-MEPLS Simulation:** adjust the reduced-order model parameters and fault scenarios. The complete sequence requires at least 75 s and uses a maximum integration step of 0.02 s. Exported time history includes rigid-body states, true/measured/estimated M1–M8 gaps, module health/effectiveness, current, electromagnetic force, pneumatic pressure and pneumatic force.
7. **3D Digital Twin:** import `.glb` or `.stl` geometry. GLB is preferred for named multi-part assemblies. STL units must be supplied explicitly. The viewer automatically checks for `<geometry-name>.manifest.json`, supports component selection/isolation, and can export a screenshot plus metadata sidecar. Imported geometry is visualization-only and does not change the numerical model.
8. **AI Calibration:** import traceable data, select the target and feature columns explicitly, and select K-fold or chronological validation. If repeated samples belong to the same experiment/run, choose a group/run column so GroupKFold is used. Preserve a final external validation set whenever possible.
9. **Batch export:** use File > Export All Current Article Figures. The program creates 600 dpi PNGs, CSV result tables, and metadata JSON sidecars containing the current software version and analysis provenance.

## Geometry manifest

A geometry sidecar is optional for static viewing but recommended before dynamic binding. Example:

```json
{
  "schema_version": 1,
  "required_components": ["PLATFORM", "EM_M01"],
  "components": {
    "PLATFORM": {"role": "platform", "dynamic_group": "platform"},
    "EM_M01": {"role": "em_module", "simulation_module": 1}
  }
}
```

Each `simulation_module` value must be between 1 and 8 and may be bound only once.

## Article wording

For illustrative outputs, use: “The figures present normalized technology-level comparisons for feasibility assessment and do not represent direct experimental measurements.”

For simulation outputs, report all model equations, parameters, solver settings, validation metrics, software version, and limitations. For 3D figures, state explicitly that imported geometry is a visualization asset unless a separate structural/FEA workflow has been performed.
