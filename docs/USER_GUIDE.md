# User guide — SC-MEPLS Analysis Studio v1.4.0

## Installation

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Run `pip install -r requirements.txt` for a source checkout or `pip install .` for an installed package.
4. Start with `python main.py` or `scmepls-studio`.

## Recommended workflow

1. **Overview:** confirm software version, coordinate convention and scope.
2. **SC-MEPLS Simulation:** begin in Manual/configured mode. Set mass, gaps, mission/fault inputs and use **Advanced Physical Parameters** for actuator constants, controller/lock gains, interlock tolerances, phase timing, inertia and M1–M8 coordinates. Review physical-truth violations, detected interlocks and hard-lock confirmation separately.
3. **3D Digital Twin:** import a named-node GLB (preferred) or STL. Select units/axis conversion and validate the optional `.manifest.json`. Imported surface geometry alone does not change plant physics.
4. **Structural FEA:** select a closed structural GLB/STL and mandatory `.fea.json`. Run the validated geometry → tetrahedral mesh → mass/CG/inertia → geometry-coupled dynamics → distributed-load FEA workflow. Keep **Use geometry-derived mass/inertia and M1–M8 coordinates** enabled for the principal digital-twin path. Run mesh convergence before local stress claims.
5. After structural validation, the validated history becomes available as **Validated geometry-coupled history** in SC-MEPLS Simulation and the 3D viewer receives validated CG/M1–M8 locations.
6. Use **Supporting Analyses** only where relevant: explicit vibration preprocessing, reduced-order control comparison, physical energy accounting, evidence-based MCDA and optional AI calibration/external validation.
7. Use **File > Save Project** to preserve numerical parameters, geometry/manifests and viewer state. Opening a project restores those references but deliberately requires structural validation to be rerun before validated FEA/dynamics are reused.
8. Use batch export to preserve figures, result tables and metadata.

## Geometry manifest (`.manifest.json`)

The geometry manifest controls semantic actor binding and optional visual actuator motion. Multiple visual components may bind to one simulation module.

```json
{
  "schema_version": 2,
  "platform_origin_m": [0.0, 0.0, 0.0],
  "required_components": ["PLATFORM", "EM_M01"],
  "components": {
    "PLATFORM": {"role": "structure", "dynamic_group": "platform"},
    "EM_M01": {"role": "electromagnetic", "dynamic_group": "module", "simulation_module": 1},
    "PNEU_M01": {"role": "pneumatic", "dynamic_group": "module", "simulation_module": 1},
    "LOCK_L": {
      "role": "lock",
      "dynamic_group": "lock",
      "motion_axis": [1, 0, 0],
      "stroke_m": 0.02,
      "motion_signal": "lock_fraction"
    }
  }
}
```

`world` is stationary. `platform` and `module` follow the transporter rigid body. `lock` is conservatively treated as launchpad/world-side; transporter-side lock hardware should be declared `dynamic_group: "platform"`. No actuator stroke is inferred from component names.

## Structural manifest (`.fea.json`)

A quantitative structural run requires explicit physical mappings. Example skeleton:

```json
{
  "schema_version": 2,
  "structural_component": "PLATFORM",
  "mass_scope": "moving_assembly",
  "material": {
    "youngs_modulus_pa": 69000000000,
    "poisson_ratio": 0.33,
    "density_kg_m3": 2700,
    "yield_strength_pa": 240000000
  },
  "mesh": {"element_size_m": 0.025, "optimize": true},
  "maximum_snap_distance_m": 0.02,
  "mapping": {
    "module_points_m": [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
    "constraint_points_m": [[0,0,0],[0,0,0],[0,0,0]],
    "propulsion_point_m": [0,0,0],
    "wind_point_m": [0,0,0],
    "lock_points_m": [[0,0,0],[0,0,0]],
    "module_patch_radius_m": 0.025,
    "propulsion_patch_radius_m": 0.035,
    "wind_patch_radius_m": 0.04,
    "lock_patch_radius_m": 0.025
  }
}
```

Replace every placeholder coordinate with the actual body-frame physical point. The fallback auto-bound mapping exists only for software tests/preview; it is not a substitute for a validated manifest.

For `mass_scope: "platform_only"`, add explicit payload mass properties and support points:

```json
{
  "payload": {
    "mass_kg": 50.0,
    "centroid_m": [0.0, 0.0, 0.5],
    "inertia_centroid_kg_m2": [[1,0,0],[0,1,0],[0,0,1]]
  },
  "mapping": {
    "payload_support_points_m": [[0.2,0.2,0.3],[-0.2,0.2,0.3],[0.2,-0.2,0.3],[-0.2,-0.2,0.3]],
    "payload_patch_radius_m": 0.04
  }
}
```

## Structural result interpretation

The **Structural FEA** tab shows tetrahedral volume results, not the render-surface wireframe. Select a critical checkpoint, von Mises stress/displacement/safety factor and deformation scale. Export a JSON validation report or VTU field as required.

Check mesh quality, mapping snap distance, pre-FEA force residual, pre-FEA moment residual, numerical reaction residual and coarse/base/fine convergence. Large pre-FEA residuals indicate an incomplete/inconsistent load model even if the stiffness equation itself solves exactly.

The solver is homogeneous isotropic, small-strain, quasi-static linear elasticity. Do not use it for local joint/contact or heterogeneous-material claims without a higher-fidelity validated model.

## Vibration

For experimental accelerometer data choose preprocessing deliberately. `Remove mean` is a reasonable general default where gravity/DC projection is not part of the desired vibration metric; `Detrend` addresses slow linear drift and `High-pass` requires a justified cutoff. Report the preprocessing choice.

## Control response

Only controls applicable to the selected first-order, second-order or mass–damper filtered-PID model are displayed. The filtered PID avoids an ideal derivative, but saturation and rate limits are outside this supporting linear tab.

## Energy

Prefer net specific energy consumption `kWh/(tonne·km)`. The relative comparison option is `best SEC / system SEC`; it should not replace physical SEC in a paper.

## AI calibration

Use grouped validation for repeated samples from one run, chronological validation for ordered data and K-fold only when samples are appropriately exchangeable. Inspect possible leakage warnings. After training, use **Evaluate Untouched External CSV** for a genuinely separate dataset; retain both training and external dataset fingerprints in the record.
