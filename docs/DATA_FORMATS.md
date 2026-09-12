# Data formats — v1.4.0

## Vibration CSV

Required structure:

```csv
time_s,Crawler Transporter,Rail Platform,Maglev Platform
0.000,0.01,0.02,0.00
0.005,0.03,0.01,0.01
```

`time_s` must be strictly increasing and finite. Every other column is acceleration in m/s². Welch PSD requires approximately uniform sampling. The UI records the selected preprocessing mode (`raw`, `remove_mean`, `detrend`, or `highpass`) and high-pass cutoff when applicable.

## SC-MEPLS rollout CSV

The time history includes rigid-body truth/controller state, interlock state and module channels:

```text
time_s, mode, mode_name
x_m, y_m, z_m, vx_mps, vy_mps, vz_mps
roll_rad, pitch_rad, yaw_rad
roll_rate_rad_s, pitch_rate_rad_s, yaw_rate_rad_s
roll_accel_rad_s2, pitch_accel_rad_s2, yaw_accel_rad_s2
ax_mps2, ay_mps2, az_mps2
lock_fraction, lock_state, hard_lock_confirmed
docking_ready, detected_interlock, physical_truth_violation, unsafe_flag
propulsion_force_x_n, lateral_control_force_y_n
passive_force_x_n, passive_force_y_n, passive_force_z_n
lock_force_x_n, lock_force_y_n, lock_force_z_n
support_moment_x_nm, support_moment_y_nm
gravity_moment_x_nm, gravity_moment_y_nm, wind_moment_x_nm
yaw_control_moment_nm, lock_moment_x_nm, lock_moment_y_nm, lock_moment_z_nm
```

For every module `i=1...8`:

```text
module_x_i_m, module_y_i_m
gap_i_m                 true physical gap
sensor_gap_i_m          raw sensor channel
estimated_gap_i_m       controller estimate
health_i, coil_efficiency_i
coil_current_i_a
em_force_i_n
pressure_i_pa
pneumatic_force_i_n
```

`p_radps/q_radps/r_radps` are retained as aliases of the roll/pitch/yaw-rate fields for interoperability.

## Geometry manifest (`<geometry>.manifest.json`)

```json
{
  "schema_version": 2,
  "platform_origin_m": [0, 0, 0],
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

A `simulation_module` must be 1–8. **Multiple visual parts may bind to the same module**; this is intentional for assemblies such as EM and pneumatic components belonging to M1.

Supported dynamic groups are `world`, `platform`, `module`, and `lock`. `motion_axis` is normalized on import. A non-zero `stroke_m` requires an axis and supported motion signal.

## Structural manifest (`<geometry>.fea.json`)

A quantitative FEA run requires an explicit structural manifest. Required concepts are:

```text
schema_version
structural_component (required for multi-part visual GLB)
mass_scope = moving_assembly | platform_only
material: E, nu, density, yield strength
mesh: element size / optional min/max / optimization
maximum_snap_distance_m
mapping.module_points_m          shape 8×3
mapping.constraint_points_m      shape 3×3
mapping.propulsion_point_m       shape 3
mapping.wind_point_m             shape 3
mapping.lock_points_m            shape N×3
mapping.*_patch_radius_m
```

For `platform_only`, also provide:

```text
payload.mass_kg
payload.centroid_m                       shape 3
payload.inertia_centroid_kg_m2           shape 3×3
mapping.payload_support_points_m          shape N×3
mapping.payload_patch_radius_m
```

All structural coordinates are in the imported/normalized SC-MEPLS body frame and metres. Load points are mapped to finite boundary-node patches, not single nearest nodes.

## AI calibration CSV

Use one row per independent sample/run as appropriate. Example:

```csv
run_id,mass_kg,speed_mps,gap_mm,wind_n,controller_kp,rms_accel_mps2
1,75,0.05,10,0,90,0.11
```

The training dataset and external validation dataset must contain the selected feature names and target. The program reports SHA-256 fingerprints for traceability. Group/run columns are excluded from model features. Near-perfect feature/target correlation is flagged as possible leakage.

## Project JSON

`File > Save Project` stores:

- software version and complete rollout parameter set;
- requested dynamics source;
- structural geometry/manifest paths, units, axis mode and validation options;
- latest structural report snapshot where available;
- 3D geometry/manifest path, selected component, result metric and camera state.

On open, external files are reloaded/validated where appropriate. Structural meshes/results are intentionally not trusted from the saved JSON; rerun Structural Validation before reusing geometry-coupled FEA/dynamics.
