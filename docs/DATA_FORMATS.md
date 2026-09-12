# Data formats

## Vibration CSV

Required structure:

```csv
time_s,Crawler Transporter,Rail Platform,Maglev Platform
0.000,0.01,0.02,0.00
0.005,0.03,0.01,0.01
```

- `time_s` must be strictly increasing and finite.
- Every other column is interpreted as acceleration in m/s² and must be numeric/finite.
- Welch PSD requires approximately uniform sampling. The software rejects time histories whose maximum sample-interval deviation exceeds 2% of the median interval; resample irregular measurements before import.
- The importer asks whether the CSV is Experimental, Simulation, or Literature-derived. It does not infer provenance from the filename.
- Imported uncertainty is marked unspecified unless supplied/calculated separately.

## SC-MEPLS rollout CSV

The saved time history contains aggregate rigid-body and support states plus module-level validation channels. Important groups include:

```text
time_s, mode, mode_name
x_m, y_m, z_m, vx_mps, vy_mps, vz_mps
roll_rad, pitch_rad, yaw_rad
roll_rate_rad_s, pitch_rate_rad_s, yaw_rate_rad_s
ax_mps2, ay_mps2, az_mps2
lock_fraction, docking_ready, unsafe_flag
```

For each module `i = 1...8`:

```text
gap_i_m                 true physical gap
sensor_gap_i_m          raw sensor channel
estimated_gap_i_m       controller/fault-tolerant estimate
health_i
coil_efficiency_i
coil_current_i_a
em_force_i_n
pressure_i_pa
pneumatic_force_i_n
```

These columns are intended for independent MATLAB/Python comparison and later GLB/STL animation binding.

## AI calibration CSV

Use one row per independent experiment or validated simulation run. Include numeric input features and one or more numeric target columns, for example:

```csv
run_id,mass_kg,speed_mps,gap_mm,wind_n,controller_kp,rms_accel_mps2,settling_time_s,net_energy_kwh
1,75,0.05,10,0,90,0.11,2.4,0.008
```

- Select one target at a time.
- Select input feature columns explicitly; do not include the target or quantities derived from it.
- For independent runs use KFold validation.
- For chronological/time-ordered data use Chronological validation.
- If several rows come from the same experiment/run, select a group/run column so GroupKFold is used and repeated samples cannot leak across folds.
- Preserve an external validation dataset for final claims whenever possible.

## External geometry

Supported files are `.glb` and `.stl`.

- SC-MEPLS world axes are X = roll-out, Y = lateral, Z = vertical.
- GLB/glTF convention is metres and Y-up. In Auto mode the viewer converts GLB Y-up to SC-MEPLS Z-up.
- STL is unitless, so source units (`m`, `cm`, or `mm`) must be selected explicitly.
- Named GLB nodes are preserved as component IDs after safe-name normalization.

Recommended GLB node names:

```text
TRACK
PLATFORM
ROCKET
EM_M01 ... EM_M08
PNEU_M01 ... PNEU_M08
LOCK_L
LOCK_R
```

### Geometry manifest sidecar

For `assembly.glb`, the automatic sidecar name is `assembly.manifest.json`.

```json
{
  "schema_version": 1,
  "required_components": ["PLATFORM", "EM_M01", "EM_M02"],
  "components": {
    "PLATFORM": {"role": "platform", "dynamic_group": "platform"},
    "EM_M01": {"role": "em_module", "simulation_module": 1},
    "EM_M02": {"role": "em_module", "simulation_module": 2}
  }
}
```

The manifest may only reference component IDs present in the imported geometry. Each `simulation_module` value must be 1–8 and may only be used once.
