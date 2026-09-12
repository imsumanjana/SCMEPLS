# SC-MEPLS Analysis Studio

A modular **single-window PyQt6 engineering application** for a swarm-coordinated modular electromagnetic–pneumatic roll-out and locking system for rocket ground transportation.

Current software version: **v1.4.0**.

## Main workflow

The top-level application is organized around the digital-twin workflow rather than treating every supporting comparison as an equal workspace:

1. **SC-MEPLS Simulation** — eight-module reduced-order rigid-body dynamics, fault scenarios and interlocked EM→pneumatic/mechanical load transfer.
2. **3D Digital Twin** — imported GLB/STL CAD, component binding, physical animation and linked module results.
3. **Structural FEA** — validated geometry → tetrahedral mesh → integrated mass/CG/inertia → geometry-coupled dynamics → distributed structural loads → linear-elastic node FEA.
4. **Supporting Analyses** — vibration, control-response, energy, radar/MCDA and optional AI calibration.

Projects can be saved/opened as JSON so model parameters, geometry references, manifests, display state and validation references remain traceable.

## SC-MEPLS dynamics integrity

The reduced-order model uses X = roll-out, Y = lateral and Z = vertical with a right-handed roll/pitch/yaw convention shared by the numerical model and 3D viewer. M1–M8 coordinates are model parameters and may be replaced by validated structural-manifest coordinates.

Three gap quantities remain distinct:

- `gap_i_m` — true physical gap used by the electromagnetic plant and physical truth checks;
- `sensor_gap_i_m` — raw measurement including injected sensor faults;
- `estimated_gap_i_m` — controller estimate. A failed channel is reconstructed from a rigid-plane fit to the healthy sensors rather than a position-blind median.

Mechanical handover now uses an interlocked lock state. Lock engagement cannot increase when the physical state is unsafe or the estimated-state interlock is active. Hard-lock control suppression occurs only after lock confirmation, not merely because a scheduled phase time has been reached. Docking permission includes position, vertical state, attitude, translational velocity and angular-rate limits.

The simulation exports translational states/accelerations, attitude/rates/angular accelerations, true/sensor/estimated gaps, M1–M8 coordinates, current, electromagnetic force, pneumatic pressure/force, health/effectiveness, control/passive/lock forces and generalized moments.

## Imported 3D geometry

The **3D Digital Twin** imports `.glb` and `.stl` files.

- GLB is preferred because named nodes and assembly structure can be retained.
- STL is unitless; source units must be selected explicitly.
- Auto mode converts GLB/glTF Y-up geometry to the SC-MEPLS Z-up frame.
- Surface geometry by itself is a visualization asset. It does **not** silently change dynamics.
- A validated Structural FEA run can explicitly supply the digital twin with integrated CG and physical M1–M8 locations.
- Geometry-manifest components may declare `motion_axis`, `stroke_m` and `motion_signal: "lock_fraction"` so lock/pneumatic parts move physically instead of using an opacity surrogate.

Example geometry manifest:

```json
{
  "schema_version": 2,
  "platform_origin_m": [0, 0, 0],
  "required_components": ["PLATFORM", "EM_M01"],
  "components": {
    "PLATFORM": {"role": "structure", "dynamic_group": "platform"},
    "EM_M01": {"role": "electromagnetic", "dynamic_group": "module", "simulation_module": 1},
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

Multiple visual components may legitimately bind to the same simulation module, for example `EM_M01` and `PNEU_M01` both mapping to module 1.

## Structural FEA

Quantitative FEA requires a separate `.fea.json` structural manifest. The structural path:

`closed GLB/STL body → Gmsh tetrahedralization → mass/CG/inertia integration → geometry-coupled SC-MEPLS history → distributed load patches → small-strain linear-elastic tetrahedral FEA`

The structural manifest defines material properties, meshing settings, eight module application points, propulsion/wind/lock locations, patch radii, numerical 3-2-1 constraint reference points and optional payload support locations. `platform_only` coupling requires explicit payload mass, CG, inertia and payload support points; stale total inertia is no longer retained.

Loads are distributed over boundary-node patches rather than concentrated at a single nearest node. Force and moment balance are checked before the stiffness solve, while the reaction residual remains a numerical equilibrium check. An optional coarse/base/fine convergence study reports changes in maximum displacement and von Mises stress.

The present solver is deliberately **homogeneous isotropic, first-order tetrahedral, small-strain and quasi-static**. It is not a substitute for nonlinear contact, multi-material joint/contact analysis, buckling, fatigue, plasticity, electromagnetic FEA, CFD or test correlation. For heterogeneous assemblies, treat a one-material model only as an explicitly documented equivalent-homogeneous screening model; use an external validated solver for local interface/joint claims.

## Supporting analyses

- **Vibration:** explicit Raw / Remove mean / Detrend / High-pass preprocessing is applied consistently before RMS, peak, crest factor and Welch PSD. Approximately uniform sampling is required.
- **Control response:** first-order, second-order or mass–damper filtered-PID linear comparison. Irrelevant fields are hidden. PID derivative roll-off is represented, but saturation/rate limits belong in the nonlinear SC-MEPLS model.
- **Energy:** physical net specific energy consumption in kWh/(tonne·km) is the preferred metric. The optional relative-efficiency value is simply `best_SEC / system_SEC`; the previous arbitrary 0.1-floor score has been removed.
- **Radar/MCDA:** 1–5 expert/evidence scores remain a supporting comparison and require a traceable scoring rubric.
- **AI calibration:** K-fold, chronological and grouped validation are available. RMSE and possible target-leakage warnings are reported, and a separate untouched external CSV can be evaluated before quantitative use.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

Installed-package workflow:

```bash
pip install .
scmepls-studio
```

## Tests

```bash
python -m pytest -q
```

GitHub Actions runs the test suite on supported Python versions for pull requests and pushes to `main`.

## Scientific integrity

Illustrative defaults verify software workflow only. Publication or PhD claims require traceable literature/identified parameters, validated geometry/material/load mappings, mesh convergence where local FEA stress is reported, independent numerical comparison and experimental or otherwise credible validation where available. Exported metadata should be retained with every figure/result set.

See `docs/SCIENTIFIC_BASIS.md`, `docs/DATA_FORMATS.md`, `docs/USER_GUIDE.md`, and `docs/LIMITATIONS.md`.
