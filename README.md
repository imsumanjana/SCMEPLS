# SC-MEPLS Analysis Studio

A modular **single-window PyQt6 application** for article-level analysis of a hybrid magnetic-levitation–mechanical rocket launchpad roll-out and locking system.

## Included analyses

- RMS vibration analysis from imported acceleration time histories
- Peak, crest-factor, dominant-frequency, and normalized vibration comparison
- First-order, second-order, or mass–damper PID closed-loop step-response analysis
- Rise time, settling time, overshoot, steady-state error, IAE, and ITAE
- Net energy accounting and specific energy consumption in kWh/(tonne·km)
- Evidence-oriented radar/MCDA with a consistent “higher is better” direction
- Reduced-order SC-MEPLS roll-out, levitation, load-transfer, CG-shift, and fault simulation
- External **GLB/STL 3D geometry import** in a dedicated PyVista/VTK digital-twin viewer
- Component selection, isolation/full-assembly view, geometry-unit conversion, axis conversion, and import validation summary
- Optional AI/data-driven calibration with cross-validated R² and MAE
- 600–2400 dpi PNG export with a provenance metadata JSON sidecar
- Batch export of current article figures and CSV result tables

## External 3D geometry

The **3D Digital Twin** tab imports `.glb` and `.stl` files instead of relying on procedural/inbuilt display geometry.

- GLB is preferred for assemblies because named scene nodes are retained as component IDs.
- STL is supported for single-part or externally managed assemblies, but STL has no reliable unit metadata; select the source unit explicitly before import.
- The SC-MEPLS world frame is X = roll-out direction, Y = lateral direction, Z = vertical.
- In the default `Auto` axis mode, GLB/glTF Y-up geometry is converted to SC-MEPLS Z-up. STL is kept as stored.
- Imported geometry is **visualization-only** in v1.1.0. Loading a model does not modify mass, inertia, actuator positions, controller gains, or simulation equations. This preserves independent MATLAB/Python validation.

For best component-level interaction, export GLB assemblies with clear node names such as `PLATFORM`, `TRACK`, `EM_M01` ... `EM_M08`, `PNEU_M01` ... `PNEU_M08`, and `LOCK_*`.

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

The plotting layer requests **Times New Roman** when it is installed. On systems without it, a suitable serif fallback is used. No font file is distributed with this repository.

## Scientific integrity

The default data are illustrative. They are intended to verify the workflow and reproduce comparison-figure concepts without presenting assumed values as experimental results. Replace defaults with literature-derived values, validated simulation data, or experimental measurements before making quantitative claims. Every exported figure should retain its metadata sidecar.

See `docs/SCIENTIFIC_BASIS.md`, `docs/DATA_FORMATS.md`, `docs/USER_GUIDE.md`, and `docs/LIMITATIONS.md`.
