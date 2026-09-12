# SC-MEPLS Analysis Studio

A modular **single-window PyQt6 application** for article-level analysis of a hybrid magnetic-levitation–mechanical rocket launchpad roll-out and locking system.

Current software version: **v1.1.6**.

## Included analyses

- RMS vibration analysis from imported acceleration time histories
- Peak, crest-factor, dominant-frequency, and normalized vibration comparison
- First-order, second-order, or mass–damper PID closed-loop step-response analysis
- Rise time, settling time, overshoot, steady-state error, IAE, and ITAE
- Net energy accounting and specific energy consumption in kWh/(tonne·km)
- Evidence-oriented radar/MCDA with a consistent “higher is better” direction
- Reduced-order SC-MEPLS roll-out, levitation, alignment, seating, load-transfer, locking, CG-shift, and fault simulation
- Module-level M1–M8 validation outputs for independent MATLAB/Python comparison
- External **GLB/STL 3D geometry import** in a dedicated PyVista/VTK digital-twin viewer
- Geometry manifest validation, component selection/isolation, threaded file parsing, mesh checks, and 3D screenshot metadata
- Optional AI/data-driven calibration with explicit feature selection and K-fold, chronological, or grouped validation
- 600–2400 dpi PNG export with provenance/version metadata JSON sidecars
- Batch export of current article figures and CSV result tables
- GitHub Actions CI on supported Python versions

## SC-MEPLS simulation integrity

The reduced-order model deliberately separates:

- **true physical gap** — used by the electromagnetic plant force law and physical safety checks;
- **raw sensor gap** — includes injected sensor faults;
- **estimated gap** — used by the controller/fault-tolerant reconstruction.

The roll-out sequence uses a maximum time step of 0.02 s and a complete validation run of at least 75 s. The mission sequence includes levitation, roll-out, alignment, pre-lock, load transfer/seating, and six-DOF reduced-order hard locking. Saved CSV histories expose per-module current, electromagnetic force, pneumatic pressure/force, health/effectiveness, and all three gap quantities.

## External 3D geometry

The **3D Digital Twin** tab imports `.glb` and `.stl` files instead of relying on procedural display geometry.

- GLB is preferred for assemblies because named scene nodes are retained as component IDs.
- STL is supported, but STL has no reliable unit metadata; select the source unit explicitly before import.
- The SC-MEPLS world frame is X = roll-out direction, Y = lateral direction, Z = vertical.
- In default `Auto` mode, GLB/glTF Y-up geometry is converted to SC-MEPLS Z-up. STL is kept as stored.
- Imported geometry is **visualization-only**. Loading a model does not modify mass, inertia, actuator positions, controller gains, or simulation equations.
- The viewer checks mesh topology/scale and automatically looks for `<geometry-name>.manifest.json`.

Example manifest:

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

For best component-level interaction, export GLB assemblies with names such as `PLATFORM`, `TRACK`, `EM_M01` ... `EM_M08`, `PNEU_M01` ... `PNEU_M08`, and `LOCK_*`.

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

The repository also runs the test suite through GitHub Actions on pushes and pull requests to `main`.

## Scientific integrity

The default data are illustrative. They verify the workflow and reproduce comparison-figure concepts without presenting assumed values as experimental results. Replace defaults with literature-derived values, independently validated simulations, or experimental measurements before making quantitative claims. Every publication export should retain its metadata sidecar.

AI calibration does not validate physics. Select input features explicitly, use grouped validation for repeated samples from one run, chronological validation for ordered data, and preserve a final external validation dataset when possible.

The plotting layer requests **Times New Roman** when installed and uses a serif fallback otherwise. No font file is distributed with this repository.

See `docs/SCIENTIFIC_BASIS.md`, `docs/DATA_FORMATS.md`, `docs/USER_GUIDE.md`, and `docs/LIMITATIONS.md`.
