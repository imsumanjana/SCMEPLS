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
- Optional AI/data-driven calibration with cross-validated R² and MAE
- 600–2400 dpi PNG export with a provenance metadata JSON sidecar
- Batch export of current article figures and CSV result tables

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
