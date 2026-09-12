from __future__ import annotations

from pathlib import Path

APP_NAME = "SC-MEPLS Analysis Studio"
APP_VERSION = "1.1.3"
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
DEFAULT_PROJECT_PATH = DATA_DIR / "default_project.json"
MIN_EXPORT_DPI = 600
DEFAULT_EXPORT_DPI = 600

DISCLAIMER = (
    "Engineering analysis only. Results require traceable validation using "
    "experimental data, verified subsystem models, and independent safety review "
    "before safety-critical use."
)
