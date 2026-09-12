from __future__ import annotations

from pathlib import Path

from .version import __version__

APP_NAME = "SC-MEPLS Analysis Studio"
APP_VERSION = __version__
PACKAGE_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = PACKAGE_DIR / "resources"
DATA_DIR = RESOURCE_DIR
OUTPUT_DIR = Path.cwd() / "outputs"
DEFAULT_PROJECT_PATH = RESOURCE_DIR / "default_project.json"
MIN_EXPORT_DPI = 600
DEFAULT_EXPORT_DPI = 600

DISCLAIMER = (
    "Engineering analysis only. Results require traceable validation using "
    "experimental data, verified subsystem models, and independent safety review "
    "before safety-critical use."
)
