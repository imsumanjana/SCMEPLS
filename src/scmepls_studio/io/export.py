from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from ..config import MIN_EXPORT_DPI


def export_figure(
    figure: Figure,
    path: str | Path,
    dpi: int = 600,
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    if dpi < MIN_EXPORT_DPI:
        raise ValueError(f"Export DPI must be at least {MIN_EXPORT_DPI}.")
    output = Path(path)
    if output.suffix.lower() != ".png":
        output = output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    sidecar = output.with_suffix(".metadata.json")
    payload = {
        "figure_file": output.name,
        "dpi": dpi,
        "exported_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": "Not specified",
        "scientific_status": "Engineering analysis; validation required",
    }
    if metadata:
        payload.update(metadata)
    sidecar.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output, sidecar
