from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationItem:
    name: str
    status: str
    message: str


def validate_provenance(provenance: str) -> ValidationItem:
    allowed = {"Illustrative", "Literature-derived", "Simulation", "Experimental", "AI-calibrated"}
    if provenance not in allowed:
        return ValidationItem("Provenance", "FAIL", "Select a valid provenance category.")
    if provenance == "Illustrative":
        return ValidationItem("Provenance", "WARN", "Illustrative values must not be described as measured results.")
    return ValidationItem("Provenance", "PASS", f"Data marked as {provenance}.")
