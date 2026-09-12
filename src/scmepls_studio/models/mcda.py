from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Criterion:
    name: str
    weight: float = 1.0
    higher_is_better: bool = True


def validate_scores(scores: pd.DataFrame, criteria: list[Criterion]) -> None:
    names = [c.name for c in criteria]
    missing = [name for name in names if name not in scores.columns]
    if missing:
        raise ValueError(f"Missing criteria columns: {missing}")
    if "system" not in scores.columns:
        raise ValueError("Scores table must contain a 'system' column.")
    values = scores[names].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Scores contain non-finite values.")
    if any(not np.isfinite(c.weight) or c.weight < 0 for c in criteria):
        raise ValueError("Criterion weights must be finite and non-negative.")


def normalize_mcda(scores: pd.DataFrame, criteria: list[Criterion], scale_min: float = 1.0, scale_max: float = 5.0) -> pd.DataFrame:
    validate_scores(scores, criteria)
    out = scores[["system"]].copy()
    for c in criteria:
        x = scores[c.name].astype(float).to_numpy()
        xmin, xmax = float(np.min(x)), float(np.max(x))
        if np.isclose(xmax, xmin):
            normalized = np.full_like(x, 0.5, dtype=float)
        else:
            normalized = (x - xmin) / (xmax - xmin)
        if not c.higher_is_better:
            normalized = 1.0 - normalized
        out[c.name] = scale_min + normalized * (scale_max - scale_min)
    return out


def weighted_scores(normalized_scores: pd.DataFrame, criteria: list[Criterion]) -> pd.DataFrame:
    validate_scores(normalized_scores, criteria)
    weights = np.array([c.weight for c in criteria], dtype=float)
    if np.sum(weights) <= 0:
        raise ValueError("At least one criterion weight must be positive.")
    weights /= np.sum(weights)
    values = normalized_scores[[c.name for c in criteria]].to_numpy(dtype=float)
    result = normalized_scores.copy()
    result["weighted_score"] = values @ weights
    return result.sort_values("weighted_score", ascending=False).reset_index(drop=True)


def direct_score_table(raw: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    rows = []
    for system, criteria in raw.items():
        rows.append({"system": system, **criteria})
    return pd.DataFrame(rows)
