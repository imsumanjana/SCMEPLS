from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict


@dataclass(frozen=True)
class CalibrationResult:
    algorithm: str
    target: str
    feature_names: list[str]
    r2: float
    mae: float
    sample_count: int
    feature_importance: dict[str, float]
    model: object


def train_regressor(df: pd.DataFrame, target: str, algorithm: str = "Random Forest") -> CalibrationResult:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found.")
    numeric = df.select_dtypes(include=[np.number]).dropna(axis=0)
    if target not in numeric.columns:
        raise ValueError("Target must be numeric.")
    features = [c for c in numeric.columns if c != target]
    if not features:
        raise ValueError("At least one numeric feature column is required.")
    if len(numeric) < 10:
        raise ValueError("At least 10 complete rows are required; 20 or more are recommended.")
    X = numeric[features]
    y = numeric[target]
    if algorithm == "Gradient Boosting":
        model = GradientBoostingRegressor(random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2)
    splits = min(5, max(2, len(numeric) // 4))
    cv = KFold(n_splits=splits, shuffle=True, random_state=42)
    predicted = cross_val_predict(model, X, y, cv=cv)
    model.fit(X, y)
    importances = getattr(model, "feature_importances_", np.zeros(len(features)))
    return CalibrationResult(
        algorithm=algorithm,
        target=target,
        feature_names=features,
        r2=float(r2_score(y, predicted)),
        mae=float(mean_absolute_error(y, predicted)),
        sample_count=len(numeric),
        feature_importance={name: float(value) for name, value in zip(features, importances)},
        model=model,
    )


def save_calibration(result: CalibrationResult, path: str | Path) -> None:
    payload = {
        "algorithm": result.algorithm,
        "target": result.target,
        "feature_names": result.feature_names,
        "r2": result.r2,
        "mae": result.mae,
        "sample_count": result.sample_count,
        "model": result.model,
    }
    joblib.dump(payload, Path(path))
