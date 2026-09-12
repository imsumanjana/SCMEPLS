import json
import re

import numpy as np
import pandas as pd
import pytest

from scmepls_studio import __version__
from scmepls_studio.config import APP_VERSION, DEFAULT_PROJECT_PATH
from scmepls_studio.models.calibration import train_regressor
from scmepls_studio.models.mcda import Criterion, weighted_scores
from scmepls_studio.models.vibration import calculate_metrics


def test_version_and_packaged_default_resource_are_consistent():
    assert APP_VERSION == __version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)
    assert DEFAULT_PROJECT_PATH.exists()
    payload = json.loads(DEFAULT_PROJECT_PATH.read_text(encoding="utf-8"))
    assert payload["rollout_simulation"]["modules"] == 8


def test_irregular_vibration_sampling_is_rejected():
    t = np.array([0.0, 0.01, 0.02, 0.05, 0.06])
    a = np.sin(t)
    with pytest.raises(ValueError, match="uniform sampling"):
        calculate_metrics(t, a)


def test_negative_mcda_weight_is_rejected():
    scores = pd.DataFrame([{"system": "A", "x": 1.0}, {"system": "B", "x": 2.0}])
    with pytest.raises(ValueError, match="non-negative"):
        weighted_scores(scores, [Criterion("x", -1.0)])


def test_calibration_supports_explicit_features_and_chronological_validation():
    x = np.linspace(0.0, 1.0, 30)
    df = pd.DataFrame({"feature": x, "unused": x**2, "target": 2.0 * x + 0.1})
    result = train_regressor(
        df,
        "target",
        algorithm="Gradient Boosting",
        feature_names=["feature"],
        validation_mode="Chronological",
    )
    assert result.feature_names == ["feature"]
    assert result.validation_mode == "Chronological"
    assert result.validation_sample_count > 0
    assert len(result.data_fingerprint_sha256) == 64


def test_calibration_grouped_validation_keeps_groups_out_of_features():
    x = np.linspace(0.0, 1.0, 30)
    df = pd.DataFrame({
        "run_id": np.repeat(np.arange(6), 5),
        "feature": x,
        "target": 3.0 * x,
    })
    result = train_regressor(
        df,
        "target",
        algorithm="Gradient Boosting",
        feature_names=["feature"],
        group_column="run_id",
    )
    assert result.validation_mode == "Grouped KFold"
    assert result.group_column == "run_id"
