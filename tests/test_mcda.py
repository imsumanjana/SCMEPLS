import pandas as pd
from scmepls_studio.models.mcda import Criterion, weighted_scores


def test_weighted_scores():
    scores = pd.DataFrame([{"system": "A", "x": 5.0, "y": 1.0}, {"system": "B", "x": 3.0, "y": 3.0}])
    ranked = weighted_scores(scores, [Criterion("x", 2), Criterion("y", 1)])
    assert ranked.iloc[0].system == "A"
