import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from spamsentinel_ml.evaluate import METRICS, classification_metrics, cross_validate, review_band_stats, summarise_cv
from spamsentinel_ml.models import MODEL_FACTORIES, make_model


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=300, n_features=6, n_informative=4, random_state=0)
    X = pd.DataFrame(X, columns=[f"x{i}" for i in range(6)])
    X.iloc[::17, 2] = np.nan  # imputer must cope with missing values
    return X, y


@pytest.mark.parametrize("name", list(MODEL_FACTORIES))
def test_models_fit_and_predict_proba(data, name):
    X, y = data
    model = make_model(name).fit(X, y)
    prob = model.predict_proba(X)
    assert prob.shape == (len(X), 2)
    assert np.all((prob >= 0) & (prob <= 1))
    assert (model.predict(X) == y).mean() > 0.7


def test_cross_validate_reports_every_metric(data):
    X, y = data
    folds = cross_validate("DecisionTree", X, y, folds=5, n_jobs=1)
    assert len(folds) == 5
    summary = summarise_cv(folds)
    for m in METRICS:
        assert f"{m}_mean" in summary and f"{m}_sd" in summary
        assert not np.isnan(summary[f"{m}_mean"])


def test_classification_metrics_definitions():
    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 0, 1, 0, 1]
    m = classification_metrics(y_true, y_pred, [0.9, 0.2, 0.8, 0.1, 0.6])
    assert m["tpr"] == pytest.approx(2 / 3)
    assert m["tnr"] == pytest.approx(1 / 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(3 / 5)


def test_review_band_stats():
    stats = review_band_stats([1, 0, 1, 0], [0.9, 0.5, 0.4, 0.1], (0.35, 0.65))
    assert stats["n_in_band"] == 2
    assert stats["accuracy_outside_band"] == 1.0
    assert stats["errors_caught_by_band"] == stats["errors_total"] == 2
