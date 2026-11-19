"""The four PRISMO classifiers (plus an optional gradient-boosting model).

WEKA -> scikit-learn equivalents used here:
  RandomForest (100 trees)          -> RandomForestClassifier(n_estimators=100)
  NaiveBayes (Gaussian estimator)   -> GaussianNB
  JRip (RIPPER rule learner)        -> wittgenstein.RIPPER
  J48 (C4.5, minNumObj=2, pruned)   -> DecisionTreeClassifier(entropy, min_samples_leaf=2,
                                       cost-complexity pruning)
Every model is wrapped in a Pipeline with a median imputer fitted on training data only.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import SEED


class RipperClassifier(ClassifierMixin, BaseEstimator):
    """scikit-learn wrapper around wittgenstein's RIPPER (JRip equivalent)."""

    def __init__(self, k: int = 2, prune_size: float = 0.33, n_discretize_bins: int = 10,
                 random_state: int = SEED):
        self.k = k
        self.prune_size = prune_size
        self.n_discretize_bins = n_discretize_bins
        self.random_state = random_state

    def fit(self, X, y):
        import wittgenstein as lw

        X = self._frame(X)
        y = np.asarray(y)
        self.classes_ = np.array([0, 1])
        self.model_ = lw.RIPPER(
            k=self.k,
            prune_size=self.prune_size,
            n_discretize_bins=self.n_discretize_bins,
            random_state=self.random_state,
            verbosity=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(X, pd.Series(y, name="label"), pos_class=1)
        return self

    def predict_proba(self, X):
        X = self._frame(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = np.asarray(self.model_.predict_proba(X), dtype=float)
        # wittgenstein returns columns ordered [neg, pos]
        return proba

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def rules(self) -> str:
        return str(self.model_.ruleset_)

    @staticmethod
    def _frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.reset_index(drop=True)
        X = np.asarray(X)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])


def _pipe(estimator) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", estimator)])


MODEL_FACTORIES: dict[str, Callable[[], Pipeline]] = {
    "RandomForest": lambda: _pipe(RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1)),
    "NaiveBayes": lambda: _pipe(GaussianNB()),
    "RIPPER": lambda: _pipe(RipperClassifier(random_state=SEED)),
    "DecisionTree": lambda: _pipe(
        DecisionTreeClassifier(criterion="entropy", min_samples_leaf=2, ccp_alpha=0.0005, random_state=SEED)
    ),
}

OPTIONAL_FACTORIES: dict[str, Callable[[], Pipeline]] = {
    "HistGradientBoosting": lambda: _pipe(HistGradientBoostingClassifier(random_state=SEED)),
}


def make_model(name: str) -> Pipeline:
    factories = {**MODEL_FACTORIES, **OPTIONAL_FACTORIES}
    if name not in factories:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(factories)}")
    return factories[name]()


def model_params(name: str) -> dict:
    """Readable hyper-parameters for the report (NFR9)."""
    est = make_model(name).named_steps["model"]
    return {k: v for k, v in est.get_params().items() if not callable(v)}
