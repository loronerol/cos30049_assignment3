"""Wrapper fitness used by GA, PSO and SA to score a candidate feature subset."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from ..config import FitnessConfig


class FitnessEvaluator:
    """fitness(S) = mean k-fold macro-F1 of a Random Forest on S - alpha * |S| / N.

    The folds are fixed once, so every subset is scored on identical splits,
    and results are cached by mask so repeated subsets cost nothing.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, cfg: FitnessConfig, seed: int):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y)
        self.cfg = cfg
        self.seed = seed
        self.n_features = self.X.shape[1]
        skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=seed)
        self.folds = list(skf.split(self.X, self.y))
        self._cache: dict[bytes, tuple[float, float]] = {}
        self.n_evals = 0
        self.cache_hits = 0

    def score(self, mask: np.ndarray) -> tuple[float, float]:
        """Return (fitness, macro_f1) for a boolean mask."""
        mask = np.asarray(mask, dtype=bool)
        key = np.packbits(mask).tobytes()
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        k = int(mask.sum())
        if k == 0:
            result = (0.0, 0.0)
        else:
            Xs = self.X[:, mask]
            scores = []
            for train_idx, val_idx in self.folds:
                clf = RandomForestClassifier(
                    n_estimators=self.cfg.rf_trees,
                    max_depth=self.cfg.rf_max_depth,
                    random_state=self.seed,
                    n_jobs=self.cfg.n_jobs,
                )
                clf.fit(Xs[train_idx], self.y[train_idx])
                pred = clf.predict(Xs[val_idx])
                scores.append(f1_score(self.y[val_idx], pred, average="macro"))
            f1 = float(np.mean(scores))
            result = (f1 - self.cfg.alpha * k / self.n_features, f1)
        self._cache[key] = result
        self.n_evals += 1
        return result

    def __call__(self, mask: np.ndarray) -> float:
        return self.score(mask)[0]


@dataclass
class SelectionResult:
    """Outcome of one Layer-1 optimiser run."""

    algorithm: str
    mask: np.ndarray
    fitness: float
    macro_f1: float
    history: list[float] = field(default_factory=list)  # best fitness per iteration
    n_evals: int = 0
    seconds: float = 0.0

    @property
    def n_selected(self) -> int:
        return int(self.mask.sum())

    def selected(self, feature_names: list[str]) -> list[str]:
        return [f for f, m in zip(feature_names, self.mask) if m]

    def to_dict(self, feature_names: list[str]) -> dict:
        return {
            "algorithm": self.algorithm,
            "features": self.selected(feature_names),
            "n_selected": self.n_selected,
            "fitness": round(self.fitness, 6),
            "macro_f1": round(self.macro_f1, 6),
            "history": [round(h, 6) for h in self.history],
            "n_evals": self.n_evals,
            "seconds": round(self.seconds, 2),
        }


def random_mask(rng: np.random.Generator, n: int, p: float) -> np.ndarray:
    """Random non-empty boolean mask."""
    mask = rng.random(n) < p
    if not mask.any():
        mask[rng.integers(n)] = True
    return mask


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
