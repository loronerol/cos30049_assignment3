import numpy as np
import pytest
from sklearn.datasets import make_classification

from spamsentinel_ml.config import FitnessConfig, GAConfig, PSOConfig, SAConfig
from spamsentinel_ml.selection import FitnessEvaluator, priority_combine, run_ga, run_pso, run_sa

N_INFORMATIVE = 4


@pytest.fixture(scope="module")
def evaluator():
    # shuffle=False puts the informative columns first: 0..3
    X, y = make_classification(
        n_samples=400, n_features=12, n_informative=N_INFORMATIVE, n_redundant=0,
        n_repeated=0, shuffle=False, class_sep=1.5, random_state=0,
    )
    return FitnessEvaluator(X, y, FitnessConfig(cv_folds=3, rf_trees=25, alpha=0.05), seed=0)


def _check(result, n_features=12):
    assert result.mask.shape == (n_features,)
    assert result.mask.dtype == bool
    assert result.mask.any()
    # every informative feature is kept and most noise is dropped
    assert result.mask[:N_INFORMATIVE].sum() >= N_INFORMATIVE - 1
    assert result.mask[N_INFORMATIVE:].sum() <= 3
    # history is monotone non-decreasing (it tracks the best so far)
    assert all(b >= a - 1e-12 for a, b in zip(result.history, result.history[1:]))


def test_fitness_cache_and_penalty(evaluator):
    full = np.ones(12, dtype=bool)
    before = evaluator.n_evals
    fit1, f1 = evaluator.score(full)
    fit2, _ = evaluator.score(full)
    assert fit1 == fit2
    assert evaluator.n_evals == before + 1  # second call hit the cache
    assert fit1 == pytest.approx(f1 - 0.05)
    assert evaluator.score(np.zeros(12, dtype=bool)) == (0.0, 0.0)


def test_ga_recovers_informative(evaluator):
    _check(run_ga(evaluator.score, 12, GAConfig(population=16, generations=12), seed=1))


def test_pso_recovers_informative(evaluator):
    _check(run_pso(evaluator.score, 12, PSOConfig(particles=16, iterations=12), seed=2))


def test_sa_recovers_informative(evaluator):
    _check(run_sa(evaluator.score, 12, SAConfig(steps=250, cool_every=5), seed=3))


def test_seeded_runs_are_reproducible(evaluator):
    a = run_ga(evaluator.score, 12, GAConfig(population=8, generations=3), seed=7)
    b = run_ga(evaluator.score, 12, GAConfig(population=8, generations=3), seed=7)
    assert np.array_equal(a.mask, b.mask)


def test_priority_combine_rules():
    names = ["a", "b", "c", "d", "e"]
    masks = {
        "GA": np.array([1, 1, 0, 1, 0], bool),
        "PSO": np.array([1, 1, 1, 0, 0], bool),
        "SA": np.array([1, 0, 0, 0, 0], bool),
    }
    out = priority_combine(masks, names)
    assert out["priority_1"] == ["a"]
    assert out["priority_2"] == ["b"]
    assert out["fs4"] == ["a", "b"]
    assert out["discarded_single_vote"] == ["c", "d"]
    assert out["votes"]["e"] == 0
    assert out["fs4_mask"].tolist() == [True, True, False, False, False]


def test_priority_combine_requires_three():
    with pytest.raises(ValueError):
        priority_combine({"GA": np.ones(2, bool)}, ["a", "b"])
