"""Simulated Annealing feature selection (PRISMO Layer 1, subset FS3)."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from ..config import SAConfig
from .fitness import SelectionResult, Timer, random_mask


def run_sa(
    fitness: Callable[[np.ndarray], tuple[float, float]],
    n_features: int,
    cfg: SAConfig,
    seed: int,
    log: Callable[[str], None] | None = None,
) -> SelectionResult:
    """Metropolis acceptance with geometric cooling; neighbours flip 1..max_flips bits."""
    rng = np.random.default_rng(seed)
    cur = random_mask(rng, n_features, cfg.init_prob)
    cur_fit, cur_f1 = fitness(cur)
    best, best_fit, best_f1 = cur.copy(), cur_fit, cur_f1
    temp = cfg.t0
    history: list[float] = []
    n_evals = 1

    with Timer() as t:
        for step in range(1, cfg.steps + 1):
            nb = cur.copy()
            n_flip = int(rng.integers(1, cfg.max_flips + 1))
            idx = rng.choice(n_features, size=n_flip, replace=False)
            nb[idx] = ~nb[idx]
            if not nb.any():
                continue
            nb_fit, nb_f1 = fitness(nb)
            n_evals += 1
            delta = nb_fit - cur_fit
            if delta >= 0 or rng.random() < math.exp(delta / (temp * cfg.temperature_scale)):
                cur, cur_fit, cur_f1 = nb, nb_fit, nb_f1
                if cur_fit > best_fit:
                    best, best_fit, best_f1 = cur.copy(), cur_fit, cur_f1
            if step % cfg.cool_every == 0:
                temp *= cfg.cooling
                history.append(float(best_fit))
                if log and step % (cfg.cool_every * 10) == 0:
                    log(f"SA step {step}/{cfg.steps} T={temp:.4f} best={best_fit:.4f} k={int(best.sum())}")

    return SelectionResult("SA", best, float(best_fit), float(best_f1), history, n_evals, t.seconds)
