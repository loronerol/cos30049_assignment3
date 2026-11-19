"""Binary Particle Swarm Optimisation feature selection (PRISMO Layer 1, subset FS2).

Follows the discrete binary PSO of Kennedy & Eberhart (1997): velocities are
real-valued and each bit is set with probability sigmoid(velocity).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..config import PSOConfig
from .fitness import SelectionResult, Timer, random_mask


def run_pso(
    fitness: Callable[[np.ndarray], tuple[float, float]],
    n_features: int,
    cfg: PSOConfig,
    seed: int,
    log: Callable[[str], None] | None = None,
) -> SelectionResult:
    rng = np.random.default_rng(seed)
    x = np.array([random_mask(rng, n_features, cfg.init_prob) for _ in range(cfg.particles)])
    v = rng.uniform(-1.0, 1.0, size=(cfg.particles, n_features))

    scored = [fitness(p) for p in x]
    n_evals = len(x)
    pbest = x.copy()
    pbest_fit = np.array([s[0] for s in scored])
    pbest_f1 = np.array([s[1] for s in scored])
    g = int(pbest_fit.argmax())
    gbest, gbest_fit, gbest_f1 = pbest[g].copy(), pbest_fit[g], pbest_f1[g]
    history: list[float] = []

    with Timer() as t:
        for it in range(cfg.iterations):
            w = cfg.w_start - (cfg.w_start - cfg.w_end) * it / max(cfg.iterations - 1, 1)
            r1 = rng.random((cfg.particles, n_features))
            r2 = rng.random((cfg.particles, n_features))
            xf = x.astype(float)
            v = w * v + cfg.c1 * r1 * (pbest - xf) + cfg.c2 * r2 * (gbest - xf)
            v = np.clip(v, -cfg.v_max, cfg.v_max)
            x = rng.random((cfg.particles, n_features)) < 1.0 / (1.0 + np.exp(-v))
            for i in range(cfg.particles):
                if not x[i].any():
                    x[i, rng.integers(n_features)] = True
                fit, f1 = fitness(x[i])
                n_evals += 1
                if fit > pbest_fit[i]:
                    pbest[i], pbest_fit[i], pbest_f1[i] = x[i].copy(), fit, f1
                    if fit > gbest_fit:
                        gbest, gbest_fit, gbest_f1 = x[i].copy(), fit, f1
            history.append(float(gbest_fit))
            if log:
                log(f"PSO iter {it + 1}/{cfg.iterations} best={gbest_fit:.4f} k={int(gbest.sum())}")

    return SelectionResult("PSO", gbest, float(gbest_fit), float(gbest_f1), history, n_evals, t.seconds)
