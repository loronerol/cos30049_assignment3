"""Genetic Algorithm feature selection (PRISMO Layer 1, subset FS1)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..config import GAConfig
from .fitness import SelectionResult, Timer, random_mask


def run_ga(
    fitness: Callable[[np.ndarray], tuple[float, float]],
    n_features: int,
    cfg: GAConfig,
    seed: int,
    log: Callable[[str], None] | None = None,
) -> SelectionResult:
    """Binary GA: tournament selection, uniform crossover, bit-flip mutation, elitism.

    `fitness` returns (fitness, macro_f1) for a boolean mask.
    """
    rng = np.random.default_rng(seed)
    mut_rate = cfg.mutation_rate if cfg.mutation_rate is not None else 1.0 / n_features
    pop = np.array([random_mask(rng, n_features, cfg.init_prob) for _ in range(cfg.population)])
    best_mask, best_fit, best_f1 = None, -np.inf, 0.0
    history: list[float] = []
    n_evals = 0

    with Timer() as t:
        for gen in range(cfg.generations):
            scored = [fitness(ind) for ind in pop]
            n_evals += len(pop)
            fits = np.array([s[0] for s in scored])
            i_best = int(fits.argmax())
            if fits[i_best] > best_fit:
                best_fit, best_f1, best_mask = fits[i_best], scored[i_best][1], pop[i_best].copy()
            history.append(float(best_fit))
            if log:
                log(f"GA gen {gen + 1}/{cfg.generations} best={best_fit:.4f} k={int(best_mask.sum())}")

            order = np.argsort(-fits)
            children = [pop[i].copy() for i in order[: cfg.elites]]
            while len(children) < cfg.population:
                p1 = pop[_tournament(rng, fits, cfg.tournament_k)]
                p2 = pop[_tournament(rng, fits, cfg.tournament_k)]
                if rng.random() < cfg.crossover_rate:
                    swap = rng.random(n_features) < 0.5
                    c1 = np.where(swap, p2, p1)
                    c2 = np.where(swap, p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                for child in (c1, c2):
                    flips = rng.random(n_features) < mut_rate
                    child = child ^ flips
                    if not child.any():
                        child[rng.integers(n_features)] = True
                    if len(children) < cfg.population:
                        children.append(child)
            pop = np.array(children)

    return SelectionResult("GA", best_mask, float(best_fit), float(best_f1), history, n_evals, t.seconds)


def _tournament(rng: np.random.Generator, fits: np.ndarray, k: int) -> int:
    contenders = rng.choice(len(fits), size=k, replace=False)
    return int(contenders[fits[contenders].argmax()])
