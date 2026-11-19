"""PRISMO Layer 2: priority-based feature formation (FS4)."""

from __future__ import annotations

import numpy as np


def priority_combine(masks: dict[str, np.ndarray], feature_names: list[str]) -> dict:
    """Combine three Layer-1 subsets with PRISMO's priority rules.

    priority 1 = selected by all three algorithms
    priority 2 = selected by exactly two
    FS4 = priority 1 + priority 2; features selected by only one are discarded.
    Feature order follows `feature_names`.
    """
    if len(masks) != 3:
        raise ValueError(f"PRISMO combines exactly three subsets, got {len(masks)}")
    stacked = np.vstack([np.asarray(m, dtype=bool) for m in masks.values()])
    if stacked.shape[1] != len(feature_names):
        raise ValueError("mask length does not match feature_names")
    votes = stacked.sum(axis=0)
    p1 = [f for f, v in zip(feature_names, votes) if v == 3]
    p2 = [f for f, v in zip(feature_names, votes) if v == 2]
    fs4 = [f for f, v in zip(feature_names, votes) if v >= 2]
    return {
        "priority_1": p1,
        "priority_2": p2,
        "discarded_single_vote": [f for f, v in zip(feature_names, votes) if v == 1],
        "fs4": fs4,
        "votes": {f: int(v) for f, v in zip(feature_names, votes)},
        "fs4_mask": votes >= 2,
    }
