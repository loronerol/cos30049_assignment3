"""Paths, random seeds and every algorithm parameter used in the experiments.

All values here are reported in the Assignment 2 report (NFR9). PRISMO
(Agrawal et al. 2018) does not publish its optimiser settings or fitness
function, so the choices below are this project's documented interpretation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "raw"
RAW_DIR = RAW_ROOT / "cresci-2017"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

BOT_REPOSITORY = "https://botometer.osome.iu.edu/bot-repository/datasets/"
DATASET_URLS = {
    "cresci-2017": BOT_REPOSITORY + "cresci-2017/cresci-2017.csv.zip",
    "cresci-2015": BOT_REPOSITORY + "cresci-2015/cresci-2015.csv.tar.gz",
}

# ---------------------------------------------------------------- data
SEED = 42

# Account groups: name -> (dataset folder under data/raw, label; 1 = spam, 0 = genuine).
# Cresci-2017: fake_followers are excluded (they follow, they do not spam), and
# traditional_spambots_2..4 are excluded because they ship without tweets, so
# content features cannot be computed. The published genuine_accounts tweets.csv
# was truncated during export and only covers 1,083 of its 3,474 accounts, so the
# genuine class is topped up with the two human groups of Cresci-2015 from the
# same research group and in the same format: TFP ("The Fake Project" volunteers)
# and E13 (Italian #elezioni2013 users, which also offsets the mostly Italian
# social_spambots_1). Cresci-2015's fake-follower groups are not used.
GROUPS: dict[str, tuple[str, int]] = {
    "genuine_accounts": ("cresci-2017", 0),
    "TFP": ("cresci-2015", 0),
    "E13": ("cresci-2015", 0),
    "social_spambots_1": ("cresci-2017", 1),
    "social_spambots_2": ("cresci-2017", 1),
    "social_spambots_3": ("cresci-2017", 1),
    "traditional_spambots_1": ("cresci-2017", 1),
}
GROUP_LABELS: dict[str, int] = {g: label for g, (_, label) in GROUPS.items()}
MAX_TWEETS_PER_ACCOUNT = 200  # most recent tweets used for content features
MIN_TWEETS_PER_ACCOUNT = 1  # accounts with no tweets cannot be described
TEST_SIZE = 0.20
CV_FOLDS = 10


# ---------------------------------------------------------------- selection
@dataclass(frozen=True)
class FitnessConfig:
    """Wrapper fitness: macro-F1 of a small Random Forest minus a size penalty."""

    cv_folds: int = 3
    rf_trees: int = 50
    rf_max_depth: int | None = None
    alpha: float = 0.02  # weight of |S|/N, rewards smaller subsets
    n_jobs: int = -1


@dataclass(frozen=True)
class GAConfig:
    population: int = 30
    generations: int = 40
    tournament_k: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float | None = None  # None -> 1/N (one bit per individual)
    elites: int = 2
    init_prob: float = 0.5


@dataclass(frozen=True)
class PSOConfig:
    particles: int = 30
    iterations: int = 40
    w_start: float = 0.9
    w_end: float = 0.4
    c1: float = 2.0
    c2: float = 2.0
    v_max: float = 4.0
    init_prob: float = 0.5


@dataclass(frozen=True)
class SAConfig:
    steps: int = 1200
    t0: float = 1.0
    cooling: float = 0.95
    cool_every: int = 10
    max_flips: int = 2
    init_prob: float = 0.5
    # Fitness differences are small (F1 units), so the acceptance test uses
    # delta / temperature_scale; 0.01 makes a 0.01 F1 drop at T0 accepted ~37%.
    temperature_scale: float = 0.01


@dataclass(frozen=True)
class SelectionConfig:
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    pso: PSOConfig = field(default_factory=PSOConfig)
    sa: SAConfig = field(default_factory=SAConfig)
    seed_ga: int = SEED
    seed_pso: int = SEED + 1
    seed_sa: int = SEED + 2

    def to_dict(self) -> dict:
        return asdict(self)


def quick_selection_config() -> SelectionConfig:
    """Small budgets for smoke tests (`--quick`)."""
    return SelectionConfig(
        fitness=FitnessConfig(cv_folds=3, rf_trees=20),
        ga=GAConfig(population=10, generations=5),
        pso=PSOConfig(particles=10, iterations=5),
        sa=SAConfig(steps=50, cool_every=5),
    )


# ---------------------------------------------------------------- decision support
REVIEW_BAND = (0.35, 0.65)  # spam probability range labelled "needs review" (FR13)
