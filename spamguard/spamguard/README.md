# SpamSentinel

An explainable, moderator-centred tool for detecting spam accounts on social media
(COS30049, Group 03). This repository holds the Stage 2 machine learning pipeline,
which reproduces the PRISMO method (Agrawal et al. 2018). The FastAPI back end and
React front end come next. The project management plan lives in [`Plan/`](Plan/).

## Dataset: Cresci-2017 (fallback, risk R1)

The PRISMO dataset (Benevenuto et al. 2010) was not released to the team by the
decision date, so the planned fallback is used: **Cresci-2017**
(Cresci et al. 2017), from the IU Bot Repository. It is for academic research use only.

| Label | Groups used | Why |
|---|---|---|
| genuine (0) | `genuine_accounts` | verified human accounts |
| spam (1) | `social_spambots_1..3`, `traditional_spambots_1` | spam accounts that come with their tweets |
| excluded | `fake_followers`, `traditional_spambots_2..4` | fake followers don't spam; the other groups ship without tweets, so content features can't be computed |

Unlike the PRISMO dataset, Cresci-2017 ships raw profiles and tweets. The pipeline
therefore computes **64 account-level features** following Benevenuto et al.'s
definitions: 39 content, 15 user-behaviour and 10 profile features, listed in
`data/processed/data_dictionary.csv`. Content features use each account's 200 most
recent tweets. Screen names are dropped from all processed outputs.

**Caveats.** The groups were collected at different times, so time-dependent
features can carry collection artefacts. Also, `social_spambots_1` is mostly
Italian. Language fields are therefore not used as features.

## Setup

Requires Python 3.12.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows (use .venv/bin/pip on macOS/Linux)
.venv/Scripts/python ml/scripts/download_data.py  # ~466 MB, about 20 min
```

## Running the pipeline

Each step reads the previous step's outputs. Add `--quick` to the last three for a smoke run.

```bash
.venv/Scripts/python ml/scripts/build_features.py    # raw CSVs -> data/processed/ (D2)
.venv/Scripts/python ml/scripts/run_selection.py     # GA, PSO, SA -> FS1..FS3, PRISMO -> FS4
.venv/Scripts/python ml/scripts/run_experiments.py   # 10-fold CV, 5 classifiers x 5 feature sets
.venv/Scripts/python ml/scripts/export_model.py      # refit, held-out test, export bundles (D3)
.venv/Scripts/python -m pytest ml/tests              # unit tests
```

Exploratory analysis: `ml/notebooks/eda.ipynb`.

## LLM extension (L0–L4)

All local training runs on one 8 GB GPU (developed on an RTX 4060 laptop GPU). 4-bit
`bitsandbytes` works natively on Windows, so WSL is not needed.

```bash
.venv/Scripts/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install -r requirements-llm.txt
```

| Step | Command | What it does |
|---|---|---|
| L0 | `ml/scripts/prepare_text.py` | English-only bio + up to 20 recent tweets per account, plus shared 5-fold OOF folds |
| L1 | `ml/scripts/train_text_encoder.py` | Fine-tunes `cardiffnlp/twitter-roberta-base`. Out-of-fold `text_spam_score` for training accounts; resumes per fold after a crash |
| L1b | `run_selection.py --with-text`, `run_experiments.py --with-text`, `export_model.py --with-text` | PRISMO with the text score as a 65th feature |
| L2 | `ml/scripts/train_llm_qlora.py [--train-limit N]` | 4-bit QLoRA on `Qwen2.5-3B-Instruct`, plus a local zero-shot baseline |
| L3 | `ml/scripts/zero_shot_claude.py [--confirm]` | Optional, **paid**: Claude zero-shot via the Batches API. Only estimates cost unless `--confirm` is given |
| L4 | `ml/scripts/adversarial_rewrite.py` | An LLM rewrites 200 test spam accounts' tweets to sound human, then every model is rescored |
| — | `ml/scripts/compare_models.py` | Writes `results/comparison.md` |

**Why English only.** `social_spambots_1` is mostly Italian. Without the filter, the text
models could learn "Italian = spam". Accounts with no English text get a marker instead.
That marker still leans spam (most such accounts are Italian spambots), which the report
discusses.

## Method: documented choices (NFR9)

PRISMO doesn't publish its optimiser parameters or its fitness function. This
project's interpretation is recorded in `ml/spamsentinel_ml/config.py` and written
to `results/selection.json` on every run.

- **Split.** A stratified 80/20 train/test split, stratified by account group, with seed 42. Feature selection and cross-validation see **only the training split**. The test split is used once, for the final model. This is stricter than PRISMO, which ran CV on the whole dataset.
- **Fitness.** Mean 3-fold macro-F1 of a 50-tree Random Forest on the subset, minus 0.02 × |S|/N.
- **GA.** Population 30, 40 generations, tournament k=3, uniform crossover p=0.8, bit-flip mutation 1/N, 2 elites.
- **Binary PSO.** 30 particles, 40 iterations, inertia 0.9 → 0.4, c1 = c2 = 2, sigmoid transfer, vmax 4.
- **SA.** 1,200 steps, T0 = 1, cooling × 0.95 every 10 steps, neighbour = flip 1–2 bits.
- **Layer 2.** Priority 1 = features chosen by all three algorithms, priority 2 = chosen by exactly two. FS4 = priority 1 ∪ priority 2.
- **Classifiers** (WEKA → scikit-learn):
  - RandomForest(100) → `RandomForestClassifier`
  - NaiveBayes → `GaussianNB`
  - JRip → `wittgenstein.RIPPER`
  - J48 → entropy `DecisionTreeClassifier`, with `min_samples_leaf=2` and cost-complexity pruning
  - HistGradientBoosting is optional

  Every classifier uses median imputation inside its pipeline. No scaler is used, because all of these models ignore feature scale.
- **Metrics.** Accuracy, TPR (spam recall), TNR, precision, F1 (spam and macro), ROC-AUC and build time (single-threaded fit, ms).
- **Review band.** Spam probabilities between 0.35 and 0.65 are labelled *needs review* (FR13).

## Results (Stage 2)

**Feature selection.** On the training split, GA picked 9 features, PSO 20 and SA 11.
PRISMO's priority rule kept **9 of 64** (PRISMO itself kept 19 of 62):

- 3 priority-1 features: `retweets_received_per_tweet_mean`, `frac_replies`, `default_profile`
- 6 priority-2 features: followers, followees, total tweets, account age, has a bio, bio length

**10-fold CV, Random Forest.** FS4 scores 0.9900 macro-F1 against 0.9896 for all 64
features, and builds in 624 ms instead of 1,400 ms. RIPPER and the decision tree score
about 0.98. Gaussian Naive Bayes is far behind (about 0.74), because the count features
are heavily skewed. Full tables: `results/cv_results.md`.

**Held-out test (1,789 accounts), Random Forest on FS4:**

| Accuracy | TPR | TNR | Precision | F1 (macro) | ROC-AUC |
|---|---|---|---|---|---|
| 0.9894 | 0.9882 | 0.9917 | 0.9957 | 0.9882 | 0.9954 |

Both targets are met: macro-F1 ≥ 0.85, and TPR above PRISMO's 72.4%. By group, accuracy
is lowest on `social_spambots_3` (0.957) and `social_spambots_1` (0.970), the bots that
imitate humans best.

**Review band (0.35–0.65).** Only 16 test accounts (0.9%) fall in the band, and it catches
2 of the 19 errors. The model is confident even when it is wrong, which matters for RQ4
(over-reliance).

**Text signal (L1).** From English text alone, the fine-tuned encoder reaches test macro-F1
0.949 and ROC-AUC 0.985 (0.973 macro-F1 on English-text accounts). When `text_spam_score`
is offered as a 65th feature, all three optimisers select it, and FS4 becomes 11 features.
Test macro-F1 is then 0.986 against 0.988 without it: the numeric features already sit at
the ceiling on this dataset.

The full model comparison, including the LLMs and the adversarial test, is in
`results/comparison.md`.

**Caveats for the report:**

- Cresci is known to be an easy benchmark.
- The account groups were collected at different times, so time-dependent features partly
  encode the collection. `retweets_received_per_tweet_mean` is the clearest case.
- The genuine accounts mix three sources.

## Outputs

| Path | Contents |
|---|---|
| `data/processed/` | `features.csv`, `train.csv`, `test.csv`, `data_dictionary.csv`, `cleaning_report.json` |
| `results/selection.json` | FS1–FS4, priorities, votes, parameters, convergence history |
| `results/cv_results.{csv,md}` | 10-fold CV summary (mean ± sd) for every model × feature set |
| `results/test_results.{json,md}` | held-out test metrics, review-band analysis, accuracy by group |
| `results/figures/` | convergence, Layer-2 votes, F1 / TPR / build-time comparisons |
| `models/spamsentinel_model.joblib` | chosen model on FS4, plus feature metadata and per-class reference values (for the API) |
| `models/spamsentinel_model_full.joblib` | same classifier on all 64 features (study mode, FR22) |
| `models/metrics.json`, `models/sample_accounts.json` | numbers for the insights page; sample accounts for the UI |

## Repository layout

```
ml/spamsentinel_ml/   library: config, data loading, features, selection (GA/PSO/SA/PRISMO), models, evaluation, export
ml/scripts/           pipeline entry points
ml/tests/             pytest suite
ml/notebooks/         EDA
Plan/                 project management plan (LaTeX)
```
<<<<<<< HEAD
#
