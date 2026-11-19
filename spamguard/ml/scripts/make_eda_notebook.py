"""Generate ml/notebooks/eda.ipynb (exploratory data analysis, WBS 2.2)."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

cells = [
    ("md", "# SpamSentinel — exploratory data analysis (Cresci-2017)\n\n"
           "Distributions, class balance and correlations of the 64 account-level features in "
           "`data/processed/features.csv`. Run `ml/scripts/build_features.py` first."),
    ("code", "import sys, json\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd().parent))\n"
             "import numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
             "from spamsentinel_ml.config import PROCESSED_DIR\n"
             "from spamsentinel_ml.features import FEATURES, FEATURE_NAMES, FEATURE_INFO\n"
             "df = pd.read_csv(PROCESSED_DIR / 'features.csv', index_col='user_id')\n"
             "print(df.shape)\njson.loads((PROCESSED_DIR / 'cleaning_report.json').read_text())"),
    ("md", "## Class balance and account groups"),
    ("code", "display(df['label'].map({0: 'genuine', 1: 'spam'}).value_counts())\n"
             "df['group'].value_counts().plot.barh(title='Accounts per group'); plt.show()"),
    ("md", "## Missing values"),
    ("code", "miss = df[FEATURE_NAMES].isna().mean().sort_values(ascending=False)\nmiss[miss > 0]"),
    ("md", "## Feature summary by class\nMedians per class and the standardised mean difference "
           "(how far apart the classes are, in pooled standard deviations)."),
    ("code", "g = df.groupby('label')[FEATURE_NAMES]\nmed = g.median().T.rename(columns={0: 'genuine_median', 1: 'spam_median'})\n"
             "m, s = g.mean().T, g.std().T\nmed['smd'] = (m[1] - m[0]) / np.sqrt((s[0]**2 + s[1]**2) / 2)\n"
             "med['category'] = [FEATURE_INFO[f].category for f in med.index]\n"
             "med.sort_values('smd', key=abs, ascending=False).round(3)"),
    ("md", "## Distributions of the most separating features (log scale where skewed)"),
    ("code", "top = med['smd'].abs().sort_values(ascending=False).index[:12]\n"
             "fig, axes = plt.subplots(3, 4, figsize=(16, 10))\n"
             "for ax, f in zip(axes.ravel(), top):\n"
             "    for lab, name, c in [(0, 'genuine', '#0F766E'), (1, 'spam', '#D50032')]:\n"
             "        v = df.loc[df.label == lab, f].dropna()\n"
             "        v = np.log1p(v) if v.max() > 50 else v\n"
             "        ax.hist(v, bins=40, alpha=0.5, density=True, label=name, color=c)\n"
             "    ax.set_title(f, fontsize=9)\naxes[0, 0].legend(); fig.tight_layout(); plt.show()"),
    ("md", "## Correlation between features\nHighly correlated pairs are redundant — the motivation for PRISMO's feature selection."),
    ("code", "corr = df[FEATURE_NAMES].corr(method='spearman')\n"
             "fig, ax = plt.subplots(figsize=(12, 10)); im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)\n"
             "ax.set_xticks(range(len(FEATURE_NAMES))); ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=5)\n"
             "ax.set_yticks(range(len(FEATURE_NAMES))); ax.set_yticklabels(FEATURE_NAMES, fontsize=5)\n"
             "fig.colorbar(im); plt.show()\n"
             "pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack()\n"
             "pairs[pairs.abs() > 0.9].sort_values(key=abs, ascending=False)"),
    ("md", "## Differences between spambot groups\nThe spam class mixes several bot families with distinct behaviour."),
    ("code", "df.groupby('group')[list(top[:8])].median().T.round(2)"),
]

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) if k == "md" else nbf.v4.new_code_cell(c) for k, c in cells]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / "notebooks" / "eda.ipynb"
nbf.write(nb, out)
print(f"wrote {out}")
