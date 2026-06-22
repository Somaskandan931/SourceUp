"""
Dataset Diagnostic — run this once, paste the output back
-----------------------------------------------------------
Answers 4 questions about why LTR isn't beating SBERT/rule-based:
  1. Are features actually varied, or mostly constant/clustered?
  2. Is the relevance label distribution skewed?
  3. Are query groups too small for NDCG@10 reranking to matter?
  4. Does faiss_score correlate with relevance more than other features?
     (if yes -> labels were likely derived from SBERT -> LTR has a ceiling)

Usage:
  python eval/dataset_diagnostic.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _find_project_root(marker: str = "config.py") -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find project root (looked for {marker})")


sys.path.insert(0, str(_find_project_root()))
from config import cfg

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

df = pd.read_csv(str(cfg.TRAINING_DATA))
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

print("=" * 70)
print(f"Loaded: {len(df):,} rows, {df['query_id'].nunique()} queries")
print("=" * 70)

# ── 1. Feature variance ──────────────────────────────────────────────
print("\n--- 1. FEATURE VARIANCE (is anything near-constant?) ---")
feature_cols = ["price_match", "price_ratio", "location_match",
                 "cert_match", "faiss_score", "faiss_rank"]
for col in feature_cols:
    if col in df.columns:
        print(f"  {col:16s}  mean={df[col].mean():.4f}  std={df[col].std():.4f}  "
              f"zero%={(df[col] == 0).mean()*100:.1f}%  unique={df[col].nunique()}")
    else:
        print(f"  {col:16s}  NOT FOUND in dataframe")

# ── 2. Label distribution ────────────────────────────────────────────
print("\n--- 2. RELEVANCE LABEL DISTRIBUTION ---")
print(df["relevance"].value_counts().sort_index().to_string())

# ── 3. Query group sizes ─────────────────────────────────────────────
print("\n--- 3. QUERY GROUP SIZES (candidates per query) ---")
group_sizes = df.groupby("query_id").size()
print(group_sizes.describe().to_string())
print(f"  queries with <10 candidates: {(group_sizes < 10).sum()} / {len(group_sizes)}")

# ── 4. Correlation with relevance ────────────────────────────────────
print("\n--- 4. CORRELATION OF EACH FEATURE WITH RELEVANCE LABEL ---")
present = [c for c in feature_cols if c in df.columns] + ["relevance"]
corr = df[present].corr()["relevance"].drop("relevance").sort_values(ascending=False)
print(corr.to_string())

print("\n" + "=" * 70)
print("DONE — paste everything above back to get a diagnosis")
print("=" * 70)