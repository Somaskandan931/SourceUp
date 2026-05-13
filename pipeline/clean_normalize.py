"""
Clean & Normalise — SourceUp pipeline step 2.
All paths from config.cfg.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from config import cfg


def load_schema():
    if not os.path.exists(str(cfg.SCHEMA_FILE)):
        raise FileNotFoundError(f"Schema file not found: {cfg.SCHEMA_FILE}")
    return [c.strip().lower() for c in pd.read_csv(str(cfg.SCHEMA_FILE), nrows=0).columns]


def clean():
    if not os.path.exists(str(cfg.MERGED_DATA)):
        raise FileNotFoundError(
            f"Merged file not found: {cfg.MERGED_DATA}\n"
            "Run validate_merge.py first."
        )
    os.makedirs(str(cfg.CLEAN_DIR), exist_ok=True)
    schema_cols = load_schema()
    df = pd.read_csv(str(cfg.MERGED_DATA))
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[[c for c in schema_cols if c in df.columns]]
    df.drop_duplicates(inplace=True)
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    df.to_csv(str(cfg.CLEAN_DATA), index=False)
    print(f"✅ Cleaned → {cfg.CLEAN_DATA}  ({len(df)} rows)")
    return df


if __name__ == "__main__":
    clean()