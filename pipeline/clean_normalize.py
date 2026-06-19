"""
Clean & Normalise — SourceUp pipeline step 2.
All paths from config.cfg.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from config import cfg


# Canonical schema — hardcoded so clean_normalize no longer depends on an
# external schema CSV (cfg.SCHEMA_FILE) that may not exist in every run.
# Any of these columns present in the merged data are kept; unknown columns
# are dropped silently. Add new fields here as they're scraped.
CANONICAL_COLUMNS = [
    "supplier_name", "product_name", "category", "price", "moq",
    "location", "city", "state", "country", "years_with_gs",
    "rating", "review_count", "response_rate", "verified",
    "description", "image_url", "profile_url", "contact",
    "tags", "certifications",
]


def clean():
    if not os.path.exists(str(cfg.MERGED_DATA)):
        raise FileNotFoundError(
            f"Merged file not found: {cfg.MERGED_DATA}\n"
            "Run validate_merge.py first."
        )
    os.makedirs(str(cfg.CLEAN_DIR), exist_ok=True)
    df = pd.read_csv(str(cfg.MERGED_DATA))
    df.columns = [c.strip().lower() for c in df.columns]
    keep_cols = [c for c in CANONICAL_COLUMNS if c in df.columns]
    if not keep_cols:
        # Fallback: schema didn't match anything (e.g. scraper changed field
        # names) — keep all columns rather than silently producing an empty df.
        print("⚠️  No canonical columns matched — keeping all columns as-is.")
        keep_cols = list(df.columns)
    df = df[keep_cols]
    df.drop_duplicates(inplace=True)
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    df.to_csv(str(cfg.CLEAN_DATA), index=False)
    print(f"✅ Cleaned → {cfg.CLEAN_DATA}  ({len(df)} rows)")
    return df


if __name__ == "__main__":
    clean()