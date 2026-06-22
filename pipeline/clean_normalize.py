"""
Clean & Normalise — SourceUp pipeline step 2.
All paths from config.cfg.
"""

import sys, os
from pathlib import Path


def _find_project_root(marker: str = "config.py") -> Path:
    """Walk up from this file until the folder containing `marker` is found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find project root (looked for {marker})")


sys.path.insert(0, str(_find_project_root()))

import pandas as pd
from config import cfg


# Canonical output schema (underscore form) → accepted input aliases.
#
# ROOT CAUSE FIX: validate_merge.py (step 1) produces space-separated
# lowercase column names (e.g. "product name", "supplier location",
# "category l1") via its own CANONICAL_COLUMNS list. This file's old
# CANONICAL_COLUMNS used underscore names ("product_name", "location") that
# never actually matched step 1's output — only "certifications" happened
# to be spelled identically in both conventions. Every other field
# (supplier_name, product_name, category, description, city, country, ...)
# was silently dropped here, which is why incremental_faiss.py's "Embedding
# columns" log only ever showed "certifications": every other preferred
# column was simply absent from suppliers_clean.csv by the time it ran.
#
# Fix: resolve each canonical output name from a list of accepted input
# spellings (mirrors backend/app/utils/fields.py's alias approach), so the
# match no longer depends on guessing whether upstream used spaces or
# underscores.
CANONICAL_ALIASES = {
    "supplier_name":   ["supplier_name", "supplier name", "company_name", "company name"],
    "product_name":    ["product_name", "product name", "description"],
    "category":        ["category", "category l1", "category_l1"],
    "price":           ["price", "price min", "price_min"],
    "moq":             ["moq", "min order qty", "min_order_qty"],
    "location":        ["location", "supplier_location", "supplier location"],
    "city":            ["city", "supplier_location", "supplier location"],
    "state":           ["state"],
    "country":         ["country"],
    "years_with_gs":   ["years_with_gs", "years with gs", "years_on_platform", "years on platform"],
    "business_type":   ["business_type", "business type"],
    "rating":          ["rating"],
    "review_count":    ["review_count", "review count"],
    "response_rate":   ["response_rate", "response rate"],
    "verified":        ["verified"],
    "description":     ["description", "product name", "product_name"],
    "image_url":       ["image_url", "image url"],
    "profile_url":     ["profile_url", "profile url", "product_url", "product url"],
    "contact":         ["contact"],
    "tags":            ["tags"],
    "certifications":  ["certifications"],
}


def _resolve_canonical_columns(df_columns):
    """
    For each canonical output name, find the first alias present in
    df_columns. Returns an ordered dict {canonical_name: source_column}.
    Two canonical names may legitimately resolve to the same source column
    (e.g. "product_name" and "description" both fall back to whichever of
    the two actually exists) — that's intentional so downstream consumers
    (incremental_faiss.py, feature_builder.py) always find *something*
    usable under the name they expect, rather than getting an empty column.
    """
    resolved = {}
    for canonical, aliases in CANONICAL_ALIASES.items():
        for alias in aliases:
            if alias in df_columns:
                resolved[canonical] = alias
                break
    return resolved


def clean():
    if not os.path.exists(str(cfg.MERGED_DATA)):
        raise FileNotFoundError(
            f"Merged file not found: {cfg.MERGED_DATA}\n"
            "Run validate_merge.py first."
        )
    os.makedirs(str(cfg.CLEAN_DIR), exist_ok=True)
    df = pd.read_csv(str(cfg.MERGED_DATA))
    df.columns = [c.strip().lower() for c in df.columns]

    resolved = _resolve_canonical_columns(df.columns)

    if not resolved:
        # Fallback: schema didn't match anything (e.g. scraper changed field
        # names) — keep all columns rather than silently producing an empty df.
        print("⚠️  No canonical columns matched — keeping all columns as-is.")
        df_out = df.copy()
    else:
        # Build the output frame under canonical (underscore) names, source
        # column data from whichever alias actually matched.
        df_out = pd.DataFrame({
            canonical: df[source] for canonical, source in resolved.items()
        })
        missing = [c for c in CANONICAL_ALIASES if c not in resolved]
        if missing:
            print(f"   ℹ️  No source column found for: {missing} — omitted from suppliers_clean.csv")
        print(f"   ✅ Resolved {len(resolved)}/{len(CANONICAL_ALIASES)} canonical columns: "
              f"{ {k: v for k, v in resolved.items()} }")

    df_out.drop_duplicates(inplace=True)
    for col in df_out.columns:
        df_out[col] = df_out[col].astype(str).str.strip()
    df_out.to_csv(str(cfg.CLEAN_DATA), index=False)
    print(f"✅ Cleaned → {cfg.CLEAN_DATA}  ({len(df_out)} rows)")
    return df_out


if __name__ == "__main__":
    clean()