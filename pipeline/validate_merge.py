import sys
import os
from pathlib import Path


def _find_project_root(marker: str = "config.py") -> Path:
    """Walk up from this file until the folder containing `marker` is found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find project root (looked for {marker})")


sys.path.insert(0, str(_find_project_root()))

import glob
import pandas as pd

from config import cfg


# ------------------------------------------------------------
# Load canonical schema
# ------------------------------------------------------------
# Canonical columns derived from GlobalSources scraper schema (primary data source)
CANONICAL_COLUMNS = [
    "query", "product name", "product id", "model number",
    "price", "price min", "price max", "min order qty", "unit",
    "lead time", "fob port", "product url", "image url",
    "company name", "supplier name", "supplier location",
    "business type", "certifications", "years with gs",
    "category l1", "category l2", "category l3", "category l4",
]

# ROOT CAUSE FIX (this version): the old code did `df = df[canonical]`
# unconditionally, which silently DROPPED any column not in
# CANONICAL_COLUMNS — including real, well-populated columns some source
# files actually have. Confirmed against the raw scraped CSVs:
#   - output8.csv has a "Rating" column, 97.5% filled, real 2.0-5.0
#     values — dropped on the floor by the old `df[canonical]` line.
#   - output8.csv also has "City" (distinct from "Supplier Location") and
#     "Certification Count" (pre-computed) — also dropped.
#   - output_full.csv has "Cert Count" (its own pre-computed cert count)
#     and ~650 other columns (Reliability Score, Composite Score, Spec:
#     * fields, etc.) — most of those are genuinely file-specific and
#     not worth threading through the canonical schema, but rating in
#     particular matters: it's a direct input to
#     rule_baseline.score_rule_based_independent()'s supplier_rating
#     term, which was previously reported as permanently constant
#     (std=0.0) and excluded from that formula. That conclusion was
#     right about the symptom (suppliers_clean.csv never had usable
#     rating data) but wrong about the cause — it wasn't that no source
#     file has ratings, it's that this file threw them away before
#     clean_normalize.py (step 2) ever got a chance to see them.
#
# Fix: in addition to the required CANONICAL_COLUMNS (always present,
# "Unknown"-filled if a source file lacks them — unchanged behaviour),
# also pass through any of these OPTIONAL_ENRICHMENT_COLUMNS a given
# source file happens to have. Files without them simply get NaN for
# that column in the merged output, same as any other missing optional
# field — this does not change CANONICAL_COLUMNS or break any existing
# downstream code that only reads the original 23 columns.
OPTIONAL_ENRICHMENT_COLUMNS = [
    "rating",
    "city",
    "certification count",
]

# Column aliases: maps alternative names from other scrapers → canonical
COLUMN_ALIASES = {
    "years on platform": "years with gs",
    "years_on_platform": "years with gs",
    "company": "company name",
    "supplier": "supplier name",
    "location": "supplier location",
    "description": "product name",
    "moq": "min order qty",
    "cert count": "certification count",  # output_full.csv (IndiaMART) spelling
}


def load_canonical_schema():
    schema_path = str(cfg.SCHEMA_FILE)
    if os.path.exists(schema_path):
        schema_df = pd.read_csv(schema_path, nrows=0)
        return [c.strip().lower() for c in schema_df.columns]
    # Fall back to hardcoded GlobalSources schema — no schema file required
    return CANONICAL_COLUMNS


# ------------------------------------------------------------
# Standardize column names
# ------------------------------------------------------------
def normalize_columns(df):

    # lowercase + trim spaces
    df.columns = [
        c.strip().lower()
        for c in df.columns
    ]

    # alternative column mappings
    rename_map = COLUMN_ALIASES

    df = df.rename(columns=rename_map)

    return df


# ------------------------------------------------------------
# Validate + Merge
# ------------------------------------------------------------
def validate_and_merge(limit=None):

    os.makedirs(
        str(cfg.MERGED_DIR),
        exist_ok=True
    )

    canonical = load_canonical_schema()

    pattern = str(
        cfg.OUTPUTS_DIR / "*.csv"
    )

    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in "
            f"{cfg.OUTPUTS_DIR}"
        )

    dfs = []

    print(
        f"📂 Found {len(files)} CSV files\n"
    )

    for f in files:

        try:

            print(
                f"📄 Processing: "
                f"{os.path.basename(f)}"
            )

            df = pd.read_csv(
                f,
                low_memory=False
            )

            # ------------------------------------------------
            # Normalize columns
            # ------------------------------------------------
            df = normalize_columns(df)

            # ------------------------------------------------
            # Remove duplicate column names
            # ------------------------------------------------
            df = df.loc[
                :,
                ~df.columns.duplicated()
            ]

            # ------------------------------------------------
            # Fill missing columns
            # ------------------------------------------------
            missing = (
                set(canonical)
                - set(df.columns)
            )

            if missing:

                print(
                    f"[WARN] Missing columns: "
                    f"{missing}"
                )

                for col in missing:
                    df[col] = "Unknown"

            # ------------------------------------------------
            # Keep canonical schema + any optional enrichment
            # columns this particular source file happens to have.
            #
            # FIX (this version): previously this was `df = df[canonical]`,
            # which dropped every column outside the 23-column canonical
            # schema — including "rating" on files that genuinely have it
            # (e.g. output8.csv, 97.5% filled). Source files that lack an
            # optional column simply don't get one added here (left out of
            # `present_optional`), so after pd.concat() those rows get NaN
            # for that column — same as any other naturally-missing field,
            # not "Unknown" (these are enrichment signals, not required
            # identity fields, so NaN is the more honest default than a
            # fabricated placeholder string).
            # ------------------------------------------------
            present_optional = [
                col for col in OPTIONAL_ENRICHMENT_COLUMNS
                if col in df.columns
            ]
            if present_optional:
                print(
                    f"   ➕ Optional enrichment columns found: "
                    f"{present_optional}"
                )

            df = df[canonical + present_optional]

            # ------------------------------------------------
            # Add source tracking
            # ------------------------------------------------
            df["source_file"] = (
                os.path.basename(f)
            )

            dfs.append(df)

            print(
                f"   ✅ Rows loaded: "
                f"{len(df):,}"
            )

        except Exception as e:

            print(
                f"❌ Failed processing "
                f"{os.path.basename(f)}"
            )

            print(f"   Reason: {e}")

    # ------------------------------------------------------------
    # Merge all datasets
    # ------------------------------------------------------------
    if not dfs:
        raise ValueError(
            "No valid CSVs were processed."
        )

    print("\n🔄 Merging datasets...")

    merged = pd.concat(
        dfs,
        ignore_index=True
    )

    # ------------------------------------------------------------
    # Remove fully duplicated rows
    # ------------------------------------------------------------
    before = len(merged)

    merged = merged.drop_duplicates()

    removed = before - len(merged)

    print(
        f"🧹 Removed "
        f"{removed:,} duplicate rows"
    )

    # ------------------------------------------------------------
    # Optional limit — random sample, not head(). head() would just take
    # the first N rows in file-concatenation order, which for uneven file
    # sizes (e.g. output.csv=35k vs output2.csv=523k) means small --limit
    # values come ENTIRELY from one source file. A random sample stays
    # proportionally representative of every file, category, and location.
    # ------------------------------------------------------------
    if limit:

        merged = merged.sample(
            n=min(limit, len(merged)),
            random_state=42
        ).reset_index(drop=True)

        print(
            f"⚡ Limited dataset to "
            f"{len(merged):,} rows (random sample, seed=42)"
        )

    # ------------------------------------------------------------
    # Save merged dataset
    # ------------------------------------------------------------
    out = str(cfg.MERGED_DATA)

    merged.to_csv(
        out,
        index=False
    )

    print(
        f"\n✅ Merged {len(files)} files "
        f"→ {out}"
    )

    print(
        f"📦 Final rows: "
        f"{len(merged):,}"
    )

    return merged


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":

    validate_and_merge()