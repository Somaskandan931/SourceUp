import sys
import os

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

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

# Column aliases: maps alternative names from other scrapers → canonical
COLUMN_ALIASES = {
    "years on platform": "years with gs",
    "years_on_platform": "years with gs",
    "company": "company name",
    "supplier": "supplier name",
    "location": "supplier location",
    "city": "supplier location",   # indiamart uses separate city column
    "description": "product name",
    "moq": "min order qty",
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
            # Keep only canonical schema
            # ------------------------------------------------
            df = df[canonical]

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
    # Optional limit
    # ------------------------------------------------------------
    if limit:

        merged = merged.head(limit)

        print(
            f"⚡ Limited dataset to "
            f"{len(merged):,} rows"
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