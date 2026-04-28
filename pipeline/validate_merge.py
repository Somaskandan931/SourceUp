"""
Validate & Merge scraped CSVs — SourceUp pipeline step 1.
All paths from config.cfg.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import pandas as pd
from config import cfg

# ✅ FIXED: lowercase keys to match normalized columns
COLUMN_MAPPING = {
    "years on platform": "years with gs",
    "years_on_platform": "years with gs",
    "years with gs": "years with gs",  # safe fallback
}

def load_canonical_schema():
    schema_path = str(cfg.SCHEMA_FILE)
    if not os.path.exists(schema_path):
        raise FileNotFoundError(
            f"Schema file not found: {schema_path}\n"
            "This file (test_output.csv) defines the canonical column set.\n"
            "Place one row of your scraper output there as the template."
        )
    return [c.strip().lower() for c in pd.read_csv(schema_path, nrows=0).columns]


def validate_and_merge():
    os.makedirs(str(cfg.MERGED_DIR), exist_ok=True)
    canonical = load_canonical_schema()

    pattern = str(cfg.OUTPUTS_DIR / "*.csv")
    files   = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {cfg.OUTPUTS_DIR}\n"
            "Run the Java scraper first (global-sources.bat on Windows, "
            "or java -jar somasjar.jar ... on Linux/Mac)."
        )

    dfs = []
    for f in files:
        df = pd.read_csv(f)

        # ✅ normalize
        df.columns = [c.strip().lower() for c in df.columns]

        # ✅ ensure mapping keys are lowercase
        mapping = {k.strip().lower(): v for k, v in COLUMN_MAPPING.items()}
        df.rename(columns=mapping, inplace=True)

        # DEBUG (optional, remove later)
        print(f"Processed {os.path.basename(f)} columns:", df.columns.tolist())

        missing = set(canonical) - set(df.columns)

        # ✅ robust handling instead of crashing
        if missing:
            print(f"[WARN] {os.path.basename(f)} missing columns: {missing}")
            for col in missing:
                df[col] = None

        df = df[canonical]
        df["source_file"] = os.path.basename(f)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    out    = str(cfg.MERGED_DATA)
    merged.to_csv(out, index=False)

    print(f"✅ Merged {len(files)} files → {out}  ({len(merged)} rows)")
    return merged


if __name__ == "__main__":
    validate_and_merge()