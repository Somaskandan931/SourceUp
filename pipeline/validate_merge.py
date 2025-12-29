import pandas as pd
import glob
import os

INPUT_DIR = "C:/Users/somas/PycharmProjects/SourceUp/data/outputs"
OUTPUT_DIR = "C:/Users/somas/PycharmProjects/SourceUp/data/merged"
SCHEMA_FILE = "C:/Users/somas/PycharmProjects/SourceUp/data/test_output.csv"

def load_canonical_schema():
    if not os.path.exists(SCHEMA_FILE):
        raise FileNotFoundError(
            "test_output.csv not found. "
            "This file defines the canonical schema."
        )
    schema_df = pd.read_csv(SCHEMA_FILE, nrows=0)
    return [c.strip().lower() for c in schema_df.columns]

def validate_and_merge():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    canonical_columns = load_canonical_schema()
    files = glob.glob(f"{INPUT_DIR}/*.csv")

    if not files:
        raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

    dfs = []

    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip().lower() for c in df.columns]

        missing = set(canonical_columns) - set(df.columns)
        if missing:
            raise ValueError(
                f"{os.path.basename(f)} missing columns: {missing}"
            )

        # Align schema exactly to test_output.csv
        df = df[canonical_columns]
        df["source_file"] = os.path.basename(f)

        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(f"{OUTPUT_DIR}/suppliers_all.csv", index=False)

    print(f"✅ Schema validated using test_output.csv")
    print(f"✅ Merged {len(files)} files successfully")

if __name__ == "__main__":
    validate_and_merge()
