import pandas as pd
import os

# Absolute paths (safe for Windows + PyCharm)
BASE_DIR = "C:/Users/somas/PycharmProjects/SourceUp"

INPUT_FILE = f"{BASE_DIR}/data/merged/suppliers_all.csv"
OUTPUT_FILE = f"{BASE_DIR}/data/clean/suppliers_clean.csv"
SCHEMA_FILE = f"{BASE_DIR}/data/test_output.csv"

def load_schema():
    if not os.path.exists(SCHEMA_FILE):
        raise FileNotFoundError(
            "test_output.csv not found. "
            "This file defines the canonical schema."
        )
    return [c.strip().lower() for c in pd.read_csv(SCHEMA_FILE, nrows=0).columns]

def clean():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            "Merged file not found. "
            "Run validate_and_merge() first."
        )

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Load canonical schema
    schema_cols = load_schema()

    df = pd.read_csv(INPUT_FILE)

    # Align strictly to schema (NO hard-coded columns)
    df = df[schema_cols]

    # Generic, schema-safe cleaning
    df.drop_duplicates(inplace=True)

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    df.to_csv(OUTPUT_FILE, index=False)
    print("✅ Cleaning completed (schema-driven)")

if __name__ == "__main__":
    clean()
