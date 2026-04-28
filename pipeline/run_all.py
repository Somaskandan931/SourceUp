"""
SourceUp Data Pipeline — run_all.py
-------------------------------------
Runs the full pipeline:
  1. validate_merge  — validate scraped CSVs against canonical schema
  2. clean_normalize — deduplicate and strip whitespace
  3. incremental_faiss — build SBERT + FAISS index
  4. feature_builder — create LTR training dataset

All paths from config.cfg. Cross-platform (Windows / Linux / Mac).

Usage:
    python pipeline/run_all.py
    python pipeline/run_all.py --skip-features   # skip LTR data build
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip LTR training data build (feature_builder.py)")
    args = parser.parse_args()

    cfg.ensure_dirs()

    print("=" * 60)
    print("🚀 SourceUp — Full Data Pipeline")
    print(f"   Root: {cfg.ROOT}")
    print("=" * 60)

    # Step 1: Validate & merge scraped CSVs
    print("\n[1/4] Validate & Merge ─────────────────────────────────────")
    from pipeline.validate_merge import validate_and_merge
    validate_and_merge()

    # Step 2: Clean & normalise
    print("\n[2/4] Clean & Normalise ────────────────────────────────────")
    from pipeline.clean_normalize import clean
    clean()

    # Step 3: Build FAISS index
    print("\n[3/4] FAISS Embedding ──────────────────────────────────────")
    from pipeline.incremental_faiss import incremental_update
    incremental_update()

    # Step 4: Build LTR training features (optional, slow)
    if not args.skip_features:
        print("\n[4/4] Feature Builder (LTR data) ───────────────────────────")
        try:
            from features.feature_builder import build_training_data
            build_training_data()
        except Exception as e:
            print(f"   ⚠️  Feature builder failed (non-fatal): {e}")
            print("   Run separately: python features/features/feature_builder.py")
    else:
        print("\n[4/4] Feature Builder — SKIPPED (--skip-features flag)")

    print("\n" + "=" * 60)
    print("✅ Pipeline complete!")
    print(f"   FAISS index  : {cfg.FAISS_INDEX}")
    print(f"   Clean data   : {cfg.CLEAN_DATA}")
    if not args.skip_features:
        print(f"   Training data: {cfg.TRAINING_DATA}")
    print("=" * 60)


if __name__ == "__main__":
    main()