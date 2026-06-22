"""
FAISS Embedding — SourceUp pipeline step 3
Optimized for large-scale supplier embedding generation.

Features:
    ✅ GPU acceleration
    ✅ Memory-safe streaming
    ✅ Better semantic embeddings
    ✅ Cosine similarity search
    ✅ Low RAM usage
    ✅ GTX 1650 optimized
"""

import sys
import os
import gc
import time

from pathlib import Path


def _find_project_root(marker: str = "config.py") -> Path:
    """Walk up from this file until the folder containing `marker` is found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find project root (looked for {marker})")


sys.path.insert(0, str(_find_project_root()))

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from config import cfg


# ------------------------------------------------------------
# CUDA Optimization
# ------------------------------------------------------------
torch.backends.cudnn.benchmark = True

MODEL_NAME = "all-mpnet-base-v2"   # Upgraded to match retriever (was all-MiniLM-L6-v2)

BATCH_SIZE = 128                    # Bumped to 256 below if CUDA+FP16 (see incremental_update)

INDEX_CHUNK_SIZE = 10000

PRINT_EVERY = 50                    # Printing every batch to a Windows console is itself slow


def incremental_update():

    # ------------------------------------------------------------
    # Validate dataset
    # ------------------------------------------------------------
    if not os.path.exists(str(cfg.CLEAN_DATA)):
        raise FileNotFoundError(
            f"Clean data not found: {cfg.CLEAN_DATA}\n"
            "Run clean_normalize.py first."
        )

    os.makedirs(str(cfg.EMBEDDINGS_DIR), exist_ok=True)

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------
    print("\n📥 Loading cleaned supplier dataset...")

    df = pd.read_csv(str(cfg.CLEAN_DATA))

    print(f"📊 Total suppliers loaded: {len(df):,}")

    # ------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------
    original_len = len(df)

    if (
        "product_name" in df.columns
        and
        "supplier_name" in df.columns
    ):

        df = df.drop_duplicates(
            subset=[
                "product_name",
                "supplier_name"
            ],
            keep="first"
        )

    else:
        df = df.drop_duplicates()

    df = df.reset_index(drop=True)

    dedup_len = len(df)

    print(
        f"📊 After deduplication: "
        f"{dedup_len:,} "
        f"(removed {original_len - dedup_len:,} duplicates)"
    )

    # ------------------------------------------------------------
    # Select meaningful columns only
    # ------------------------------------------------------------
    preferred_cols = [
        "supplier_name",
        "product_name",
        "category",
        "country",
        "city",
        "certifications",
        "description"
    ]

    embed_cols = [
        col for col in preferred_cols
        if col in df.columns
    ]

    if not embed_cols:
        embed_cols = df.columns.tolist()

    # product_name is the single most valuable retrieval field — force it
    # to the front of the embedding text even if preferred_cols matching
    # above somehow missed it.
    if "product_name" in df.columns and "product_name" not in embed_cols:
        embed_cols.insert(0, "product_name")

    print("\n🧠 Embedding columns:")

    for col in embed_cols:
        print(f"   • {col}")

    # ------------------------------------------------------------
    # Build corpus
    # ------------------------------------------------------------
    corpus = (
        df[embed_cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .tolist()
    )

    # ------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\n⚡ Using device: {device}")

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    print(f"🤖 Loading model: {MODEL_NAME}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device
    )

    model.max_seq_length = 128  # Trimmed from 256 — most fields here (name/category/
    # city/certifications) are short; 256 was mostly wasted padding compute.
    # Revert to 256 if you notice longer "description" text getting truncated.

    # FP16 inference — halves VRAM use (real win, reduces crash risk) and is a
    # speed win IF this GPU has Tensor Cores (RTX-class). On a plain GTX 1650
    # (no Tensor Cores) it mainly just saves memory, not raw speed.
    if device == "cuda":
        model.half()
        BATCH_SIZE = 256  # freed VRAM headroom (FP16) lets us push more per batch

    # ------------------------------------------------------------
    # Create sample embedding
    # ------------------------------------------------------------
    sample_embedding = model.encode(
        ["test"],
        convert_to_numpy=True
    ).astype(np.float32)

    dim = sample_embedding.shape[1]

    del sample_embedding

    gc.collect()

    # ------------------------------------------------------------
    # Create FAISS index
    # ------------------------------------------------------------
    print(f"\n🧠 Creating FAISS index (dim={dim})")

    # Cosine similarity
    index = faiss.IndexFlatIP(dim)

    # ------------------------------------------------------------
    # Streaming batch encoding
    # ------------------------------------------------------------
    total = len(corpus)

    total_batches = (
        total + BATCH_SIZE - 1
    ) // BATCH_SIZE

    print(
        f"\n🔄 Encoding {total:,} suppliers "
        f"in {total_batches:,} batches (batch_size={BATCH_SIZE})..."
    )

    encode_start = time.time()

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch = corpus[start:end]

        # Retry on transient CUDA/cuBLAS failures (memory fragmentation from
        # thousands of consecutive forward passes) instead of crashing the
        # whole multi-hour run. Falls back to CPU for that one batch if GPU
        # retries are exhausted.
        embeddings = None
        for attempt in range(3):
            try:
                embeddings = model.encode(
                    batch,
                    batch_size=BATCH_SIZE,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True   # unit-norm embeddings for cosine similarity
                ).astype(np.float32)
                break
            except RuntimeError as e:
                msg = str(e)
                if "CUDA" in msg or "CUBLAS" in msg or "cublas" in msg:
                    print(f"   ⚠️  GPU error on batch {start // BATCH_SIZE + 1} "
                          f"(attempt {attempt + 1}/3): {e}")
                    if device == "cuda":
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        gc.collect()
                else:
                    raise
        if embeddings is None:
            print(f"   ⚠️  Falling back to CPU for batch {start // BATCH_SIZE + 1}")
            cpu_model = SentenceTransformer(MODEL_NAME, device="cpu")
            cpu_model.max_seq_length = 128
            embeddings = cpu_model.encode(
                batch,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(np.float32)
            del cpu_model

        # Explicit L2 normalization — guarantees IndexFlatIP computes true cosine similarity
        faiss.normalize_L2(embeddings)

        # Add directly to FAISS
        index.add(embeddings)

        processed = end
        batch_num = start // BATCH_SIZE + 1

        # Printing + gc.collect() every single one of 6,476 batches is real
        # overhead on its own (console I/O is slow, especially in Windows
        # terminals). Only do both periodically.
        if batch_num % PRINT_EVERY == 0 or processed == total:
            percent = (processed / total) * 100
            elapsed = time.time() - encode_start
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (total - processed) / rate if rate > 0 else 0
            print(
                f"   Batch {batch_num:,}/{total_batches:,} → "
                f"{processed:,}/{total:,} ({percent:.1f}%) "
                f"| {rate:,.0f} rows/s "
                f"| elapsed {elapsed/60:.1f}m | ETA {remaining/60:.1f}m"
            )

        # RAM cleanup — periodic CUDA cache clear prevents the allocator
        # fragmentation that leads to cuBLASLt workspace failures over a
        # 6,000+ batch run.
        del embeddings
        if device == "cuda" and batch_num % 200 == 0:
            torch.cuda.empty_cache()
        if batch_num % PRINT_EVERY == 0:
            gc.collect()

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    print("\n💾 Saving FAISS index...")

    faiss.write_index(
        index,
        str(cfg.FAISS_INDEX)
    )

    print("💾 Saving metadata...")

    df.to_csv(
        str(cfg.FAISS_META),
        index=False
    )

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------
    print("\n✅ Validation checks")

    assert (
        index.ntotal == len(df)
    ), (
        f"Mismatch detected!\n"
        f"Index vectors: {index.ntotal:,}\n"
        f"Metadata rows: {len(df):,}"
    )

    print("✅ Index and metadata aligned")

    print(
        f"\n🎉 Completed successfully"
    )

    print(
        f"📦 Final indexed suppliers: "
        f"{index.ntotal:,}"
    )

    return index


if __name__ == "__main__":
    incremental_update()