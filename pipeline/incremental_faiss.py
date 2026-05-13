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

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

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

MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 256

INDEX_CHUNK_SIZE = 10000


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
        "product name" in df.columns
        and
        "supplier name" in df.columns
    ):

        df = df.drop_duplicates(
            subset=[
                "product name",
                "supplier name"
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
        "supplier name",
        "product name",
        "category",
        "country",
        "city",
        "description"
    ]

    embed_cols = [
        col for col in preferred_cols
        if col in df.columns
    ]

    if not embed_cols:
        embed_cols = df.columns.tolist()

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

    model.max_seq_length = 128

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
        f"in {total_batches:,} batches..."
    )

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

        embeddings = model.encode(
            batch,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        # Add directly to FAISS
        index.add(embeddings)

        processed = end

        percent = (
            processed / total
        ) * 100

        print(
            f"   Batch "
            f"{start // BATCH_SIZE + 1:,}/"
            f"{total_batches:,} "
            f"→ {processed:,}/{total:,} "
            f"({percent:.2f}%)"
        )

        # RAM cleanup
        del embeddings

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