"""
FAISS Embedding — SourceUp pipeline step 3.
All paths from config.cfg.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import cfg


def incremental_update():
    if not os.path.exists(str(cfg.CLEAN_DATA)):
        raise FileNotFoundError(
            f"Clean data not found: {cfg.CLEAN_DATA}\n"
            "Run clean_normalize.py first."
        )
    os.makedirs(str(cfg.EMBEDDINGS_DIR), exist_ok=True)

    df     = pd.read_csv(str(cfg.CLEAN_DATA))
    corpus = df.astype(str).agg(" ".join, axis=1).tolist()

    print(f"  Encoding {len(corpus)} suppliers with all-MiniLM-L6-v2...")
    model      = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        corpus, show_progress_bar=True, convert_to_numpy=True
    ).astype(np.float32)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(cfg.FAISS_INDEX))
    df.to_csv(str(cfg.FAISS_META), index=False)

    print(f"✅ FAISS index → {cfg.FAISS_INDEX}  ({index.ntotal} vectors)")
    print(f"✅ Metadata    → {cfg.FAISS_META}")
    return index


if __name__ == "__main__":
    incremental_update()