import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_FILE = "C:/Users/somas/PycharmProjects/SourceUp/data/clean/suppliers_clean.csv"
EMBEDDING_DIR = "C:/Users/somas/PycharmProjects/SourceUp/data/embeddings"
INDEX_FILE = f"{EMBEDDING_DIR}/suppliers.faiss"
META_FILE = f"{EMBEDDING_DIR}/suppliers_meta.csv"

def incremental_update():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            "Cleaned data not found. Run clean() first."
        )

    os.makedirs(EMBEDDING_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    # Build text corpus (safe + generic)
    corpus = df.astype(str).agg(" ".join, axis=1).tolist()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        corpus,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    df.to_csv(META_FILE, index=False)

    print("✅ FAISS index created successfully")

if __name__ == "__main__":
    incremental_update()
