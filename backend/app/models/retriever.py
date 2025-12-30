"""
Retriever Module
----------------
Semantic search using FAISS and sentence transformers.
"""

import os
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

BASE_DIR = "C:/Users/somas/PycharmProjects/SourceUp/data/embeddings"
INDEX_FILE = f"{BASE_DIR}/suppliers.faiss"
META_FILE = f"{BASE_DIR}/suppliers_meta.csv"

# Global singletons for model and index
_model = None
_index = None
_meta = None


def get_model () :
    """Load and cache the sentence transformer model."""
    global _model
    if _model is None :
        _model = SentenceTransformer( "all-MiniLM-L6-v2" )
    return _model


def load_index () :
    """Load and cache the FAISS index and metadata."""
    global _index, _meta

    if _index is not None :
        return _index, _meta

    if not os.path.exists( INDEX_FILE ) :
        raise RuntimeError(
            f"FAISS index not found at {INDEX_FILE}. "
            "Run pipeline/run_all.py first to create the index."
        )

    if not os.path.exists( META_FILE ) :
        raise RuntimeError(
            f"Metadata file not found at {META_FILE}. "
            "Run pipeline/run_all.py first to create metadata."
        )

    _index = faiss.read_index( INDEX_FILE )
    _meta = pd.read_csv( META_FILE )

    print( f"✅ Loaded FAISS index with {_index.ntotal} vectors" )
    print( f"✅ Loaded metadata with {len( _meta )} suppliers" )

    return _index, _meta


def retrieve ( query: str, k: int = 10 ) -> List[Dict] :
    """
    Retrieve top-k suppliers using semantic search.

    Args:
        query: Search query string
        k: Number of results to return

    Returns:
        List of supplier dictionaries with scores
    """
    model = get_model()
    index, meta = load_index()

    # Encode query
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    # Search FAISS index
    distances, ids = index.search( query_embedding, k )

    # Retrieve metadata for results
    results = []
    for idx, (distance, supplier_id) in enumerate( zip( distances[0], ids[0] ) ) :
        if supplier_id < len( meta ) :
            supplier = meta.iloc[supplier_id].to_dict()
            supplier["faiss_score"] = float( distance )
            supplier["rank"] = idx + 1
            results.append( supplier )

    return results