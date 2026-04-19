# data/dataset_loader.py
"""
Dataset Loader for Supplier Ranking
------------------------------------
Loads and validates the training/evaluation dataset.
"""

import json
import os
from typing import List, Dict, Optional


def load_dataset ( path: str ) -> List[Dict] :
    """
    Load the supplier ranking dataset.

    Expected format:
    [
        {
            "query": "steel suppliers",
            "constraints": {"budget": 50000, "max_delivery": 7},
            "suppliers": [...]
        }
    ]

    Args:
        path: Path to JSON dataset file

    Returns:
        List of query dictionaries with suppliers
    """
    if not os.path.exists( path ) :
        raise FileNotFoundError( f"Dataset not found: {path}" )

    with open( path, 'r', encoding='utf-8' ) as f :
        data = json.load( f )

    # Validate structure
    _validate_dataset( data )

    return data


def _validate_dataset ( data: List[Dict] ) -> None :
    """Validate dataset structure."""
    required_keys = {'query', 'suppliers'}

    for idx, item in enumerate( data ) :
        missing = required_keys - set( item.keys() )
        if missing :
            raise ValueError( f"Query {idx} missing required keys: {missing}" )

        for supp_idx, supplier in enumerate( item['suppliers'] ) :
            if 'id' not in supplier :
                raise ValueError( f"Supplier {supp_idx} in query {idx} missing 'id'" )


def split_dataset ( data: List[Dict], train_ratio: float = 0.7,
                    val_ratio: float = 0.15, seed: int = 42 ) -> Dict :
    """
    Split dataset into train/validation/test sets.

    Args:
        data: Full dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        seed: Random seed

    Returns:
        Dictionary with 'train', 'val', 'test' splits
    """
    import numpy as np
    np.random.seed( seed )

    indices = np.random.permutation( len( data ) )

    train_end = int( len( data ) * train_ratio )
    val_end = train_end + int( len( data ) * val_ratio )

    return {
        'train' : [data[i] for i in indices[:train_end]],
        'val' : [data[i] for i in indices[train_end :val_end]],
        'test' : [data[i] for i in indices[val_end :]]
    }


def get_query_groups ( dataset: List[Dict] ) -> List[int] :
    """Generate query group IDs for LTR training."""
    groups = []
    for qid, query in enumerate( dataset ) :
        groups.extend( [qid] * len( query['suppliers'] ) )
    return groups