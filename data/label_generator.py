# data/label_generator.py
"""
Weak Supervision Label Generator
--------------------------------
Generates training labels using heuristic scoring with controlled noise.
This enables Learning-to-Rank training without manual annotations.
"""

import numpy as np
from typing import Dict, Optional


def generate_label ( supplier: Dict, constraints: Optional[Dict] = None,
                     noise_std: float = 0.1 ) -> float :
    """
    Generate weak supervision label for a supplier.

    Labels are derived from heuristic scoring and treated as weak supervision.
    This is scientifically justified when ground truth interaction data is unavailable.

    Args:
        supplier: Supplier data dictionary
        constraints: Optional business constraints
        noise_std: Standard deviation of injected Gaussian noise

    Returns:
        Relevance score between 0 and 5
    """
    # Base relevance from heuristic features
    base_score = 0.0

    # Similarity contribution (30%)
    similarity = supplier.get( 'similarity_score', 0.5 )
    base_score += similarity * 1.5

    # Price competitiveness (25%)
    price = supplier.get( 'price', 0 )
    if price > 0 :
        # Lower price = higher score (inverse relationship)
        price_score = max( 0, 1 - (price / 100000) )
        base_score += price_score * 1.25
    else :
        base_score += 0.5 * 1.25

    # Rating contribution (20%)
    rating = supplier.get( 'rating', 3.0 )
    base_score += (rating / 5.0) * 1.0

    # Delivery speed (15%)
    delivery_days = supplier.get( 'delivery_days', 14 )
    delivery_score = max( 0, 1 - (delivery_days / 30) )
    base_score += delivery_score * 0.75

    # Certification bonus (10%)
    if supplier.get( 'certified', False ) :
        base_score += 0.5

    # Apply constraint penalties if provided
    if constraints :
        penalty = 0.0

        # Budget constraint
        if 'budget' in constraints and price > constraints['budget'] :
            penalty += 0.5

        # Delivery constraint
        if 'max_delivery' in constraints :
            if delivery_days > constraints['max_delivery'] :
                penalty += 0.3

        base_score = max( 0, base_score - penalty )

    # Add controlled noise (weak supervision characteristic)
    noise = np.random.normal( 0, noise_std )
    label = base_score + noise

    # Clip to valid range [0, 5]
    return float( np.clip( label, 0, 5 ) )


def generate_labels_batch ( suppliers: list, constraints: Optional[Dict] = None ) -> list :
    """Generate labels for multiple suppliers."""
    return [generate_label( s, constraints ) for s in suppliers]


def validate_label_distribution ( labels: list ) -> Dict :
    """Validate label distribution for training readiness."""
    return {
        'mean' : np.mean( labels ),
        'std' : np.std( labels ),
        'min' : np.min( labels ),
        'max' : np.max( labels ),
        'skewness' : float( np.mean( (labels - np.mean( labels )) ** 3 ) / (np.std( labels ) ** 3) ) if np.std(
            labels ) > 0 else 0
    }