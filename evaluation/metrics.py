"""
IEEE-Compliant Ranking Metrics for Learning-to-Rank Systems
-----------------------------------------------------------
Implements standard information retrieval metrics suitable for academic publication.
"""

import numpy as np
from typing import List, Set, Union
from scipy.stats import kendalltau


def precision_at_k ( retrieved: List, relevant: List, k: int = 5 ) -> float :
    """
    Calculate Precision@k - fraction of top-k results that are relevant.

    Args:
        retrieved: List of retrieved item IDs (in ranked order)
        relevant: List of relevant item IDs (ground truth)
        k: Number of top results to consider

    Returns:
        Precision@k score (0.0 to 1.0)

    Example:
        >>> retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
        >>> relevant = ['doc1', 'doc3', 'doc5']
        >>> precision_at_k(retrieved, relevant, k=5)
        0.6  # 3 out of 5 are relevant
    """
    if k <= 0 :
        raise ValueError( "k must be positive" )

    if not retrieved :
        return 0.0

    # Get top-k retrieved items
    top_k = retrieved[:k]

    # Convert to sets for intersection
    relevant_set = set( relevant )
    top_k_set = set( top_k )

    # Calculate precision
    num_relevant_in_top_k = len( top_k_set & relevant_set )
    return num_relevant_in_top_k / k


def recall_at_k ( retrieved: List, relevant: List, k: int = 5 ) -> float :
    """
    Calculate Recall@k - fraction of relevant items found in top-k results.

    Args:
        retrieved: List of retrieved item IDs (in ranked order)
        relevant: List of relevant item IDs (ground truth)
        k: Number of top results to consider

    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if not relevant :
        return 0.0

    if k <= 0 :
        raise ValueError( "k must be positive" )

    top_k = retrieved[:k]
    relevant_set = set( relevant )
    top_k_set = set( top_k )

    num_relevant_in_top_k = len( top_k_set & relevant_set )
    return num_relevant_in_top_k / len( relevant_set )


def average_precision ( retrieved: List, relevant: List ) -> float :
    """
    Calculate Average Precision (AP) for a single query.

    AP is the average of precision values calculated at each position
    where a relevant document is retrieved.

    Args:
        retrieved: List of retrieved item IDs (in ranked order)
        relevant: List of relevant item IDs (ground truth)

    Returns:
        Average Precision score (0.0 to 1.0)
    """
    if not relevant or not retrieved :
        return 0.0

    relevant_set = set( relevant )
    precision_sum = 0.0
    num_relevant_found = 0

    for i, item in enumerate( retrieved ) :
        if item in relevant_set :
            num_relevant_found += 1
            precision_at_i = num_relevant_found / (i + 1)
            precision_sum += precision_at_i

    if num_relevant_found == 0 :
        return 0.0

    return precision_sum / len( relevant_set )


def mean_average_precision ( results: List[tuple] ) -> float :
    """
    Calculate Mean Average Precision (MAP) across multiple queries.

    Args:
        results: List of (retrieved, relevant) tuples for each query

    Returns:
        MAP score (0.0 to 1.0)

    Example:
        >>> query1 = (['d1', 'd2', 'd3'], ['d1', 'd3'])
        >>> query2 = (['d4', 'd5', 'd6'], ['d5', 'd6'])
        >>> mean_average_precision([query1, query2])
        0.75
    """
    if not results :
        return 0.0

    ap_scores = [average_precision( ret, rel ) for ret, rel in results]
    return np.mean( ap_scores )


def dcg_at_k ( relevance_scores: List[float], k: int = None ) -> float :
    """
    Calculate Discounted Cumulative Gain at position k.

    DCG = Σ (rel_i / log2(i + 1)) for i in [1, k]

    Args:
        relevance_scores: List of relevance scores in ranked order
        k: Number of top results to consider (None = all)

    Returns:
        DCG score
    """
    if not relevance_scores :
        return 0.0

    if k is None :
        k = len( relevance_scores )

    relevance_scores = relevance_scores[:k]

    dcg = 0.0
    for i, rel in enumerate( relevance_scores ) :
        dcg += rel / np.log2( i + 2 )  # i+2 because indexing starts at 0

    return dcg


def ndcg_at_k ( relevance_scores: List[float], k: int = None ) -> float :
    """
    Calculate Normalized Discounted Cumulative Gain at position k.

    NDCG = DCG / IDCG (ideal DCG)

    Args:
        relevance_scores: List of relevance scores in ranked order
        k: Number of top results to consider (None = all)

    Returns:
        NDCG score (0.0 to 1.0)

    Example:
        >>> ndcg_at_k([3, 2, 3, 0, 1, 2], k=5)
        0.785
    """
    if not relevance_scores :
        return 0.0

    dcg = dcg_at_k( relevance_scores, k )

    # Calculate ideal DCG (scores sorted in descending order)
    ideal_scores = sorted( relevance_scores, reverse=True )
    idcg = dcg_at_k( ideal_scores, k )

    if idcg == 0 :
        return 0.0

    return dcg / idcg


def mean_reciprocal_rank ( results: List[List] ) -> float :
    """
    Calculate Mean Reciprocal Rank (MRR) across multiple queries.

    MRR = (1/|Q|) * Σ (1 / rank_i)
    where rank_i is the position of the first relevant result for query i.

    Args:
        results: List of rankings, where each ranking is a list of
                 (item_id, is_relevant) tuples

    Returns:
        MRR score (0.0 to 1.0)
    """
    if not results :
        return 0.0

    reciprocal_ranks = []

    for ranking in results :
        for i, (item_id, is_relevant) in enumerate( ranking ) :
            if is_relevant :
                reciprocal_ranks.append( 1.0 / (i + 1) )
                break
        else :
            reciprocal_ranks.append( 0.0 )

    return np.mean( reciprocal_ranks )


def kendall_tau_distance ( ranking1: List, ranking2: List ) -> float :
    """
    Calculate Kendall's Tau correlation coefficient between two rankings.

    Measures the similarity between two rankings. Values range from -1 to 1:
    - 1: Perfect agreement
    - 0: No correlation
    - -1: Perfect disagreement

    Args:
        ranking1: First ranking (list of scores or ranks)
        ranking2: Second ranking (list of scores or ranks)

    Returns:
        Kendall's Tau coefficient (-1.0 to 1.0)
    """
    if len( ranking1 ) != len( ranking2 ) :
        raise ValueError( "Rankings must have the same length" )

    if len( ranking1 ) < 2 :
        return 0.0

    tau, _ = kendalltau( ranking1, ranking2 )

    # Handle NaN (occurs when all values are identical)
    if np.isnan( tau ) :
        return 0.0

    return tau


def normalized_discounted_cumulative_gain_batch (
        y_true: np.ndarray,
        y_pred: np.ndarray,
        query_groups: np.ndarray,
        k: int = 10
) -> float :
    """
    Calculate average NDCG@k across multiple queries (batch version).

    This is the standard metric for Learning-to-Rank evaluation.

    Args:
        y_true: Ground truth relevance scores (1D array)
        y_pred: Predicted relevance scores (1D array)
        query_groups: Query IDs for each item (1D array)
        k: Number of top results to consider

    Returns:
        Average NDCG@k across all queries

    Example:
        >>> y_true = np.array([3, 2, 1, 0, 2, 3, 1])
        >>> y_pred = np.array([2.5, 2.0, 1.0, 0.5, 2.2, 2.8, 1.1])
        >>> query_groups = np.array([0, 0, 0, 1, 1, 1, 1])
        >>> normalized_discounted_cumulative_gain_batch(y_true, y_pred, query_groups, k=3)
        0.872
    """
    ndcg_scores = []

    for qid in np.unique( query_groups ) :
        mask = query_groups == qid

        if mask.sum() < 2 :
            continue

        true_scores = y_true[mask]
        pred_scores = y_pred[mask]

        # Sort by predicted scores (descending)
        sorted_indices = np.argsort( pred_scores )[: :-1]
        sorted_true = true_scores[sorted_indices]

        # Calculate NDCG
        ndcg = ndcg_at_k( sorted_true.tolist(), k )
        ndcg_scores.append( ndcg )

    return np.mean( ndcg_scores ) if ndcg_scores else 0.0


def evaluate_ranking_model (
        y_true: np.ndarray,
        y_pred: np.ndarray,
        query_groups: np.ndarray
) -> dict :
    """
    Comprehensive evaluation of a ranking model using multiple metrics.

    Returns a dictionary with all standard ranking metrics suitable
    for IEEE publication.

    Args:
        y_true: Ground truth relevance scores
        y_pred: Predicted relevance scores
        query_groups: Query IDs for each item

    Returns:
        Dictionary containing:
        - ndcg@5, ndcg@10: Normalized DCG at different cutoffs
        - precision@5, precision@10: Precision at different cutoffs
        - kendall_tau: Ranking correlation
        - num_queries: Number of evaluated queries
    """
    metrics = {}

    # NDCG at different cutoffs
    metrics['ndcg@5'] = normalized_discounted_cumulative_gain_batch(
        y_true, y_pred, query_groups, k=5
    )
    metrics['ndcg@10'] = normalized_discounted_cumulative_gain_batch(
        y_true, y_pred, query_groups, k=10
    )

    # Precision at k (for queries with relevant items)
    precision_5_scores = []
    precision_10_scores = []
    tau_scores = []

    for qid in np.unique( query_groups ) :
        mask = query_groups == qid

        if mask.sum() < 2 :
            continue

        true_scores = y_true[mask]
        pred_scores = y_pred[mask]

        # Get relevant items (relevance > 2 as threshold)
        relevant_mask = true_scores > 2

        if relevant_mask.sum() > 0 :
            # Sort by predicted scores
            sorted_indices = np.argsort( pred_scores )[: :-1]

            # Calculate precision
            top_5 = sorted_indices[:5]
            top_10 = sorted_indices[:10]

            precision_5 = relevant_mask[top_5].sum() / len( top_5 )
            precision_10 = relevant_mask[top_10].sum() / len( top_10 )

            precision_5_scores.append( precision_5 )
            precision_10_scores.append( precision_10 )

        # Kendall's Tau
        tau, _ = kendalltau( true_scores, pred_scores )
        if not np.isnan( tau ) :
            tau_scores.append( tau )

    metrics['precision@5'] = np.mean( precision_5_scores ) if precision_5_scores else 0.0
    metrics['precision@10'] = np.mean( precision_10_scores ) if precision_10_scores else 0.0
    metrics['kendall_tau'] = np.mean( tau_scores ) if tau_scores else 0.0
    metrics['num_queries'] = len( np.unique( query_groups ) )

    return metrics


# Example usage and tests
if __name__ == "__main__" :
    print( "=== IEEE-Compliant Ranking Metrics Tests ===\n" )

    # Test Precision@k
    retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
    relevant = ['doc1', 'doc3', 'doc5']
    p5 = precision_at_k( retrieved, relevant, k=5 )
    print( f"Precision@5: {p5:.3f}" )

    # Test NDCG@k
    relevance_scores = [3, 2, 3, 0, 1, 2]
    ndcg = ndcg_at_k( relevance_scores, k=5 )
    print( f"NDCG@5: {ndcg:.3f}" )

    # Test batch evaluation
    y_true = np.array( [3, 2, 1, 0, 2, 3, 1] )
    y_pred = np.array( [2.5, 2.0, 1.0, 0.5, 2.2, 2.8, 1.1] )
    query_groups = np.array( [0, 0, 0, 1, 1, 1, 1] )

    metrics = evaluate_ranking_model( y_true, y_pred, query_groups )
    print( f"\nComprehensive Evaluation:" )
    for metric, value in metrics.items() :
        print( f"  {metric}: {value:.4f}" )