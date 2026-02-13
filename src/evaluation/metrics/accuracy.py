"""
Accuracy metrics for recommendation systems.
Implements standard metrics: Precision@K, Recall@K, NDCG@K, F1@K.
"""

import numpy as np
from typing import List, Set, Dict, Union
import logging

logger = logging.getLogger(__name__)


def precision_at_k(
    true_items: Union[List, Set],
    recommended_items: List,
    k: int = 10
) -> float:
    """
    Calculate Precision@K.

    Precision@K = (# of recommended items in top-K that are relevant) / K

    Args:
        true_items: Set or list of relevant items
        recommended_items: List of recommended items (ordered by score)
        k: Number of top recommendations to consider

    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if not recommended_items or k <= 0:
        return 0.0

    # Convert to set for faster lookup
    true_set = set(true_items) if not isinstance(true_items, set) else true_items

    # Consider only top-k recommendations
    top_k = recommended_items[:k]

    # Count relevant items in top-k
    relevant_in_top_k = len([item for item in top_k if item in true_set])

    return relevant_in_top_k / min(k, len(top_k))


def recall_at_k(
    true_items: Union[List, Set],
    recommended_items: List,
    k: int = 10
) -> float:
    """
    Calculate Recall@K.

    Recall@K = (# of recommended items in top-K that are relevant) / (# of relevant items)

    Args:
        true_items: Set or list of relevant items
        recommended_items: List of recommended items (ordered by score)
        k: Number of top recommendations to consider

    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if not recommended_items or k <= 0:
        return 0.0

    # Convert to set for faster lookup
    true_set = set(true_items) if not isinstance(true_items, set) else true_items

    if not true_set:
        return 0.0

    # Consider only top-k recommendations
    top_k = recommended_items[:k]

    # Count relevant items in top-k
    relevant_in_top_k = len([item for item in top_k if item in true_set])

    return relevant_in_top_k / len(true_set)


def f1_at_k(
    true_items: Union[List, Set],
    recommended_items: List,
    k: int = 10
) -> float:
    """
    Calculate F1@K (harmonic mean of Precision@K and Recall@K).

    F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)

    Args:
        true_items: Set or list of relevant items
        recommended_items: List of recommended items
        k: Number of top recommendations to consider

    Returns:
        F1@K score (0.0 to 1.0)
    """
    precision = precision_at_k(true_items, recommended_items, k)
    recall = recall_at_k(true_items, recommended_items, k)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def ndcg_at_k(
    true_items: Union[List, Set, Dict[str, float]],
    recommended_items: List,
    k: int = 10
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain@K.

    NDCG@K considers both relevance and ranking position.
    Items ranked higher have more impact on the score.

    Args:
        true_items: Set/list of relevant items, or dict mapping item to relevance score
        recommended_items: List of recommended items (ordered by score)
        k: Number of top recommendations to consider

    Returns:
        NDCG@K score (0.0 to 1.0)
    """
    if not recommended_items or k <= 0:
        return 0.0

    # Handle relevance scores
    if isinstance(true_items, dict):
        relevance = true_items
    else:
        # Binary relevance: 1 if relevant, 0 otherwise
        true_set = set(true_items) if not isinstance(true_items, set) else true_items
        relevance = {item: 1.0 for item in true_set}

    if not relevance:
        return 0.0

    # Consider only top-k recommendations
    top_k = recommended_items[:k]

    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevance:
            # Relevance discounted by log position (1-indexed)
            # DCG = sum(rel_i / log2(i + 1))
            dcg += relevance[item] / np.log2(i + 2)  # i+2 because i is 0-indexed

    # Calculate IDCG (Ideal DCG)
    # Sort relevance scores in descending order
    ideal_relevances = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(
        rel / np.log2(i + 2)
        for i, rel in enumerate(ideal_relevances)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mean_reciprocal_rank(
    true_items: Union[List, Set],
    recommended_items: List
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    MRR = 1 / rank_of_first_relevant_item

    Args:
        true_items: Set or list of relevant items
        recommended_items: List of recommended items (ordered by score)

    Returns:
        MRR score (0.0 to 1.0)
    """
    if not recommended_items:
        return 0.0

    # Convert to set for faster lookup
    true_set = set(true_items) if not isinstance(true_items, set) else true_items

    if not true_set:
        return 0.0

    # Find rank of first relevant item (1-indexed)
    for i, item in enumerate(recommended_items):
        if item in true_set:
            return 1.0 / (i + 1)

    return 0.0


def average_precision_at_k(
    true_items: Union[List, Set],
    recommended_items: List,
    k: int = 10
) -> float:
    """
    Calculate Average Precision@K.

    AP@K = (sum of P@i for each relevant item i in top-K) / min(# relevant, K)

    Args:
        true_items: Set or list of relevant items
        recommended_items: List of recommended items
        k: Number of top recommendations to consider

    Returns:
        Average Precision@K score (0.0 to 1.0)
    """
    if not recommended_items or k <= 0:
        return 0.0

    # Convert to set for faster lookup
    true_set = set(true_items) if not isinstance(true_items, set) else true_items

    if not true_set:
        return 0.0

    # Consider only top-k
    top_k = recommended_items[:k]

    # Calculate precision at each position where a relevant item appears
    num_relevant = 0
    precision_sum = 0.0

    for i, item in enumerate(top_k):
        if item in true_set:
            num_relevant += 1
            # Precision at this position
            precision_at_i = num_relevant / (i + 1)
            precision_sum += precision_at_i

    if num_relevant == 0:
        return 0.0

    return precision_sum / min(len(true_set), k)


def hit_rate_at_k(
    true_items: Union[List, Set],
    recommended_items: List,
    k: int = 10
) -> float:
    """
    Calculate Hit Rate@K (binary: 1 if any relevant item in top-K, 0 otherwise).

    Args:
        true_items: Set or list of relevant items
        recommended_items: List of recommended items
        k: Number of top recommendations to consider

    Returns:
        1.0 if hit, 0.0 otherwise
    """
    if not recommended_items or k <= 0:
        return 0.0

    # Convert to set for faster lookup
    true_set = set(true_items) if not isinstance(true_items, set) else true_items

    if not true_set:
        return 0.0

    # Check if any item in top-k is relevant
    top_k = recommended_items[:k]

    for item in top_k:
        if item in true_set:
            return 1.0

    return 0.0


class MetricsCalculator:
    """Calculate multiple metrics at once."""

    @staticmethod
    def calculate_all(
        true_items: Union[List, Set, Dict[str, float]],
        recommended_items: List,
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        Calculate all metrics at multiple K values.

        Args:
            true_items: Relevant items (set/list for binary, dict for relevance scores)
            recommended_items: Recommended items list
            k_values: List of K values to evaluate

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        # Extract item IDs if recommended_items contains tuples
        if recommended_items and isinstance(recommended_items[0], tuple):
            rec_item_ids = [item[0] for item in recommended_items]
        else:
            rec_item_ids = recommended_items

        for k in k_values:
            metrics[f'precision@{k}'] = precision_at_k(true_items, rec_item_ids, k)
            metrics[f'recall@{k}'] = recall_at_k(true_items, rec_item_ids, k)
            metrics[f'f1@{k}'] = f1_at_k(true_items, rec_item_ids, k)
            metrics[f'ndcg@{k}'] = ndcg_at_k(true_items, rec_item_ids, k)
            metrics[f'hit_rate@{k}'] = hit_rate_at_k(true_items, rec_item_ids, k)
            metrics[f'ap@{k}'] = average_precision_at_k(true_items, rec_item_ids, k)

        # MRR doesn't depend on K
        metrics['mrr'] = mean_reciprocal_rank(true_items, rec_item_ids)

        return metrics


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple users/queries.

    Args:
        metrics_list: List of metric dictionaries

    Returns:
        Dictionary with averaged metrics
    """
    if not metrics_list:
        return {}

    # Get all metric names
    metric_names = metrics_list[0].keys()

    # Average each metric
    aggregated = {}
    for name in metric_names:
        values = [m[name] for m in metrics_list if name in m]
        aggregated[name] = np.mean(values) if values else 0.0
        aggregated[f'{name}_std'] = np.std(values) if values else 0.0

    return aggregated
