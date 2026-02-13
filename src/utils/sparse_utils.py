"""
Sparse matrix utilities for efficient recommendation computations.
Handles user-item matrices and similarity computations using scipy sparse matrices.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def build_user_item_matrix(
    ratings_df: pd.DataFrame,
    user_col: str = 'user_id',
    item_col: str = 'item_id',
    rating_col: str = 'rating'
) -> Tuple[csr_matrix, Dict[str, int], Dict[str, int], Dict[int, str], Dict[int, str]]:
    """
    Build a sparse user-item matrix from ratings DataFrame.

    Args:
        ratings_df: DataFrame with user_id, item_id, rating columns
        user_col: Name of user ID column
        item_col: Name of item ID column
        rating_col: Name of rating column

    Returns:
        Tuple of:
        - user_item_matrix: CSR sparse matrix (users x items)
        - user_to_idx: Dict mapping user_id to matrix row index
        - item_to_idx: Dict mapping item_id to matrix column index
        - idx_to_user: Dict mapping matrix row index to user_id
        - idx_to_item: Dict mapping matrix column index to item_id
    """
    logger.info(f"Building user-item matrix from {len(ratings_df):,} ratings")

    # Create user and item mappings
    unique_users = ratings_df[user_col].unique()
    unique_items = ratings_df[item_col].unique()

    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}

    logger.info(f"Matrix dimensions: {len(unique_users):,} users x {len(unique_items):,} items")

    # Map IDs to indices
    user_indices = ratings_df[user_col].map(user_to_idx).values
    item_indices = ratings_df[item_col].map(item_to_idx).values
    ratings = ratings_df[rating_col].values

    # Build sparse matrix
    user_item_matrix = csr_matrix(
        (ratings, (user_indices, item_indices)),
        shape=(len(unique_users), len(unique_items))
    )

    sparsity = 1.0 - (user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]))
    logger.info(f"Matrix sparsity: {sparsity:.4%} ({user_item_matrix.nnz:,} non-zero entries)")

    return user_item_matrix, user_to_idx, item_to_idx, idx_to_user, idx_to_item


def save_sparse_matrix(matrix: csr_matrix, filepath: str) -> None:
    """
    Save a sparse matrix to NPZ format.

    Args:
        matrix: Sparse matrix to save
        filepath: Path to save file (should end in .npz)
    """
    logger.info(f"Saving sparse matrix to {filepath}")
    save_npz(filepath, matrix)
    logger.info(f"Matrix saved successfully")


def load_sparse_matrix(filepath: str) -> csr_matrix:
    """
    Load a sparse matrix from NPZ format.

    Args:
        filepath: Path to NPZ file

    Returns:
        Loaded sparse matrix
    """
    logger.info(f"Loading sparse matrix from {filepath}")
    matrix = load_npz(filepath)
    logger.info(f"Matrix loaded: shape {matrix.shape}, {matrix.nnz:,} non-zero entries")
    return matrix


def get_top_k_sparse(sparse_row: csr_matrix, k: int, exclude_indices: Optional[set] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get top-k values and indices from a sparse matrix row efficiently.

    Args:
        sparse_row: Sparse matrix row (1 x n)
        k: Number of top items to return
        exclude_indices: Optional set of indices to exclude from results

    Returns:
        Tuple of (top_k_indices, top_k_scores)
    """
    # Convert to dense array for the row
    row_data = sparse_row.toarray().flatten()

    # Exclude specified indices
    if exclude_indices:
        row_data[list(exclude_indices)] = -np.inf

    # Get top-k indices using argpartition for efficiency
    # argpartition is O(n) vs argsort which is O(n log n)
    if k >= len(row_data):
        top_k_indices = np.argsort(row_data)[::-1]
    else:
        # Partition to find k largest elements
        partition_indices = np.argpartition(row_data, -k)[-k:]
        # Sort just the top-k elements
        top_k_indices = partition_indices[np.argsort(row_data[partition_indices])[::-1]]

    top_k_scores = row_data[top_k_indices]

    # Filter out any non-positive scores
    valid_mask = top_k_scores > 0
    top_k_indices = top_k_indices[valid_mask]
    top_k_scores = top_k_scores[valid_mask]

    return top_k_indices, top_k_scores


def keep_top_k_per_row(matrix: csr_matrix, k: int) -> csr_matrix:
    """
    Keep only top-k values per row in a sparse matrix.
    Useful for keeping only the most similar items in a similarity matrix.

    Args:
        matrix: Input sparse matrix
        k: Number of top values to keep per row

    Returns:
        Sparse matrix with only top-k values per row
    """
    logger.info(f"Keeping top-{k} values per row")

    rows = []
    cols = []
    data = []

    for i in range(matrix.shape[0]):
        row = matrix.getrow(i)
        row_data = row.data
        row_indices = row.indices

        if len(row_data) <= k:
            # Keep all values if fewer than k
            rows.extend([i] * len(row_data))
            cols.extend(row_indices)
            data.extend(row_data)
        else:
            # Keep top-k values
            top_k_idx = np.argpartition(row_data, -k)[-k:]
            rows.extend([i] * k)
            cols.extend(row_indices[top_k_idx])
            data.extend(row_data[top_k_idx])

    result = csr_matrix(
        (data, (rows, cols)),
        shape=matrix.shape
    )

    original_nnz = matrix.nnz
    new_nnz = result.nnz
    reduction = 100 * (1 - new_nnz / original_nnz) if original_nnz > 0 else 0
    logger.info(f"Matrix size reduced: {original_nnz:,} → {new_nnz:,} non-zero entries ({reduction:.1f}% reduction)")

    return result


def normalize_matrix_rows(matrix: csr_matrix, norm: str = 'l2') -> csr_matrix:
    """
    Normalize rows of a sparse matrix.

    Args:
        matrix: Input sparse matrix
        norm: Normalization type ('l1' or 'l2')

    Returns:
        Row-normalized sparse matrix
    """
    from sklearn.preprocessing import normalize
    return normalize(matrix, norm=norm, axis=1)


def compute_matrix_stats(matrix: csr_matrix) -> Dict[str, float]:
    """
    Compute statistics about a sparse matrix.

    Args:
        matrix: Sparse matrix

    Returns:
        Dictionary with statistics
    """
    total_elements = matrix.shape[0] * matrix.shape[1]
    sparsity = 1.0 - (matrix.nnz / total_elements) if total_elements > 0 else 1.0

    stats = {
        'shape': matrix.shape,
        'nnz': matrix.nnz,
        'sparsity': sparsity,
        'density': 1.0 - sparsity,
        'avg_nnz_per_row': matrix.nnz / matrix.shape[0] if matrix.shape[0] > 0 else 0,
    }

    # Compute statistics on non-zero values
    if matrix.nnz > 0:
        stats['min_value'] = matrix.data.min()
        stats['max_value'] = matrix.data.max()
        stats['mean_value'] = matrix.data.mean()
        stats['std_value'] = matrix.data.std()

    return stats
