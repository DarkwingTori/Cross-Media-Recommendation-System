"""
Item-Based Collaborative Filtering with Cosine Similarity.
Implements memory-efficient CF using sparse matrices.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.sparse_utils import (
    build_user_item_matrix,
    save_sparse_matrix,
    load_sparse_matrix,
    keep_top_k_per_row,
    get_top_k_sparse
)

logger = logging.getLogger(__name__)


class ItemBasedCF:
    """Item-based collaborative filtering with cosine similarity."""

    def __init__(self, k_neighbors: int = 50):
        """
        Initialize item-based CF model.

        Args:
            k_neighbors: Number of similar items to consider per item
        """
        self.k_neighbors = k_neighbors
        self.similarity_matrix = None
        self.item_to_idx = None
        self.idx_to_item = None
        self.user_to_idx = None
        self.idx_to_user = None
        self.user_item_matrix = None

        logger.info(f"Initialized ItemBasedCF with k_neighbors={k_neighbors}")

    def fit(self, ratings_df: pd.DataFrame) -> 'ItemBasedCF':
        """
        Fit the collaborative filtering model.

        Args:
            ratings_df: DataFrame with columns: user_id, item_id, rating

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting ItemBasedCF on {len(ratings_df):,} ratings")

        # Build user-item matrix
        logger.info("Building user-item matrix...")
        (
            self.user_item_matrix,
            self.user_to_idx,
            self.item_to_idx,
            self.idx_to_user,
            self.idx_to_item
        ) = build_user_item_matrix(ratings_df)

        # Compute item-item similarity
        logger.info("Computing item-item similarity matrix...")
        item_item_matrix = self.user_item_matrix.T  # Transpose to get item-user matrix

        # Use cosine similarity (sklearn handles sparse matrices efficiently)
        self.similarity_matrix = cosine_similarity(item_item_matrix, dense_output=False)

        logger.info(f"Similarity matrix shape: {self.similarity_matrix.shape}")
        logger.info(f"Similarity matrix non-zero entries: {self.similarity_matrix.nnz:,}")

        # Keep only top-k similar items per item for efficiency
        if self.k_neighbors > 0:
            logger.info(f"Keeping top-{self.k_neighbors} neighbors per item...")
            self.similarity_matrix = keep_top_k_per_row(
                self.similarity_matrix,
                self.k_neighbors
            )

        # Zero out diagonal (item similarity with itself)
        self.similarity_matrix.setdiag(0)
        self.similarity_matrix.eliminate_zeros()

        logger.info("Model fitting complete")

        return self

    def recommend(
        self,
        user_ratings: Dict[str, float],
        top_n: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user based on their ratings.

        Args:
            user_ratings: Dict mapping item_id to rating
            top_n: Number of recommendations to return
            exclude_rated: Whether to exclude already-rated items

        Returns:
            List of (item_id, score) tuples sorted by score descending
        """
        if self.similarity_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if not user_ratings:
            logger.warning("No user ratings provided, returning empty recommendations")
            return []

        # Convert user ratings to indices
        item_indices = []
        ratings_vector = []

        for item_id, rating in user_ratings.items():
            if item_id in self.item_to_idx:
                item_indices.append(self.item_to_idx[item_id])
                ratings_vector.append(rating)
            else:
                logger.debug(f"Item {item_id} not in training data, skipping")

        if not item_indices:
            logger.warning("None of the rated items found in training data")
            return []

        # Convert to numpy array
        item_indices = np.array(item_indices)
        ratings_vector = np.array(ratings_vector)

        # Compute recommendation scores
        # Score = weighted sum of similarities to rated items
        # scores[i] = sum(similarity[rated_item_j, i] * rating_j)
        similarity_subset = self.similarity_matrix[item_indices, :]  # Shape: (num_rated_items, num_all_items)

        # Compute weighted scores
        scores = similarity_subset.T.dot(ratings_vector)  # Shape: (num_all_items,)

        # Exclude already-rated items if requested
        exclude_indices = set(item_indices) if exclude_rated else set()

        # Get top-N items
        top_k_indices, top_k_scores = get_top_k_sparse(
            csr_matrix(scores).reshape(1, -1),
            k=top_n,
            exclude_indices=exclude_indices
        )

        # Convert back to item IDs
        recommendations = [
            (self.idx_to_item[idx], float(score))
            for idx, score in zip(top_k_indices, top_k_scores)
        ]

        return recommendations

    def recommend_batch(
        self,
        user_ratings_batch: List[Dict[str, float]],
        top_n: int = 10,
        exclude_rated: bool = True
    ) -> List[List[Tuple[str, float]]]:
        """
        Generate recommendations for multiple users.

        Args:
            user_ratings_batch: List of user rating dicts
            top_n: Number of recommendations per user
            exclude_rated: Whether to exclude already-rated items

        Returns:
            List of recommendation lists
        """
        recommendations = []

        for user_ratings in user_ratings_batch:
            recs = self.recommend(user_ratings, top_n=top_n, exclude_rated=exclude_rated)
            recommendations.append(recs)

        return recommendations

    def get_similar_items(
        self,
        item_id: str,
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get most similar items to a given item.

        Args:
            item_id: Item ID
            top_n: Number of similar items to return

        Returns:
            List of (item_id, similarity_score) tuples
        """
        if self.similarity_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if item_id not in self.item_to_idx:
            logger.warning(f"Item {item_id} not found in training data")
            return []

        item_idx = self.item_to_idx[item_id]

        # Get similarity row for this item
        similarity_row = self.similarity_matrix.getrow(item_idx)

        # Get top-k similar items (excluding the item itself)
        top_k_indices, top_k_scores = get_top_k_sparse(
            similarity_row,
            k=top_n,
            exclude_indices={item_idx}
        )

        # Convert to item IDs
        similar_items = [
            (self.idx_to_item[idx], float(score))
            for idx, score in zip(top_k_indices, top_k_scores)
        ]

        return similar_items

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save file (NPZ format)
        """
        if self.similarity_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")

        logger.info(f"Saving model to {filepath}")

        # Save similarity matrix
        save_sparse_matrix(self.similarity_matrix, filepath)

        # Save mappings
        mappings_path = Path(filepath).with_suffix('.mappings.npz')
        np.savez(
            mappings_path,
            item_to_idx=self.item_to_idx,
            idx_to_item=self.idx_to_item,
            user_to_idx=self.user_to_idx,
            idx_to_user=self.idx_to_user,
            k_neighbors=self.k_neighbors
        )

        logger.info(f"Model saved successfully")

    def load_model(self, filepath: str) -> 'ItemBasedCF':
        """
        Load a trained model from disk.

        Args:
            filepath: Path to model file

        Returns:
            Self for method chaining
        """
        logger.info(f"Loading model from {filepath}")

        # Load similarity matrix
        self.similarity_matrix = load_sparse_matrix(filepath)

        # Load mappings
        mappings_path = Path(filepath).with_suffix('.mappings.npz')
        data = np.load(mappings_path, allow_pickle=True)

        self.item_to_idx = data['item_to_idx'].item()
        self.idx_to_item = data['idx_to_item'].item()
        self.user_to_idx = data['user_to_idx'].item()
        self.idx_to_user = data['idx_to_user'].item()
        self.k_neighbors = int(data['k_neighbors'])

        logger.info(f"Model loaded successfully")
        logger.info(f"  Items: {len(self.item_to_idx):,}")
        logger.info(f"  Users: {len(self.user_to_idx):,}")
        logger.info(f"  k_neighbors: {self.k_neighbors}")

        return self

    def get_model_info(self) -> Dict:
        """
        Get information about the fitted model.

        Returns:
            Dictionary with model statistics
        """
        if self.similarity_matrix is None:
            return {'fitted': False}

        return {
            'fitted': True,
            'num_items': len(self.item_to_idx),
            'num_users': len(self.user_to_idx),
            'k_neighbors': self.k_neighbors,
            'similarity_matrix_shape': self.similarity_matrix.shape,
            'similarity_matrix_nnz': self.similarity_matrix.nnz,
            'avg_neighbors_per_item': self.similarity_matrix.nnz / self.similarity_matrix.shape[0]
        }

    def __repr__(self) -> str:
        info = self.get_model_info()

        if not info['fitted']:
            return "ItemBasedCF(not fitted)"

        return (
            f"ItemBasedCF(\n"
            f"  items={info['num_items']:,},\n"
            f"  users={info['num_users']:,},\n"
            f"  k_neighbors={info['k_neighbors']},\n"
            f"  avg_neighbors={info['avg_neighbors_per_item']:.1f}\n"
            f")"
        )


# Convenience function for training
def train_item_based_cf(
    ratings_df: pd.DataFrame,
    k_neighbors: int = 50
) -> ItemBasedCF:
    """
    Train an item-based CF model.

    Args:
        ratings_df: DataFrame with user_id, item_id, rating columns
        k_neighbors: Number of neighbors to keep

    Returns:
        Fitted ItemBasedCF model
    """
    model = ItemBasedCF(k_neighbors=k_neighbors)
    model.fit(ratings_df)
    return model
