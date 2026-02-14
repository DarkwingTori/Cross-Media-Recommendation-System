"""
Genre/theme-based content filtering.
Recommends items similar to user-liked items based on genre and theme embeddings.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data_pipeline.transform.embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


class GenreBasedRecommender:
    """Content-based recommender using genre/theme TF-IDF embeddings."""

    def __init__(
        self,
        items_df: pd.DataFrame,
        embeddings: csr_matrix,
        item_to_idx: Dict[str, int],
        idx_to_item: Dict[int, str]
    ):
        """
        Initialize content-based recommender.

        Args:
            items_df: Items DataFrame with genres/themes
            embeddings: TF-IDF embeddings (num_items × num_features)
            item_to_idx: Mapping from item_id to embedding index
            idx_to_item: Mapping from embedding index to item_id
        """
        self.items_df = items_df
        self.embeddings = embeddings
        self.item_to_idx = item_to_idx
        self.idx_to_item = idx_to_item

        logger.info(f"Initialized GenreBasedRecommender")
        logger.info(f"  Items: {len(items_df):,}")
        logger.info(f"  Embedding shape: {embeddings.shape}")

    def recommend(
        self,
        user_ratings: Dict[str, float],
        top_n: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Generate content-based recommendations.

        Args:
            user_ratings: Dict mapping item_id to rating
            top_n: Number of recommendations to return
            exclude_rated: Whether to exclude already-rated items

        Returns:
            List of (item_id, score) tuples sorted by score descending
        """
        if not user_ratings:
            logger.warning("No user ratings provided")
            return []

        # Get indices of rated items
        rated_indices = []
        rating_values = []

        for item_id, rating in user_ratings.items():
            if item_id in self.item_to_idx:
                rated_indices.append(self.item_to_idx[item_id])
                rating_values.append(rating)
            else:
                logger.debug(f"Item {item_id} not in embeddings, skipping")

        if not rated_indices:
            logger.warning("None of the rated items found in embeddings")
            return []

        logger.info(f"Generating recommendations from {len(rated_indices)} rated items")

        # Create user profile: weighted average of rated item embeddings
        rated_embeddings = self.embeddings[rated_indices]
        rating_weights = np.array(rating_values)

        # Normalize weights to sum to 1
        rating_weights = rating_weights / rating_weights.sum()

        # Weighted average (user profile embedding)
        user_profile = rated_embeddings.T.dot(rating_weights)

        # Compute similarity to all items
        user_profile_csr = csr_matrix(user_profile).T  # Convert to row vector
        similarities = cosine_similarity(user_profile_csr, self.embeddings)
        scores = similarities.flatten()

        # Exclude rated items
        exclude_indices = set(rated_indices) if exclude_rated else set()
        for idx in exclude_indices:
            scores[idx] = -np.inf

        # Get top-N indices
        top_k = min(top_n, len(scores))
        top_indices = np.argsort(scores)[-top_k:][::-1]

        # Filter out -inf scores
        valid_mask = scores[top_indices] > -np.inf
        top_indices = top_indices[valid_mask]
        top_scores = scores[top_indices]

        # Convert to item IDs
        recommendations = [
            (self.idx_to_item[idx], float(score))
            for idx, score in zip(top_indices, top_scores)
        ]

        logger.info(f"Generated {len(recommendations)} recommendations")

        return recommendations

    @classmethod
    def from_embeddings_file(
        cls,
        items_df: pd.DataFrame,
        embeddings_path: str
    ) -> 'GenreBasedRecommender':
        """
        Create recommender from saved embeddings file.

        Args:
            items_df: Items DataFrame
            embeddings_path: Path to embeddings pickle file

        Returns:
            GenreBasedRecommender instance
        """
        logger.info(f"Loading content-based model from {embeddings_path}")

        # Load embeddings
        generator = EmbeddingGenerator()
        embeddings, item_to_idx, idx_to_item = generator.load_embeddings(embeddings_path)

        return cls(items_df, embeddings, item_to_idx, idx_to_item)

    def get_similar_items(
        self,
        item_id: str,
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get items most similar to a given item (genre/theme similarity).

        Args:
            item_id: Item ID
            top_n: Number of similar items to return

        Returns:
            List of (item_id, similarity_score) tuples
        """
        if item_id not in self.item_to_idx:
            logger.warning(f"Item {item_id} not found")
            return []

        item_idx = self.item_to_idx[item_id]

        # Get item embedding
        item_embedding = self.embeddings.getrow(item_idx)

        # Compute similarity to all items
        similarities = cosine_similarity(item_embedding, self.embeddings).flatten()

        # Exclude the item itself
        similarities[item_idx] = -np.inf

        # Get top-N
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        top_scores = similarities[top_indices]

        similar_items = [
            (self.idx_to_item[idx], float(score))
            for idx, score in zip(top_indices, top_scores)
            if score > -np.inf
        ]

        return similar_items

    def get_model_info(self) -> Dict:
        """Get information about the model."""
        return {
            'num_items': len(self.item_to_idx),
            'embedding_shape': self.embeddings.shape,
            'embedding_sparsity': 1.0 - (self.embeddings.nnz / (self.embeddings.shape[0] * self.embeddings.shape[1]))
        }

    def __repr__(self) -> str:
        info = self.get_model_info()
        return (
            f"GenreBasedRecommender(\n"
            f"  items={info['num_items']:,},\n"
            f"  features={info['embedding_shape'][1]},\n"
            f"  sparsity={info['embedding_sparsity']:.2%}\n"
            f")"
        )


# Convenience function
def create_genre_based_recommender(
    items_path: str,
    embeddings_path: str
) -> GenreBasedRecommender:
    """
    Create genre-based recommender from files.

    Args:
        items_path: Path to items parquet file
        embeddings_path: Path to embeddings pickle file

    Returns:
        GenreBasedRecommender instance
    """
    items_df = pd.read_parquet(items_path)
    return GenreBasedRecommender.from_embeddings_file(items_df, embeddings_path)
