"""
Weighted hybrid recommender combining multiple recommendation signals.
Orchestrates CF, content-based, and cross-domain models with configurable weights.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class WeightedHybridRecommender:
    """Hybrid recommender combining CF + content + cross-domain."""

    def __init__(
        self,
        cf_model: Optional[object] = None,
        content_model: Optional[object] = None,
        preference_transfer: Optional[object] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize hybrid recommender.

        Args:
            cf_model: Collaborative filtering model (ItemBasedCF)
            content_model: Content-based model (GenreBasedRecommender)
            preference_transfer: Cross-domain transfer (PreferenceTransfer)
            weights: Dict with keys 'cf', 'content', 'cross_domain'
        """
        self.cf_model = cf_model
        self.content_model = content_model
        self.preference_transfer = preference_transfer

        # Default weights (can be optimized)
        self.weights = weights or {
            'cf': 0.5,
            'content': 0.3,
            'cross_domain': 0.2
        }

        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}

        logger.info("Initialized WeightedHybridRecommender")
        logger.info(f"  CF model: {'✅' if cf_model else '❌'}")
        logger.info(f"  Content model: {'✅' if content_model else '❌'}")
        logger.info(f"  Preference transfer: {'✅' if preference_transfer else '❌'}")
        logger.info(f"  Weights: {self.weights}")

    def recommend(
        self,
        user_ratings: Dict[str, float],
        target_media: str = 'movie',
        top_n: int = 10,
        exclude_rated: bool = True,
        top_k_per_model: int = 100
    ) -> List[Tuple[str, float]]:
        """
        Generate hybrid recommendations.

        Args:
            user_ratings: Dict mapping item_id to rating
            target_media: Target media type ('movie', 'anime', 'manga')
            top_n: Number of recommendations to return
            exclude_rated: Whether to exclude already-rated items
            top_k_per_model: How many items to get from each model before combining

        Returns:
            List of (item_id, score) tuples sorted by score descending
        """
        if not user_ratings:
            logger.warning("No user ratings provided")
            return []

        logger.info(f"Generating hybrid recommendations for {target_media}")
        logger.info(f"  User ratings: {len(user_ratings)}")

        all_scores = {}  # Accumulator for weighted scores

        # 1. Collaborative Filtering (within-domain only)
        if self.cf_model and target_media == 'movie' and self.weights.get('cf', 0) > 0:
            try:
                logger.info("Getting CF recommendations...")
                cf_recs = self.cf_model.recommend(user_ratings, top_n=top_k_per_model, exclude_rated=False)

                if cf_recs:
                    cf_scores = self._normalize_scores(cf_recs)
                    self._add_weighted_scores(all_scores, cf_scores, self.weights['cf'])
                    logger.info(f"  CF: {len(cf_recs)} recommendations (weight={self.weights['cf']})")

            except Exception as e:
                logger.warning(f"CF model failed: {e}")

        # 2. Content-Based (genre/theme similarity)
        if self.content_model and self.weights.get('content', 0) > 0:
            try:
                logger.info("Getting content-based recommendations...")
                content_recs = self.content_model.recommend(user_ratings, top_n=top_k_per_model, exclude_rated=False)

                if content_recs:
                    content_scores = self._normalize_scores(content_recs)
                    self._add_weighted_scores(all_scores, content_scores, self.weights['content'])
                    logger.info(f"  Content: {len(content_recs)} recommendations (weight={self.weights['content']})")

            except Exception as e:
                logger.warning(f"Content model failed: {e}")

        # 3. Cross-Domain Transfer (for non-movie targets)
        if self.preference_transfer and target_media != 'movie' and self.weights.get('cross_domain', 0) > 0:
            try:
                logger.info(f"Getting cross-domain recommendations (movie→{target_media})...")
                xd_recs = self.preference_transfer.transfer_ratings(
                    user_ratings,
                    source_media='movie',
                    target_media=target_media,
                    top_n=top_k_per_model
                )

                if xd_recs:
                    xd_scores = self._normalize_scores(xd_recs)
                    self._add_weighted_scores(all_scores, xd_scores, self.weights['cross_domain'])
                    logger.info(f"  Cross-domain: {len(xd_recs)} recommendations (weight={self.weights['cross_domain']})")

            except Exception as e:
                logger.warning(f"Cross-domain transfer failed: {e}")

        if not all_scores:
            logger.warning("No recommendations generated from any model")
            return []

        # Exclude rated items
        if exclude_rated:
            for item_id in user_ratings.keys():
                all_scores.pop(item_id, None)

        # Sort by score and return top-N
        sorted_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        logger.info(f"Hybrid generated {len(sorted_items)} unique recommendations")

        return sorted_items[:top_n]

    @staticmethod
    def _normalize_scores(recs: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        Normalize scores to [0, 1] range using min-max scaling.

        Args:
            recs: List of (item_id, score) tuples

        Returns:
            Dict mapping item_id to normalized score
        """
        if not recs:
            return {}

        # Extract scores
        scores = np.array([score for _, score in recs])

        # Min-max normalization
        min_score = scores.min()
        max_score = scores.max()

        if max_score > min_score:
            normalized = (scores - min_score) / (max_score - min_score)
        else:
            # All scores same, set to 0.5
            normalized = np.ones_like(scores) * 0.5

        # Create mapping
        normalized_dict = {
            item_id: float(norm_score)
            for (item_id, _), norm_score in zip(recs, normalized)
        }

        return normalized_dict

    @staticmethod
    def _add_weighted_scores(
        all_scores: Dict[str, float],
        new_scores: Dict[str, float],
        weight: float
    ) -> None:
        """
        Add weighted scores to accumulator dict.

        Args:
            all_scores: Accumulator dict (modified in place)
            new_scores: New scores to add
            weight: Weight to apply to new scores
        """
        for item_id, score in new_scores.items():
            all_scores[item_id] = all_scores.get(item_id, 0.0) + weight * score

    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Update model weights.

        Args:
            weights: New weights dict
        """
        self.weights = weights

        # Normalize to sum to 1.0
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}

        logger.info(f"Updated weights: {self.weights}")

    def save_config(self, filepath: str) -> None:
        """
        Save hybrid configuration (weights).

        Args:
            filepath: Path to save config
        """
        config = {
            'weights': self.weights
        }

        with open(filepath, 'wb') as f:
            pickle.dump(config, f)

        logger.info(f"Saved hybrid config to {filepath}")

    @classmethod
    def load_config(cls, filepath: str) -> Dict:
        """
        Load hybrid configuration.

        Args:
            filepath: Path to config file

        Returns:
            Configuration dict
        """
        with open(filepath, 'rb') as f:
            config = pickle.load(f)

        logger.info(f"Loaded hybrid config from {filepath}")
        return config

    def get_model_info(self) -> Dict:
        """Get information about the hybrid model."""
        return {
            'has_cf': self.cf_model is not None,
            'has_content': self.content_model is not None,
            'has_cross_domain': self.preference_transfer is not None,
            'weights': self.weights,
            'normalized_weights': sum(self.weights.values()) == 1.0
        }

    def __repr__(self) -> str:
        info = self.get_model_info()
        models = []
        if info['has_cf']:
            models.append(f"CF (w={self.weights.get('cf', 0):.2f})")
        if info['has_content']:
            models.append(f"Content (w={self.weights.get('content', 0):.2f})")
        if info['has_cross_domain']:
            models.append(f"CrossDomain (w={self.weights.get('cross_domain', 0):.2f})")

        return f"WeightedHybridRecommender({', '.join(models)})"


# Convenience function
def create_hybrid_recommender(
    cf_model,
    content_model,
    preference_transfer,
    weights: Optional[Dict[str, float]] = None
) -> WeightedHybridRecommender:
    """
    Create a hybrid recommender with all models.

    Args:
        cf_model: Collaborative filtering model
        content_model: Content-based model
        preference_transfer: Cross-domain transfer
        weights: Optional custom weights

    Returns:
        WeightedHybridRecommender instance
    """
    return WeightedHybridRecommender(
        cf_model=cf_model,
        content_model=content_model,
        preference_transfer=preference_transfer,
        weights=weights
    )
