"""
Preference transfer for cross-domain recommendations.
Transfers user ratings from source media to target media recommendations.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class PreferenceTransfer:
    """Transfer user preferences across media domains."""

    def __init__(self, bridges: Dict[str, 'DomainBridge']):
        """
        Initialize preference transfer.

        Args:
            bridges: Dictionary mapping "source→target" to DomainBridge instances
                    Example: {'movie→anime': bridge1, 'movie→manga': bridge2}
        """
        self.bridges = bridges

        logger.info(f"Initialized PreferenceTransfer with {len(bridges)} bridges:")
        for key in bridges.keys():
            logger.info(f"  {key}")

    def transfer_ratings(
        self,
        user_ratings: Dict[str, float],
        source_media: str,
        target_media: str,
        top_n: int = 10,
        min_score: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Transfer user ratings to target media recommendations.

        Args:
            user_ratings: Dict mapping source item_id to rating
            source_media: Source media type ('movie', 'anime', etc.)
            target_media: Target media type
            top_n: Number of recommendations to return
            min_score: Minimum score threshold

        Returns:
            List of (target_item_id, score) tuples sorted by score descending
        """
        bridge_key = f"{source_media}→{target_media}"

        if bridge_key not in self.bridges:
            logger.error(f"Bridge not found: {bridge_key}")
            logger.error(f"Available bridges: {list(self.bridges.keys())}")
            return []

        bridge = self.bridges[bridge_key]

        if bridge.bridge_matrix is None:
            raise ValueError(f"Bridge {bridge_key} not built")

        if not user_ratings:
            logger.warning("No user ratings provided")
            return []

        logger.info(f"Transferring {len(user_ratings)} ratings from {source_media} to {target_media}")

        # Initialize scores for all target items
        num_target_items = bridge.bridge_matrix.shape[1]
        scores = np.zeros(num_target_items)

        # Accumulate weighted scores
        valid_source_count = 0

        for source_item_id, rating in user_ratings.items():
            if source_item_id not in bridge.source_item_to_idx:
                logger.debug(f"Source item {source_item_id} not in bridge, skipping")
                continue

            source_idx = bridge.source_item_to_idx[source_item_id]

            # Get similarity scores to all target items
            similarities = bridge.bridge_matrix.getrow(source_idx).toarray().flatten()

            # Weight by user rating and accumulate
            scores += rating * similarities
            valid_source_count += 1

        if valid_source_count == 0:
            logger.warning("None of the rated items found in bridge")
            return []

        logger.info(f"Used {valid_source_count}/{len(user_ratings)} rated items for transfer")

        # Normalize scores by number of rated items
        scores = scores / valid_source_count

        # Filter by minimum score
        valid_indices = np.where(scores >= min_score)[0]
        scores = scores[valid_indices]

        # Get top-N
        if len(scores) == 0:
            logger.warning("No items meet minimum score threshold")
            return []

        top_k = min(top_n, len(scores))
        top_indices = valid_indices[np.argsort(scores)[-top_k:][::-1]]
        top_scores = scores[np.argsort(scores)[-top_k:][::-1]]

        # Convert to item IDs
        recommendations = [
            (bridge.target_idx_to_item[idx], float(score))
            for idx, score in zip(top_indices, top_scores)
        ]

        logger.info(f"Generated {len(recommendations)} {target_media} recommendations")

        return recommendations

    def transfer_batch(
        self,
        user_ratings_batch: List[Dict[str, float]],
        source_media: str,
        target_media: str,
        top_n: int = 10
    ) -> List[List[Tuple[str, float]]]:
        """
        Transfer ratings for multiple users.

        Args:
            user_ratings_batch: List of user rating dicts
            source_media: Source media type
            target_media: Target media type
            top_n: Number of recommendations per user

        Returns:
            List of recommendation lists
        """
        recommendations = []

        for user_ratings in user_ratings_batch:
            recs = self.transfer_ratings(user_ratings, source_media, target_media, top_n)
            recommendations.append(recs)

        return recommendations

    def get_transfer_explanation(
        self,
        source_item_id: str,
        target_item_id: str,
        source_media: str,
        target_media: str
    ) -> Optional[Dict]:
        """
        Get explanation for why a target item was recommended.

        Args:
            source_item_id: Source item that user rated
            target_item_id: Target item that was recommended
            source_media: Source media type
            target_media: Target media type

        Returns:
            Dictionary with explanation details
        """
        bridge_key = f"{source_media}→{target_media}"

        if bridge_key not in self.bridges:
            return None

        bridge = self.bridges[bridge_key]

        if source_item_id not in bridge.source_item_to_idx:
            return None
        if target_item_id not in bridge.target_item_to_idx:
            return None

        source_idx = bridge.source_item_to_idx[source_item_id]
        target_idx = bridge.target_item_to_idx[target_item_id]

        # Get similarity score
        similarity = bridge.bridge_matrix[source_idx, target_idx]

        explanation = {
            'source_item': source_item_id,
            'target_item': target_item_id,
            'similarity': float(similarity),
            'source_media': source_media,
            'target_media': target_media
        }

        return explanation

    def list_available_bridges(self) -> List[str]:
        """
        List all available bridge connections.

        Returns:
            List of bridge keys
        """
        return list(self.bridges.keys())


# Convenience function
def create_preference_transfer(bridge_paths: Dict[str, str]) -> PreferenceTransfer:
    """
    Create PreferenceTransfer by loading bridges from files.

    Args:
        bridge_paths: Dict mapping bridge keys to file paths
                     Example: {'movie→anime': 'data/models/bridge_movie_to_anime.npz'}

    Returns:
        PreferenceTransfer instance
    """
    from models.cross_domain.domain_bridge import DomainBridge

    bridges = {}

    for bridge_key, filepath in bridge_paths.items():
        logger.info(f"Loading bridge: {bridge_key}")
        bridge = DomainBridge.load_bridge(filepath)
        bridges[bridge_key] = bridge

    return PreferenceTransfer(bridges)
