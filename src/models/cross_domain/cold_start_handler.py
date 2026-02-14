"""
Cold start handler for users with limited rating history.
Adaptively switches between genre-based popularity and full hybrid recommendations.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class ColdStartHandler:
    """Handles recommendations for both cold start and warmstart users."""

    def __init__(
        self,
        items_df: pd.DataFrame,
        hybrid_model: Optional[object] = None,
        cold_start_threshold: int = 10
    ):
        """
        Initialize cold start handler.

        Args:
            items_df: All items with media_type, genres, avg_rating, rating_count
            hybrid_model: Full hybrid model for warmstart users
            cold_start_threshold: Minimum ratings before using full model
        """
        self.items_df = items_df
        self.hybrid_model = hybrid_model
        self.cold_start_threshold = cold_start_threshold

        logger.info(f"Initialized ColdStartHandler")
        logger.info(f"  Items: {len(items_df):,}")
        logger.info(f"  Cold start threshold: {cold_start_threshold} ratings")
        logger.info(f"  Hybrid model: {'✅' if hybrid_model else '❌'}")

    def recommend(
        self,
        user_ratings: Dict[str, float],
        target_media: str = 'movie',
        top_n: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Generate recommendations with adaptive strategy.

        Args:
            user_ratings: Dict mapping item_id to rating
            target_media: Target media type
            top_n: Number of recommendations
            exclude_rated: Whether to exclude rated items

        Returns:
            List of (item_id, score) tuples
        """
        num_ratings = len(user_ratings)

        if num_ratings >= self.cold_start_threshold:
            # Warmstart: use full hybrid model
            logger.info(f"Warmstart user ({num_ratings} ratings) → using hybrid model")

            if self.hybrid_model:
                return self.hybrid_model.recommend(
                    user_ratings,
                    target_media=target_media,
                    top_n=top_n,
                    exclude_rated=exclude_rated
                )
            else:
                logger.warning("Hybrid model not available, falling back to cold start")
                return self._cold_start_recommend(user_ratings, target_media, top_n, exclude_rated)

        else:
            # Cold start: use genre-based popularity
            logger.info(f"Cold start user ({num_ratings} ratings) → using genre-based popularity")
            return self._cold_start_recommend(user_ratings, target_media, top_n, exclude_rated)

    def _cold_start_recommend(
        self,
        user_ratings: Dict[str, float],
        target_media: str,
        top_n: int,
        exclude_rated: bool
    ) -> List[Tuple[str, float]]:
        """
        Cold start recommendation: popular items in user's preferred genres.

        Algorithm:
        1. Extract genre preferences from rated items (weighted by rating)
        2. Filter to target media type
        3. Score items by: genre_match_count × popularity
        4. Return top-N

        Args:
            user_ratings: User ratings dict
            target_media: Target media type
            top_n: Number of recommendations
            exclude_rated: Whether to exclude rated items

        Returns:
            List of (item_id, score) tuples
        """
        # Extract preferred genres
        preferred_genres = self._extract_genre_preferences(user_ratings)

        if not preferred_genres:
            logger.warning("No genre preferences extracted, using pure popularity")
            return self._popularity_based_recommend(target_media, top_n, exclude_rated, user_ratings)

        logger.info(f"Preferred genres: {preferred_genres[:5]}")

        # Filter items by target media
        target_items = self.items_df[self.items_df['media_type'] == target_media].copy()

        if len(target_items) == 0:
            logger.warning(f"No items found for media type: {target_media}")
            return []

        # Score by genre match + popularity
        scores = []

        for _, item in target_items.iterrows():
            # Skip rated items if requested
            if exclude_rated and item['item_id'] in user_ratings:
                continue

            # Count genre matches
            item_genres = set(item['genres']) if isinstance(item['genres'], (list, np.ndarray)) else set()
            genre_match = len(item_genres & set(preferred_genres))

            if genre_match == 0:
                # No genre match, low score
                popularity = 0.1 * item['rating_count'] * item['avg_rating']
            else:
                # Genre match, boost by match count
                popularity = item['rating_count'] * item['avg_rating']
                score = genre_match * popularity
                scores.append((item['item_id'], float(score)))

        if not scores:
            # Fallback to pure popularity
            return self._popularity_based_recommend(target_media, top_n, exclude_rated, user_ratings)

        # Sort and return top-N
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_n]

    def _extract_genre_preferences(
        self,
        user_ratings: Dict[str, float]
    ) -> List[str]:
        """
        Extract preferred genres from user-rated items.

        Args:
            user_ratings: User ratings dict

        Returns:
            List of genres sorted by preference (weighted votes)
        """
        genre_votes = {}

        for item_id, rating in user_ratings.items():
            # Find item in items_df
            item_match = self.items_df[self.items_df['item_id'] == item_id]

            if len(item_match) == 0:
                logger.debug(f"Item {item_id} not found in items_df")
                continue

            item_genres = item_match.iloc[0]['genres']

            # Handle different genre formats
            if isinstance(item_genres, (list, np.ndarray)):
                genres = item_genres
            elif isinstance(item_genres, str):
                import ast
                try:
                    genres = ast.literal_eval(item_genres)
                except:
                    genres = [item_genres]
            else:
                continue

            # Weight genres by rating
            for genre in genres:
                if genre:
                    genre_votes[genre] = genre_votes.get(genre, 0.0) + float(rating)

        # Sort genres by weighted votes
        sorted_genres = sorted(genre_votes.items(), key=lambda x: x[1], reverse=True)

        return [genre for genre, _ in sorted_genres]

    def _popularity_based_recommend(
        self,
        target_media: str,
        top_n: int,
        exclude_rated: bool,
        user_ratings: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """
        Pure popularity-based recommendations (fallback).

        Args:
            target_media: Target media type
            top_n: Number of recommendations
            exclude_rated: Whether to exclude rated items
            user_ratings: User ratings (for exclusion)

        Returns:
            List of (item_id, score) tuples
        """
        logger.info("Using pure popularity-based recommendations")

        target_items = self.items_df[self.items_df['media_type'] == target_media].copy()

        # Score by popularity
        target_items['popularity'] = target_items['rating_count'] * target_items['avg_rating']

        # Exclude rated items
        if exclude_rated:
            target_items = target_items[~target_items['item_id'].isin(user_ratings.keys())]

        # Sort by popularity
        target_items = target_items.sort_values('popularity', ascending=False)

        # Return top-N
        recommendations = [
            (row['item_id'], float(row['popularity']))
            for _, row in target_items.head(top_n).iterrows()
        ]

        return recommendations

    def evaluate_strategy_split(
        self,
        all_users_ratings: List[Dict[str, float]]
    ) -> Dict:
        """
        Analyze how users split between cold start and warmstart.

        Args:
            all_users_ratings: List of user rating dicts

        Returns:
            Statistics dict
        """
        cold_start_count = 0
        warmstart_count = 0

        for user_ratings in all_users_ratings:
            if len(user_ratings) < self.cold_start_threshold:
                cold_start_count += 1
            else:
                warmstart_count += 1

        total = len(all_users_ratings)

        stats = {
            'total_users': total,
            'cold_start': cold_start_count,
            'warmstart': warmstart_count,
            'cold_start_pct': cold_start_count / total * 100 if total > 0 else 0,
            'warmstart_pct': warmstart_count / total * 100 if total > 0 else 0,
            'threshold': self.cold_start_threshold
        }

        logger.info(f"Strategy split:")
        logger.info(f"  Cold start: {cold_start_count} ({stats['cold_start_pct']:.1f}%)")
        logger.info(f"  Warmstart: {warmstart_count} ({stats['warmstart_pct']:.1f}%)")

        return stats

    def __repr__(self) -> str:
        return (
            f"ColdStartHandler(\n"
            f"  threshold={self.cold_start_threshold},\n"
            f"  items={len(self.items_df):,},\n"
            f"  hybrid={'available' if self.hybrid_model else 'unavailable'}\n"
            f")"
        )


# Convenience function
def create_cold_start_handler(
    items_df: pd.DataFrame,
    hybrid_model,
    threshold: int = 10
) -> ColdStartHandler:
    """
    Create a cold start handler.

    Args:
        items_df: Items DataFrame
        hybrid_model: Hybrid recommender
        threshold: Cold start threshold

    Returns:
        ColdStartHandler instance
    """
    return ColdStartHandler(items_df, hybrid_model, threshold)
