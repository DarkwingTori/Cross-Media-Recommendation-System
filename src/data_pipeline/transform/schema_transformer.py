"""
Schema transformation utilities for unified data format.
Transforms media-specific data formats into a unified schema.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from data_pipeline.preprocessing.genre_mapper import GenreMapper

logger = logging.getLogger(__name__)


class SchemaTransformer:
    """Transforms media data to unified schema."""

    def __init__(self, genre_mapper: Optional[GenreMapper] = None):
        """
        Initialize schema transformer.

        Args:
            genre_mapper: GenreMapper instance (creates new one if None)
        """
        self.genre_mapper = genre_mapper or GenreMapper()
        logger.info("Initialized SchemaTransformer")

    def transform_movies(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        tags_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Transform MovieLens movies to unified item schema.

        Unified Item Schema:
            item_id: str (e.g., 'mov_12345')
            media_type: str ('movie')
            title: str
            year: int
            genres: List[str] (universal genres)
            themes: List[str] (theme keywords)
            avg_rating: float
            rating_count: int

        Args:
            movies_df: Cleaned movies DataFrame
            ratings_df: Cleaned ratings DataFrame (for statistics)
            tags_df: Optional cleaned tags DataFrame (for themes)

        Returns:
            DataFrame with unified schema
        """
        logger.info(f"Transforming {len(movies_df):,} movies to unified schema")

        df = movies_df.copy()

        # Add 'mov_' prefix to item_id
        df['item_id'] = 'mov_' + df['movieId'].astype(str)

        # Set media type
        df['media_type'] = 'movie'

        # Use clean_title if available, otherwise use title
        if 'clean_title' in df.columns:
            df['title'] = df['clean_title']

        # Ensure year column exists
        if 'year' not in df.columns:
            df['year'] = 0

        # Map genres to universal genres
        logger.info("Mapping genres to universal taxonomy...")
        df['genres'] = df['genres'].apply(
            lambda x: self.genre_mapper.map_genres(x, 'movie')
        )

        # Extract themes from tags if available
        if tags_df is not None and len(tags_df) > 0:
            logger.info("Extracting themes from tags...")
            df['themes'] = df['movieId'].apply(
                lambda mid: self._extract_themes_for_movie(mid, tags_df)
            )
        else:
            logger.info("No tags available, themes will be empty")
            df['themes'] = [[] for _ in range(len(df))]

        # Compute rating statistics
        logger.info("Computing rating statistics...")
        rating_stats = self._compute_rating_statistics(ratings_df)

        df['avg_rating'] = df['movieId'].map(rating_stats['avg_rating']).fillna(0.0)
        df['rating_count'] = df['movieId'].map(rating_stats['rating_count']).fillna(0).astype(int)

        # Select and order columns for unified schema
        unified_df = df[[
            'item_id',
            'media_type',
            'title',
            'year',
            'genres',
            'themes',
            'avg_rating',
            'rating_count'
        ]].copy()

        logger.info(f"Transformed to unified schema: {len(unified_df):,} items")
        logger.info(f"Items with genres: {(unified_df['genres'].str.len() > 0).sum():,}")
        logger.info(f"Items with themes: {(unified_df['themes'].str.len() > 0).sum():,}")

        return unified_df

    def transform_ratings(
        self,
        ratings_df: pd.DataFrame,
        media_type: str = 'movie'
    ) -> pd.DataFrame:
        """
        Transform ratings to unified rating schema.

        Unified Rating Schema:
            user_id: str
            item_id: str (with media prefix)
            rating: float (0-5 scale)
            timestamp: int

        Args:
            ratings_df: Cleaned ratings DataFrame
            media_type: Type of media ('movie', 'anime', 'manga', 'game')

        Returns:
            DataFrame with unified schema
        """
        logger.info(f"Transforming {len(ratings_df):,} {media_type} ratings to unified schema")

        df = ratings_df.copy()

        # Convert user_id to string
        df['user_id'] = df['userId'].astype(str)

        # Add media prefix to item_id
        prefix = self._get_media_prefix(media_type)
        df['item_id'] = prefix + df['movieId'].astype(str)

        # Ensure rating is float
        df['rating'] = df['rating'].astype(float)

        # Keep timestamp as int
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].astype(int)
        else:
            # If no timestamp, use 0
            df['timestamp'] = 0

        # Select columns for unified schema
        unified_df = df[[
            'user_id',
            'item_id',
            'rating',
            'timestamp'
        ]].copy()

        logger.info(f"Transformed to unified schema: {len(unified_df):,} ratings")

        return unified_df

    def _compute_rating_statistics(self, ratings_df: pd.DataFrame) -> Dict[int, Dict]:
        """
        Compute average rating and rating count for each movie.

        Args:
            ratings_df: Ratings DataFrame

        Returns:
            Dictionary mapping movieId to stats dict
        """
        stats = ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        })

        stats.columns = ['avg_rating', 'rating_count']

        return {
            'avg_rating': stats['avg_rating'].to_dict(),
            'rating_count': stats['rating_count'].to_dict()
        }

    def _extract_themes_for_movie(
        self,
        movie_id: int,
        tags_df: pd.DataFrame
    ) -> List[str]:
        """
        Extract theme keywords for a specific movie from tags.

        Args:
            movie_id: Movie ID
            tags_df: Tags DataFrame

        Returns:
            List of theme keywords
        """
        # Get tags for this movie
        movie_tags = tags_df[tags_df['movieId'] == movie_id]['tag'].tolist()

        if not movie_tags:
            return []

        # Extract themes using genre mapper
        themes = self.genre_mapper.extract_themes(movie_tags, 'movie')

        return themes

    @staticmethod
    def _get_media_prefix(media_type: str) -> str:
        """
        Get item ID prefix for media type.

        Args:
            media_type: Type of media

        Returns:
            Prefix string (e.g., 'mov_', 'ani_')
        """
        prefixes = {
            'movie': 'mov_',
            'anime': 'ani_',
            'manga': 'man_',
            'game': 'gam_'
        }

        return prefixes.get(media_type, 'unk_')

    def validate_unified_schema(
        self,
        items_df: pd.DataFrame,
        ratings_df: pd.DataFrame
    ) -> bool:
        """
        Validate that DataFrames conform to unified schema.

        Args:
            items_df: Unified items DataFrame
            ratings_df: Unified ratings DataFrame

        Returns:
            True if valid

        Raises:
            ValueError if validation fails
        """
        logger.info("Validating unified schema...")

        # Check items schema
        required_item_cols = ['item_id', 'media_type', 'title', 'year', 'genres', 'themes', 'avg_rating', 'rating_count']
        missing_cols = set(required_item_cols) - set(items_df.columns)
        if missing_cols:
            raise ValueError(f"Items DataFrame missing columns: {missing_cols}")

        # Check ratings schema
        required_rating_cols = ['user_id', 'item_id', 'rating', 'timestamp']
        missing_cols = set(required_rating_cols) - set(ratings_df.columns)
        if missing_cols:
            raise ValueError(f"Ratings DataFrame missing columns: {missing_cols}")

        # Validate data types
        if not items_df['item_id'].dtype == 'object':
            raise ValueError("item_id must be string type")

        if not ratings_df['user_id'].dtype == 'object':
            raise ValueError("user_id must be string type")

        if not ratings_df['rating'].dtype in ['float64', 'float32']:
            raise ValueError("rating must be float type")

        # Validate rating range
        if not ratings_df['rating'].between(0, 5).all():
            raise ValueError("Ratings must be in range [0, 5]")

        # Check referential integrity
        rated_items = set(ratings_df['item_id'].unique())
        available_items = set(items_df['item_id'].unique())
        orphan_ratings = rated_items - available_items

        if orphan_ratings:
            logger.warning(f"Found {len(orphan_ratings)} item IDs in ratings not in items table")

        logger.info("Schema validation passed")

        return True

    def get_schema_summary(
        self,
        items_df: pd.DataFrame,
        ratings_df: pd.DataFrame
    ) -> str:
        """
        Generate summary of unified data.

        Args:
            items_df: Unified items DataFrame
            ratings_df: Unified ratings DataFrame

        Returns:
            Formatted summary string
        """
        lines = [
            "\n" + "=" * 60,
            "Unified Schema Summary",
            "=" * 60,
            f"Items: {len(items_df):,}",
            f"  Media types: {items_df['media_type'].value_counts().to_dict()}",
            f"  With genres: {(items_df['genres'].str.len() > 0).sum():,}",
            f"  With themes: {(items_df['themes'].str.len() > 0).sum():,}",
            f"  Avg rating: {items_df['avg_rating'].mean():.2f}",
            "",
            f"Ratings: {len(ratings_df):,}",
            f"  Users: {ratings_df['user_id'].nunique():,}",
            f"  Items: {ratings_df['item_id'].nunique():,}",
            f"  Rating range: {ratings_df['rating'].min():.1f} - {ratings_df['rating'].max():.1f}",
            f"  Mean rating: {ratings_df['rating'].mean():.2f}",
            "=" * 60 + "\n"
        ]

        return "\n".join(lines)


# Convenience function
def transform_movielens_to_unified(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    tags_df: Optional[pd.DataFrame] = None,
    genre_mapper: Optional[GenreMapper] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform MovieLens data to unified schema.

    Args:
        movies_df: Cleaned movies DataFrame
        ratings_df: Cleaned ratings DataFrame
        tags_df: Optional cleaned tags DataFrame
        genre_mapper: Optional GenreMapper instance

    Returns:
        Tuple of (unified_items_df, unified_ratings_df)
    """
    transformer = SchemaTransformer(genre_mapper)

    unified_items = transformer.transform_movies(movies_df, ratings_df, tags_df)
    unified_ratings = transformer.transform_ratings(ratings_df, media_type='movie')

    transformer.validate_unified_schema(unified_items, unified_ratings)

    logger.info(transformer.get_schema_summary(unified_items, unified_ratings))

    return unified_items, unified_ratings
