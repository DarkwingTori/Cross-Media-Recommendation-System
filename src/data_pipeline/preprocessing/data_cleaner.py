"""
Data cleaning utilities for recommendation system.
Handles deduplication, missing values, filtering, and normalization.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans and normalizes data for recommendation system."""

    def __init__(
        self,
        min_item_ratings: int = 10,
        min_user_ratings: int = 10,
        rating_scale: Tuple[float, float] = (0.5, 5.0)
    ):
        """
        Initialize data cleaner.

        Args:
            min_item_ratings: Minimum number of ratings an item must have
            min_user_ratings: Minimum number of ratings a user must have
            rating_scale: (min, max) rating scale
        """
        self.min_item_ratings = min_item_ratings
        self.min_user_ratings = min_user_ratings
        self.rating_scale = rating_scale

        logger.info(f"Initialized DataCleaner:")
        logger.info(f"  min_item_ratings: {min_item_ratings}")
        logger.info(f"  min_user_ratings: {min_user_ratings}")
        logger.info(f"  rating_scale: {rating_scale}")

    def clean_movies(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean movies DataFrame.

        Args:
            movies_df: Raw movies DataFrame

        Returns:
            Cleaned movies DataFrame
        """
        logger.info(f"Cleaning {len(movies_df):,} movies")
        original_count = len(movies_df)

        df = movies_df.copy()

        # Remove duplicates based on movieId
        df = df.drop_duplicates(subset=['movieId'], keep='first')
        if len(df) < original_count:
            logger.info(f"Removed {original_count - len(df)} duplicate movies")

        # Remove movies with missing essential fields
        essential_cols = ['movieId', 'title']
        before = len(df)
        df = df.dropna(subset=essential_cols)
        if len(df) < before:
            logger.info(f"Removed {before - len(df)} movies with missing essential fields")

        # Handle missing genres (set to empty string or 'Unknown')
        if 'genres' in df.columns:
            df['genres'] = df['genres'].fillna('')
            # Remove "(no genres listed)" placeholder
            df['genres'] = df['genres'].replace('(no genres listed)', '')

        # Ensure year is integer where available
        if 'year' in df.columns:
            df['year'] = df['year'].fillna(0).astype(int)

        # Reset index
        df = df.reset_index(drop=True)

        logger.info(f"Cleaned movies: {len(df):,} remaining ({len(df)/original_count*100:.1f}%)")

        return df

    def clean_ratings(
        self,
        ratings_df: pd.DataFrame,
        valid_movie_ids: Optional[set] = None,
        valid_user_ids: Optional[set] = None
    ) -> pd.DataFrame:
        """
        Clean ratings DataFrame.

        Args:
            ratings_df: Raw ratings DataFrame
            valid_movie_ids: Set of valid movie IDs (for referential integrity)
            valid_user_ids: Set of valid user IDs (optional)

        Returns:
            Cleaned ratings DataFrame
        """
        logger.info(f"Cleaning {len(ratings_df):,} ratings")
        original_count = len(ratings_df)

        df = ratings_df.copy()

        # Remove duplicates (keep first occurrence)
        df = df.drop_duplicates(subset=['userId', 'movieId', 'timestamp'], keep='first')
        if len(df) < original_count:
            logger.info(f"Removed {original_count - len(df):,} duplicate ratings")

        # Remove ratings with missing values
        before = len(df)
        df = df.dropna(subset=['userId', 'movieId', 'rating'])
        if len(df) < before:
            logger.info(f"Removed {before - len(df):,} ratings with missing values")

        # Validate rating range
        before = len(df)
        df = df[df['rating'].between(*self.rating_scale)]
        if len(df) < before:
            logger.info(f"Removed {before - len(df):,} ratings outside valid range {self.rating_scale}")

        # Filter by valid movie IDs (referential integrity)
        if valid_movie_ids is not None:
            before = len(df)
            df = df[df['movieId'].isin(valid_movie_ids)]
            if len(df) < before:
                logger.info(f"Removed {before - len(df):,} ratings for non-existent movies")

        # Filter by valid user IDs (if provided)
        if valid_user_ids is not None:
            before = len(df)
            df = df[df['userId'].isin(valid_user_ids)]
            if len(df) < before:
                logger.info(f"Removed {before - len(df):,} ratings for non-existent users")

        # Filter by minimum ratings per user
        if self.min_user_ratings > 0:
            before = len(df)
            user_counts = df['userId'].value_counts()
            valid_users = user_counts[user_counts >= self.min_user_ratings].index
            df = df[df['userId'].isin(valid_users)]
            if len(df) < before:
                removed_users = len(user_counts) - len(valid_users)
                logger.info(f"Removed {removed_users:,} users with < {self.min_user_ratings} ratings")
                logger.info(f"Removed {before - len(df):,} ratings from these users")

        # Filter by minimum ratings per item
        if self.min_item_ratings > 0:
            before = len(df)
            item_counts = df['movieId'].value_counts()
            valid_items = item_counts[item_counts >= self.min_item_ratings].index
            df = df[df['movieId'].isin(valid_items)]
            if len(df) < before:
                removed_items = len(item_counts) - len(valid_items)
                logger.info(f"Removed {removed_items:,} movies with < {self.min_item_ratings} ratings")
                logger.info(f"Removed {before - len(df):,} ratings for these movies")

        # Reset index
        df = df.reset_index(drop=True)

        logger.info(f"Cleaned ratings: {len(df):,} remaining ({len(df)/original_count*100:.1f}%)")
        logger.info(f"Final statistics:")
        logger.info(f"  Unique users: {df['userId'].nunique():,}")
        logger.info(f"  Unique movies: {df['movieId'].nunique():,}")
        logger.info(f"  Rating range: {df['rating'].min():.1f} - {df['rating'].max():.1f}")
        logger.info(f"  Mean rating: {df['rating'].mean():.2f}")

        return df

    def clean_tags(
        self,
        tags_df: pd.DataFrame,
        valid_movie_ids: Optional[set] = None,
        valid_user_ids: Optional[set] = None
    ) -> pd.DataFrame:
        """
        Clean tags DataFrame.

        Args:
            tags_df: Raw tags DataFrame
            valid_movie_ids: Set of valid movie IDs
            valid_user_ids: Set of valid user IDs

        Returns:
            Cleaned tags DataFrame
        """
        logger.info(f"Cleaning {len(tags_df):,} tags")
        original_count = len(tags_df)

        df = tags_df.copy()

        # Remove duplicates
        df = df.drop_duplicates(subset=['userId', 'movieId', 'tag'], keep='first')
        if len(df) < original_count:
            logger.info(f"Removed {original_count - len(df):,} duplicate tags")

        # Remove tags with missing values
        before = len(df)
        df = df.dropna(subset=['userId', 'movieId', 'tag'])
        if len(df) < before:
            logger.info(f"Removed {before - len(df):,} tags with missing values")

        # Clean tag text
        df['tag'] = df['tag'].str.lower().str.strip()

        # Remove empty tags
        before = len(df)
        df = df[df['tag'].str.len() > 0]
        if len(df) < before:
            logger.info(f"Removed {before - len(df):,} empty tags")

        # Filter by valid movie IDs
        if valid_movie_ids is not None:
            before = len(df)
            df = df[df['movieId'].isin(valid_movie_ids)]
            if len(df) < before:
                logger.info(f"Removed {before - len(df):,} tags for non-existent movies")

        # Filter by valid user IDs
        if valid_user_ids is not None:
            before = len(df)
            df = df[df['userId'].isin(valid_user_ids)]
            if len(df) < before:
                logger.info(f"Removed {before - len(df):,} tags for non-existent users")

        # Reset index
        df = df.reset_index(drop=True)

        logger.info(f"Cleaned tags: {len(df):,} remaining ({len(df)/original_count*100:.1f}% if original_count > 0 else 0)")

        return df

    def normalize_ratings(
        self,
        ratings_df: pd.DataFrame,
        target_scale: Tuple[float, float] = (0.0, 5.0)
    ) -> pd.DataFrame:
        """
        Normalize ratings to target scale.

        Args:
            ratings_df: Ratings DataFrame
            target_scale: (min, max) target scale

        Returns:
            DataFrame with normalized ratings
        """
        df = ratings_df.copy()

        current_min = df['rating'].min()
        current_max = df['rating'].max()
        target_min, target_max = target_scale

        logger.info(f"Normalizing ratings from [{current_min}, {current_max}] to [{target_min}, {target_max}]")

        # Linear normalization
        if current_max > current_min:
            df['rating'] = (
                (df['rating'] - current_min) / (current_max - current_min) *
                (target_max - target_min) + target_min
            )
        else:
            logger.warning("All ratings have the same value, no normalization needed")

        logger.info(f"Normalized ratings: mean={df['rating'].mean():.2f}, std={df['rating'].std():.2f}")

        return df

    def filter_cold_start_items(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        min_ratings: int = 10
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter out cold start items (items with too few ratings).

        Args:
            movies_df: Movies DataFrame
            ratings_df: Ratings DataFrame
            min_ratings: Minimum number of ratings required

        Returns:
            Tuple of (filtered_movies_df, filtered_ratings_df)
        """
        logger.info(f"Filtering cold start items (min {min_ratings} ratings)")

        # Count ratings per movie
        rating_counts = ratings_df['movieId'].value_counts()
        valid_movies = rating_counts[rating_counts >= min_ratings].index

        # Filter movies
        filtered_movies = movies_df[movies_df['movieId'].isin(valid_movies)].copy()

        # Filter ratings
        filtered_ratings = ratings_df[ratings_df['movieId'].isin(valid_movies)].copy()

        removed_movies = len(movies_df) - len(filtered_movies)
        removed_ratings = len(ratings_df) - len(filtered_ratings)

        logger.info(f"Removed {removed_movies:,} cold start movies")
        logger.info(f"Removed {removed_ratings:,} ratings for cold start movies")

        return filtered_movies, filtered_ratings

    def get_cleaning_report(
        self,
        original_movies: int,
        original_ratings: int,
        final_movies: int,
        final_ratings: int
    ) -> str:
        """
        Generate a cleaning report.

        Args:
            original_movies: Original number of movies
            original_ratings: Original number of ratings
            final_movies: Final number of movies
            final_ratings: Final number of ratings

        Returns:
            Formatted report string
        """
        report = [
            "\n" + "=" * 60,
            "Data Cleaning Report",
            "=" * 60,
            f"Movies: {original_movies:,} → {final_movies:,} ({final_movies/original_movies*100:.1f}% retained)",
            f"Ratings: {original_ratings:,} → {final_ratings:,} ({final_ratings/original_ratings*100:.1f}% retained)",
            "=" * 60 + "\n"
        ]

        return "\n".join(report)


# Convenience function
def clean_movielens_data(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    tags_df: pd.DataFrame,
    min_item_ratings: int = 10,
    min_user_ratings: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Clean all MovieLens data.

    Args:
        movies_df: Raw movies DataFrame
        ratings_df: Raw ratings DataFrame
        tags_df: Raw tags DataFrame
        min_item_ratings: Minimum ratings per item
        min_user_ratings: Minimum ratings per user

    Returns:
        Tuple of (cleaned_movies, cleaned_ratings, cleaned_tags)
    """
    cleaner = DataCleaner(
        min_item_ratings=min_item_ratings,
        min_user_ratings=min_user_ratings
    )

    # Clean movies first
    movies_clean = cleaner.clean_movies(movies_df)

    # Get valid movie IDs
    valid_movie_ids = set(movies_clean['movieId'].unique())

    # Clean ratings with referential integrity
    ratings_clean = cleaner.clean_ratings(ratings_df, valid_movie_ids=valid_movie_ids)

    # Get valid user IDs from cleaned ratings
    valid_user_ids = set(ratings_clean['userId'].unique())

    # Update valid movie IDs from cleaned ratings
    valid_movie_ids = set(ratings_clean['movieId'].unique())

    # Filter movies to only those with ratings
    movies_clean = movies_clean[movies_clean['movieId'].isin(valid_movie_ids)]

    # Clean tags with both constraints
    tags_clean = cleaner.clean_tags(
        tags_df,
        valid_movie_ids=valid_movie_ids,
        valid_user_ids=valid_user_ids
    )

    return movies_clean, ratings_clean, tags_clean
