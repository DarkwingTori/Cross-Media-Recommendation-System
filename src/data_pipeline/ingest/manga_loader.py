"""
Manga dataset loader.
Handles loading and validation of manga data.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Optional
import ast

logger = logging.getLogger(__name__)


class MangaLoader:
    """Loads manga dataset files with validation."""

    def __init__(self, data_dir: str = None):
        """
        Initialize manga loader.

        Args:
            data_dir: Path to manga data directory.
                     Defaults to data/raw/manga/
        """
        if data_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            self.data_dir = project_root / "data" / "raw" / "manga"
        else:
            self.data_dir = Path(data_dir)

        logger.info(f"Initialized MangaLoader with data directory: {self.data_dir}")

    def load_manga(self) -> pd.DataFrame:
        """
        Load manga.csv file with manga metadata.

        Returns:
            DataFrame with columns: Title, Score, Genres, Themes, etc.
        """
        filepath = self.data_dir / "manga.csv"
        logger.info(f"Loading manga from {filepath}")

        try:
            manga_df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(manga_df):,} manga")

            # Create manga_id if not present (use index)
            if 'manga_id' not in manga_df.columns:
                manga_df['manga_id'] = range(1, len(manga_df) + 1)

            # Normalize column names
            if 'Title' in manga_df.columns:
                manga_df = manga_df.rename(columns={'Title': 'title'})
            if 'Genres' in manga_df.columns:
                manga_df = manga_df.rename(columns={'Genres': 'genres'})

            # Extract year from Published column
            if 'Published' in manga_df.columns:
                manga_df['year'] = manga_df['Published'].apply(self._extract_year)
            else:
                manga_df['year'] = None

            logger.info(f"Extracted years for {manga_df['year'].notna().sum():,} manga")

            return manga_df

        except FileNotFoundError:
            logger.error(f"Manga file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading manga: {e}")
            raise

    def create_synthetic_ratings(
        self,
        manga_df: pd.DataFrame,
        num_users: int = 10000,
        ratings_per_user: int = 20
    ) -> pd.DataFrame:
        """
        Create synthetic ratings based on manga scores.

        Note: This dataset doesn't include user ratings, so we generate synthetic ones
        based on the manga scores and popularity metrics.

        Args:
            manga_df: Manga DataFrame with Score and Members columns
            num_users: Number of synthetic users to create
            ratings_per_user: Average ratings per user

        Returns:
            DataFrame with synthetic ratings
        """
        import numpy as np

        logger.info(f"Creating synthetic ratings for {num_users:,} users...")

        # Filter to manga with scores
        scored_manga = manga_df[manga_df['Score'].notna() & (manga_df['Score'] > 0)].copy()

        if len(scored_manga) == 0:
            logger.warning("No scored manga found")
            return pd.DataFrame(columns=['user_id', 'manga_id', 'rating', 'timestamp'])

        # Create probability distribution based on Members (popularity)
        if 'Members' in scored_manga.columns:
            # Convert Members string to int (e.g., "670,559" -> 670559)
            scored_manga['members_int'] = scored_manga['Members'].str.replace(',', '').astype(float)
            weights = scored_manga['members_int'].fillna(1).values
            weights = weights / weights.sum()
        else:
            weights = None

        ratings_list = []

        for user_id in range(1, num_users + 1):
            # Vary number of ratings per user
            n_ratings = np.random.poisson(ratings_per_user)
            n_ratings = min(n_ratings, len(scored_manga))

            if n_ratings == 0:
                continue

            # Sample manga based on popularity
            sampled_manga = np.random.choice(
                scored_manga['manga_id'].values,
                size=n_ratings,
                replace=False,
                p=weights
            )

            for manga_id in sampled_manga:
                # Get base score
                base_score = scored_manga[scored_manga['manga_id'] == manga_id]['Score'].values[0]

                # Add user variance (±2 points, clipped to 1-10)
                user_rating = base_score + np.random.normal(0, 1.5)
                user_rating = np.clip(user_rating, 1, 10)

                ratings_list.append({
                    'user_id': f'manga_user_{user_id}',
                    'manga_id': int(manga_id),
                    'rating': float(user_rating),
                    'timestamp': 0
                })

        ratings_df = pd.DataFrame(ratings_list)

        logger.info(f"Created {len(ratings_df):,} synthetic ratings")
        logger.info(f"  Users: {ratings_df['user_id'].nunique():,}")
        logger.info(f"  Manga: {ratings_df['manga_id'].nunique():,}")
        logger.info(f"  Mean rating: {ratings_df['rating'].mean():.2f}")

        return ratings_df

    def load_all(self, create_ratings: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load all manga files.

        Args:
            create_ratings: Whether to create synthetic ratings

        Returns:
            Tuple of (manga_df, ratings_df)
        """
        logger.info("Loading all manga files...")

        manga_df = self.load_manga()

        if create_ratings:
            ratings_df = self.create_synthetic_ratings(manga_df)
        else:
            ratings_df = pd.DataFrame(columns=['user_id', 'manga_id', 'rating', 'timestamp'])

        logger.info("All manga files loaded successfully")

        return manga_df, ratings_df

    def validate_data(
        self,
        manga_df: pd.DataFrame,
        ratings_df: pd.DataFrame
    ) -> bool:
        """
        Validate consistency across datasets.

        Args:
            manga_df: Manga DataFrame
            ratings_df: Ratings DataFrame

        Returns:
            True if validation passes
        """
        logger.info("Validating manga dataset consistency...")

        # Check for missing values
        if manga_df['manga_id'].isna().any():
            raise ValueError("Missing manga_id in manga data")

        if len(ratings_df) > 0:
            if ratings_df[['user_id', 'manga_id', 'rating']].isna().any().any():
                raise ValueError("Missing values in critical rating columns")

            # Check rating range
            if not (ratings_df['rating'].between(1, 10).all()):
                logger.warning("Some ratings outside typical range [1, 10]")

        logger.info("Manga data validation completed successfully")
        return True

    @staticmethod
    def _extract_year(published_str: str) -> Optional[int]:
        """
        Extract year from Published string like "Aug  25, 1989 to ?".

        Args:
            published_str: Published date string

        Returns:
            Year as integer, or None if not found
        """
        try:
            if pd.isna(published_str) or not published_str:
                return None

            # Extract first 4-digit year
            import re
            match = re.search(r'(\d{4})', published_str)
            if match:
                return int(match.group(1))

        except Exception:
            pass

        return None


# Convenience function
def load_manga_data(data_dir: str = None, create_ratings: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load all manga data.

    Args:
        data_dir: Path to manga data directory
        create_ratings: Whether to create synthetic ratings

    Returns:
        Tuple of (manga_df, ratings_df)
    """
    loader = MangaLoader(data_dir)
    return loader.load_all(create_ratings=create_ratings)
