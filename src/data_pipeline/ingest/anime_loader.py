"""
MyAnimeList dataset loader.
Handles loading and validation of anime data from MyAnimeList dataset.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Optional
import ast

logger = logging.getLogger(__name__)


class AnimeLoader:
    """Loads MyAnimeList anime dataset files with validation."""

    def __init__(self, data_dir: str = None):
        """
        Initialize anime loader.

        Args:
            data_dir: Path to anime data directory.
                     Defaults to data/raw/anime/
        """
        if data_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            self.data_dir = project_root / "data" / "raw" / "anime"
        else:
            self.data_dir = Path(data_dir)

        logger.info(f"Initialized AnimeLoader with data directory: {self.data_dir}")

    def load_anime(self) -> pd.DataFrame:
        """
        Load AnimeList.csv file with anime metadata.

        Returns:
            DataFrame with columns: anime_id, title, genre, score, type, etc.
        """
        filepath = self.data_dir / "AnimeList.csv"
        logger.info(f"Loading anime from {filepath}")

        try:
            anime_df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(anime_df):,} anime")

            # Rename columns for consistency
            if 'anime_id' not in anime_df.columns and 'mal_id' in anime_df.columns:
                anime_df = anime_df.rename(columns={'mal_id': 'anime_id'})

            # Normalize column names
            if 'genre' in anime_df.columns and 'genres' not in anime_df.columns:
                anime_df = anime_df.rename(columns={'genre': 'genres'})

            # Extract year from aired column if available
            if 'aired' in anime_df.columns:
                anime_df['year'] = anime_df['aired'].apply(self._extract_year)
            elif 'aired_string' in anime_df.columns:
                anime_df['year'] = anime_df['aired_string'].apply(self._extract_year_from_string)
            else:
                anime_df['year'] = None

            logger.info(f"Extracted years for {anime_df['year'].notna().sum():,} anime")

            return anime_df

        except FileNotFoundError:
            logger.error(f"Anime file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading anime: {e}")
            raise

    def load_ratings(self, chunksize: Optional[int] = 1_000_000) -> pd.DataFrame:
        """
        Load UserAnimeList.csv with user ratings.

        Args:
            chunksize: Number of rows to load at a time (4.7 GB file!)

        Returns:
            DataFrame with columns: username, anime_id, my_score, my_status
        """
        filepath = self.data_dir / "UserAnimeList.csv"
        logger.info(f"Loading anime ratings from {filepath}")

        try:
            if chunksize is None:
                ratings_df = pd.read_csv(filepath)
                logger.info(f"Loaded {len(ratings_df):,} ratings")
            else:
                logger.info(f"Loading ratings in chunks of {chunksize:,} rows")
                chunks = []
                for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunksize)):
                    # Filter to only completed/watched anime with scores
                    chunk = chunk[chunk['my_score'] > 0]  # Only rated anime
                    chunks.append(chunk)

                    if (i + 1) % 5 == 0:
                        logger.info(f"Loaded {(i + 1) * chunksize:,} ratings...")

                ratings_df = pd.concat(chunks, ignore_index=True)
                logger.info(f"Loaded {len(ratings_df):,} ratings total (with scores)")

            # Rename columns for consistency
            ratings_df = ratings_df.rename(columns={
                'username': 'user_id',
                'my_score': 'rating'
            })

            # Convert to proper types
            ratings_df['user_id'] = ratings_df['user_id'].astype(str)
            ratings_df['anime_id'] = ratings_df['anime_id'].astype(int)
            ratings_df['rating'] = ratings_df['rating'].astype(float)

            # Use my_last_updated as timestamp
            if 'my_last_updated' in ratings_df.columns:
                ratings_df['timestamp'] = ratings_df['my_last_updated'].fillna(0).astype(int)
            else:
                ratings_df['timestamp'] = 0

            logger.info(f"Rating statistics:")
            logger.info(f"  Users: {ratings_df['user_id'].nunique():,}")
            logger.info(f"  Anime: {ratings_df['anime_id'].nunique():,}")
            logger.info(f"  Rating range: {ratings_df['rating'].min():.1f} - {ratings_df['rating'].max():.1f}")
            logger.info(f"  Mean rating: {ratings_df['rating'].mean():.2f}")

            return ratings_df

        except FileNotFoundError:
            logger.error(f"Ratings file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading ratings: {e}")
            raise

    def load_users(self) -> pd.DataFrame:
        """
        Load UserList.csv with user metadata.

        Returns:
            DataFrame with user information
        """
        filepath = self.data_dir / "UserList.csv"
        logger.info(f"Loading users from {filepath}")

        try:
            users_df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(users_df):,} users")
            return users_df

        except FileNotFoundError:
            logger.warning(f"Users file not found: {filepath} (optional file)")
            return pd.DataFrame(columns=['username'])
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            raise

    def load_all(self, load_ratings_chunksize: Optional[int] = 1_000_000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all anime files.

        Args:
            load_ratings_chunksize: Chunksize for loading ratings

        Returns:
            Tuple of (anime_df, ratings_df, users_df)
        """
        logger.info("Loading all anime files...")

        anime_df = self.load_anime()
        ratings_df = self.load_ratings(chunksize=load_ratings_chunksize)
        users_df = self.load_users()

        logger.info("All anime files loaded successfully")

        return anime_df, ratings_df, users_df

    def validate_data(
        self,
        anime_df: pd.DataFrame,
        ratings_df: pd.DataFrame
    ) -> bool:
        """
        Validate consistency across datasets.

        Args:
            anime_df: Anime DataFrame
            ratings_df: Ratings DataFrame

        Returns:
            True if validation passes
        """
        logger.info("Validating anime dataset consistency...")

        # Check for missing values
        if anime_df['anime_id'].isna().any():
            raise ValueError("Missing anime_id in anime data")

        if ratings_df[['user_id', 'anime_id', 'rating']].isna().any().any():
            raise ValueError("Missing values in critical rating columns")

        # Check rating range (MyAnimeList uses 1-10 scale)
        if not (ratings_df['rating'].between(1, 10).all()):
            logger.warning("Some ratings outside typical range [1, 10]")

        # Check referential integrity
        rated_anime = set(ratings_df['anime_id'].unique())
        available_anime = set(anime_df['anime_id'].unique())
        orphan_ratings = rated_anime - available_anime

        if orphan_ratings:
            logger.warning(f"Found {len(orphan_ratings)} anime in ratings not in anime table")

        # Check for duplicates
        if anime_df['anime_id'].duplicated().any():
            raise ValueError("Duplicate anime_ids found in anime data")

        logger.info("Anime data validation completed successfully")
        return True

    @staticmethod
    def _extract_year(aired_dict_str: str) -> Optional[int]:
        """
        Extract year from aired dictionary string.

        Args:
            aired_dict_str: String representation of dict like "{'from': '2012-01-13', ...}"

        Returns:
            Year as integer, or None if not found
        """
        try:
            if pd.isna(aired_dict_str) or not aired_dict_str:
                return None

            aired_dict = ast.literal_eval(aired_dict_str)
            if isinstance(aired_dict, dict) and 'from' in aired_dict:
                date_str = aired_dict['from']
                if date_str and len(date_str) >= 4:
                    return int(date_str[:4])

        except (ValueError, SyntaxError, KeyError):
            pass

        return None

    @staticmethod
    def _extract_year_from_string(aired_str: str) -> Optional[int]:
        """
        Extract year from aired string like "Jan 13, 2012 to Mar 30, 2012".

        Args:
            aired_str: Aired date string

        Returns:
            Year as integer, or None if not found
        """
        try:
            if pd.isna(aired_str) or not aired_str:
                return None

            # Extract first 4-digit year
            import re
            match = re.search(r'(\d{4})', aired_str)
            if match:
                return int(match.group(1))

        except Exception:
            pass

        return None


# Convenience function
def load_anime_data(data_dir: str = None, chunksize: int = 1_000_000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load all anime data.

    Args:
        data_dir: Path to anime data directory
        chunksize: Chunksize for loading ratings

    Returns:
        Tuple of (anime_df, ratings_df, users_df)
    """
    loader = AnimeLoader(data_dir)
    return loader.load_all(load_ratings_chunksize=chunksize)
