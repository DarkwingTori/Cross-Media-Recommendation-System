"""
Steam games dataset loader.
Handles loading and validation of Steam games data.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Optional
import json
import ast

logger = logging.getLogger(__name__)


class GamesLoader:
    """Loads Steam games dataset files with validation."""

    def __init__(self, data_dir: str = None):
        """
        Initialize games loader.

        Args:
            data_dir: Path to games data directory.
                     Defaults to data/raw/games/
        """
        if data_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            self.data_dir = project_root / "data" / "raw" / "games"
        else:
            self.data_dir = Path(data_dir)

        logger.info(f"Initialized GamesLoader with data directory: {self.data_dir}")

    def load_games(self) -> pd.DataFrame:
        """
        Load games.csv from steam-games directory.

        Returns:
            DataFrame with columns: AppID, Name, Genres, Tags, Price, etc.
        """
        filepath = self.data_dir / "steam-games" / "games.csv"
        logger.info(f"Loading games from {filepath}")

        try:
            games_df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(games_df):,} games")

            # Rename AppID to game_id for consistency
            if 'AppID' in games_df.columns:
                games_df = games_df.rename(columns={'AppID': 'game_id'})

            # Rename Name to title for consistency
            if 'Name' in games_df.columns:
                games_df = games_df.rename(columns={'Name': 'title'})

            # Extract year from Release date
            if 'Release date' in games_df.columns:
                games_df['year'] = games_df['Release date'].apply(self._extract_year)
            else:
                games_df['year'] = None

            logger.info(f"Extracted years for {games_df['year'].notna().sum():,} games")

            return games_df

        except FileNotFoundError:
            logger.error(f"Games file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading games: {e}")
            raise

    def load_ratings(self, chunksize: Optional[int] = 1_000_000) -> pd.DataFrame:
        """
        Load recommendations.csv from steam-recommendations directory.

        Args:
            chunksize: Number of rows to load at a time (1.9 GB file!)

        Returns:
            DataFrame with columns: user_id, app_id, is_recommended, hours
        """
        filepath = self.data_dir / "steam-recommendations" / "recommendations.csv"
        logger.info(f"Loading game recommendations from {filepath}")

        try:
            if chunksize is None:
                ratings_df = pd.read_csv(filepath)
                logger.info(f"Loaded {len(ratings_df):,} recommendations")
            else:
                logger.info(f"Loading recommendations in chunks of {chunksize:,} rows")
                chunks = []
                for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunksize)):
                    chunks.append(chunk)

                    if (i + 1) % 5 == 0:
                        logger.info(f"Loaded {(i + 1) * chunksize:,} recommendations...")

                ratings_df = pd.concat(chunks, ignore_index=True)
                logger.info(f"Loaded {len(ratings_df):,} recommendations total")

            # Rename columns for consistency
            ratings_df = ratings_df.rename(columns={
                'app_id': 'game_id'
            })

            # Convert is_recommended (boolean) to rating (0-10 scale)
            # Also incorporate hours played
            ratings_df['rating'] = ratings_df.apply(self._convert_to_rating, axis=1)

            # Convert to proper types
            ratings_df['user_id'] = ratings_df['user_id'].astype(str)
            ratings_df['game_id'] = ratings_df['game_id'].astype(int)

            # Use date as timestamp if available
            if 'date' in ratings_df.columns:
                ratings_df['timestamp'] = pd.to_datetime(ratings_df['date'], errors='coerce').astype(int) // 10**9
                ratings_df['timestamp'] = ratings_df['timestamp'].fillna(0).astype(int)
            else:
                ratings_df['timestamp'] = 0

            logger.info(f"Rating statistics:")
            logger.info(f"  Users: {ratings_df['user_id'].nunique():,}")
            logger.info(f"  Games: {ratings_df['game_id'].nunique():,}")
            logger.info(f"  Rating range: {ratings_df['rating'].min():.1f} - {ratings_df['rating'].max():.1f}")
            logger.info(f"  Mean rating: {ratings_df['rating'].mean():.2f}")

            return ratings_df

        except FileNotFoundError:
            logger.error(f"Recommendations file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading recommendations: {e}")
            raise

    def load_users(self) -> pd.DataFrame:
        """
        Load users.csv from steam-recommendations directory.

        Returns:
            DataFrame with user information
        """
        filepath = self.data_dir / "steam-recommendations" / "users.csv"
        logger.info(f"Loading users from {filepath}")

        try:
            users_df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(users_df):,} users")
            return users_df

        except FileNotFoundError:
            logger.warning(f"Users file not found: {filepath} (optional file)")
            return pd.DataFrame(columns=['user_id'])
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            raise

    def load_all(self, load_ratings_chunksize: Optional[int] = 1_000_000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all game files.

        Args:
            load_ratings_chunksize: Chunksize for loading recommendations

        Returns:
            Tuple of (games_df, ratings_df, users_df)
        """
        logger.info("Loading all game files...")

        games_df = self.load_games()
        ratings_df = self.load_ratings(chunksize=load_ratings_chunksize)
        users_df = self.load_users()

        logger.info("All game files loaded successfully")

        return games_df, ratings_df, users_df

    def validate_data(
        self,
        games_df: pd.DataFrame,
        ratings_df: pd.DataFrame
    ) -> bool:
        """
        Validate consistency across datasets.

        Args:
            games_df: Games DataFrame
            ratings_df: Ratings DataFrame

        Returns:
            True if validation passes
        """
        logger.info("Validating games dataset consistency...")

        # Check for missing values
        if games_df['game_id'].isna().any():
            raise ValueError("Missing game_id in games data")

        if ratings_df[['user_id', 'game_id', 'rating']].isna().any().any():
            raise ValueError("Missing values in critical rating columns")

        # Check rating range
        if not (ratings_df['rating'].between(1, 10).all()):
            logger.warning("Some ratings outside typical range [1, 10]")

        # Check referential integrity
        rated_games = set(ratings_df['game_id'].unique())
        available_games = set(games_df['game_id'].unique())
        orphan_ratings = rated_games - available_games

        if orphan_ratings:
            logger.warning(f"Found {len(orphan_ratings)} games in ratings not in games table")

        # Check for duplicates
        if games_df['game_id'].duplicated().any():
            logger.warning("Duplicate game_ids found in games data")

        logger.info("Games data validation completed successfully")
        return True

    @staticmethod
    def _convert_to_rating(row) -> float:
        """
        Convert Steam recommendation to rating scale (1-10).

        Uses is_recommended (thumbs up/down) and hours played.

        Args:
            row: DataFrame row with is_recommended and hours columns

        Returns:
            Rating from 1 to 10
        """
        is_recommended = row.get('is_recommended', False)
        hours = row.get('hours', 0)

        if pd.isna(hours):
            hours = 0

        # Base rating on recommendation
        if is_recommended:
            # Positive review: 6-10 based on hours
            if hours < 5:
                return 6.0  # Liked it, but not much playtime
            elif hours < 20:
                return 7.0
            elif hours < 50:
                return 8.0
            elif hours < 100:
                return 9.0
            else:
                return 10.0  # Loved it, lots of playtime
        else:
            # Negative review: 1-5 based on hours
            if hours < 2:
                return 1.0  # Didn't like it, quit early
            elif hours < 5:
                return 2.0
            elif hours < 10:
                return 3.0
            elif hours < 20:
                return 4.0
            else:
                return 5.0  # Mixed feelings, but played a bit

    @staticmethod
    def _extract_year(date_str: str) -> Optional[int]:
        """
        Extract year from release date string.

        Args:
            date_str: Date string in various formats

        Returns:
            Year as integer, or None if not found
        """
        try:
            if pd.isna(date_str) or not date_str:
                return None

            # Extract first 4-digit year
            import re
            match = re.search(r'(\d{4})', date_str)
            if match:
                return int(match.group(1))

        except Exception:
            pass

        return None


# Convenience function
def load_games_data(data_dir: str = None, chunksize: int = 1_000_000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load all games data.

    Args:
        data_dir: Path to games data directory
        chunksize: Chunksize for loading recommendations

    Returns:
        Tuple of (games_df, ratings_df, users_df)
    """
    loader = GamesLoader(data_dir)
    return loader.load_all(load_ratings_chunksize=chunksize)
