"""
MovieLens dataset loader.
Handles loading and validation of MovieLens 32M dataset files.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Optional
import re

logger = logging.getLogger(__name__)


class MovieLensLoader:
    """Loads MovieLens dataset files with validation and error handling."""

    def __init__(self, data_dir: str = None):
        """
        Initialize MovieLens loader.

        Args:
            data_dir: Path to MovieLens data directory.
                     Defaults to data/raw/movies/ml-32m/
        """
        if data_dir is None:
            # Default to project data directory
            project_root = Path(__file__).parent.parent.parent.parent
            self.data_dir = project_root / "data" / "raw" / "movies" / "ml-32m"
        else:
            self.data_dir = Path(data_dir)

        logger.info(f"Initialized MovieLensLoader with data directory: {self.data_dir}")

    def load_movies(self) -> pd.DataFrame:
        """
        Load movies.csv file.

        Returns:
            DataFrame with columns: movieId, title, genres

        Format:
            movieId,title,genres
            1,"Toy Story (1995)","Adventure|Animation|Children|Comedy|Fantasy"
        """
        filepath = self.data_dir / "movies.csv"
        logger.info(f"Loading movies from {filepath}")

        try:
            movies_df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(movies_df):,} movies")

            # Validate columns
            required_cols = ['movieId', 'title', 'genres']
            missing_cols = set(required_cols) - set(movies_df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Parse year from title
            movies_df['year'] = movies_df['title'].apply(self._extract_year)

            # Clean title (remove year)
            movies_df['clean_title'] = movies_df['title'].apply(self._clean_title)

            logger.info(f"Parsed years from {movies_df['year'].notna().sum():,} titles")

            return movies_df

        except FileNotFoundError:
            logger.error(f"Movies file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading movies: {e}")
            raise

    def load_ratings(self, chunksize: Optional[int] = 1_000_000) -> pd.DataFrame:
        """
        Load ratings.csv file with optional chunking for memory efficiency.

        Args:
            chunksize: Number of rows to load at a time.
                      If None, loads entire file at once.

        Returns:
            DataFrame with columns: userId, movieId, rating, timestamp

        Format:
            userId,movieId,rating,timestamp
            1,296,5.0,1147880044
        """
        filepath = self.data_dir / "ratings.csv"
        logger.info(f"Loading ratings from {filepath}")

        try:
            if chunksize is None:
                # Load entire file
                ratings_df = pd.read_csv(filepath)
                logger.info(f"Loaded {len(ratings_df):,} ratings")
            else:
                # Load in chunks
                logger.info(f"Loading ratings in chunks of {chunksize:,} rows")
                chunks = []
                for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunksize)):
                    chunks.append(chunk)
                    if (i + 1) % 10 == 0:
                        logger.info(f"Loaded {(i + 1) * chunksize:,} ratings...")

                ratings_df = pd.concat(chunks, ignore_index=True)
                logger.info(f"Loaded {len(ratings_df):,} ratings total")

            # Validate columns
            required_cols = ['userId', 'movieId', 'rating', 'timestamp']
            missing_cols = set(required_cols) - set(ratings_df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Validate data types
            ratings_df['userId'] = ratings_df['userId'].astype(int)
            ratings_df['movieId'] = ratings_df['movieId'].astype(int)
            ratings_df['rating'] = ratings_df['rating'].astype(float)
            ratings_df['timestamp'] = ratings_df['timestamp'].astype(int)

            logger.info(f"Rating statistics:")
            logger.info(f"  Users: {ratings_df['userId'].nunique():,}")
            logger.info(f"  Movies: {ratings_df['movieId'].nunique():,}")
            logger.info(f"  Rating range: {ratings_df['rating'].min():.1f} - {ratings_df['rating'].max():.1f}")
            logger.info(f"  Mean rating: {ratings_df['rating'].mean():.2f}")

            return ratings_df

        except FileNotFoundError:
            logger.error(f"Ratings file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading ratings: {e}")
            raise

    def load_tags(self) -> pd.DataFrame:
        """
        Load tags.csv file.

        Returns:
            DataFrame with columns: userId, movieId, tag, timestamp

        Format:
            userId,movieId,tag,timestamp
            3,260,"classic",1474784818
        """
        filepath = self.data_dir / "tags.csv"
        logger.info(f"Loading tags from {filepath}")

        try:
            tags_df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(tags_df):,} tags")

            # Validate columns
            required_cols = ['userId', 'movieId', 'tag', 'timestamp']
            missing_cols = set(required_cols) - set(tags_df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Clean tags (lowercase, strip whitespace)
            tags_df['tag'] = tags_df['tag'].str.lower().str.strip()

            logger.info(f"Tags statistics:")
            logger.info(f"  Unique tags: {tags_df['tag'].nunique():,}")
            logger.info(f"  Tagged movies: {tags_df['movieId'].nunique():,}")

            return tags_df

        except FileNotFoundError:
            logger.error(f"Tags file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading tags: {e}")
            raise

    def load_links(self) -> pd.DataFrame:
        """
        Load links.csv file (IMDB and TMDB IDs).

        Returns:
            DataFrame with columns: movieId, imdbId, tmdbId
        """
        filepath = self.data_dir / "links.csv"
        logger.info(f"Loading links from {filepath}")

        try:
            links_df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(links_df):,} links")
            return links_df

        except FileNotFoundError:
            logger.warning(f"Links file not found: {filepath} (optional file)")
            return pd.DataFrame(columns=['movieId', 'imdbId', 'tmdbId'])
        except Exception as e:
            logger.error(f"Error loading links: {e}")
            raise

    def load_all(self, load_ratings_chunksize: Optional[int] = 1_000_000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all MovieLens files.

        Args:
            load_ratings_chunksize: Chunksize for loading ratings

        Returns:
            Tuple of (movies_df, ratings_df, tags_df, links_df)
        """
        logger.info("Loading all MovieLens files...")

        movies_df = self.load_movies()
        ratings_df = self.load_ratings(chunksize=load_ratings_chunksize)
        tags_df = self.load_tags()
        links_df = self.load_links()

        logger.info("All files loaded successfully")

        return movies_df, ratings_df, tags_df, links_df

    def validate_data(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        tags_df: pd.DataFrame
    ) -> bool:
        """
        Validate consistency across datasets.

        Args:
            movies_df: Movies DataFrame
            ratings_df: Ratings DataFrame
            tags_df: Tags DataFrame

        Returns:
            True if validation passes

        Raises:
            ValueError if validation fails
        """
        logger.info("Validating dataset consistency...")

        # Check for missing values in critical columns
        if movies_df['movieId'].isna().any():
            raise ValueError("Missing movieId in movies data")

        if ratings_df[['userId', 'movieId', 'rating']].isna().any().any():
            raise ValueError("Missing values in critical rating columns")

        # Check rating range
        if not (ratings_df['rating'].between(0, 5).all()):
            raise ValueError("Ratings outside valid range [0, 5]")

        # Check referential integrity: ratings should reference valid movies
        rated_movies = set(ratings_df['movieId'].unique())
        available_movies = set(movies_df['movieId'].unique())
        orphan_ratings = rated_movies - available_movies

        if orphan_ratings:
            logger.warning(f"Found {len(orphan_ratings)} movies in ratings that don't exist in movies table")
            logger.warning(f"This will be handled during data cleaning")

        # Check for duplicates
        if movies_df['movieId'].duplicated().any():
            raise ValueError("Duplicate movieIds found in movies data")

        duplicate_ratings = ratings_df.duplicated(subset=['userId', 'movieId', 'timestamp'])
        if duplicate_ratings.any():
            logger.warning(f"Found {duplicate_ratings.sum()} duplicate ratings (will be removed during cleaning)")

        logger.info("Data validation completed successfully")
        return True

    @staticmethod
    def _extract_year(title: str) -> Optional[int]:
        """
        Extract year from movie title.

        Args:
            title: Movie title (e.g., "Toy Story (1995)")

        Returns:
            Year as integer, or None if not found
        """
        match = re.search(r'\((\d{4})\)', title)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _clean_title(title: str) -> str:
        """
        Remove year from movie title.

        Args:
            title: Movie title (e.g., "Toy Story (1995)")

        Returns:
            Clean title without year (e.g., "Toy Story")
        """
        # Remove year in parentheses at the end
        cleaned = re.sub(r'\s*\(\d{4}\)\s*$', '', title)
        return cleaned.strip()


# Convenience function for quick loading
def load_movielens(data_dir: str = None, chunksize: int = 1_000_000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load all MovieLens data.

    Args:
        data_dir: Path to MovieLens data directory
        chunksize: Chunksize for loading ratings

    Returns:
        Tuple of (movies_df, ratings_df, tags_df, links_df)
    """
    loader = MovieLensLoader(data_dir)
    return loader.load_all(load_ratings_chunksize=chunksize)
