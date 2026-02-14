"""
ETL Pipeline for Media Recommendation System.
Orchestrates data loading, cleaning, transformation, and export to Parquet.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data_pipeline.ingest.movielens_loader import MovieLensLoader
from data_pipeline.ingest.anime_loader import AnimeLoader
from data_pipeline.ingest.manga_loader import MangaLoader
from data_pipeline.ingest.games_loader import GamesLoader
from data_pipeline.preprocessing.genre_mapper import GenreMapper
from data_pipeline.preprocessing.data_cleaner import DataCleaner
from data_pipeline.transform.schema_transformer import SchemaTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class MovieLensPipeline:
    """ETL pipeline for MovieLens dataset."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        min_item_ratings: int = 10,
        min_user_ratings: int = 10
    ):
        """
        Initialize pipeline.

        Args:
            data_dir: Path to MovieLens data directory
            output_dir: Path to output directory for processed data
            min_item_ratings: Minimum ratings per item
            min_user_ratings: Minimum ratings per user
        """
        # Set up paths
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            self.output_dir = project_root / "data" / "processed"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.loader = MovieLensLoader(data_dir)
        self.genre_mapper = GenreMapper()
        self.cleaner = DataCleaner(
            min_item_ratings=min_item_ratings,
            min_user_ratings=min_user_ratings
        )
        self.transformer = SchemaTransformer(self.genre_mapper)

        logger.info(f"Initialized MovieLensPipeline")
        logger.info(f"Output directory: {self.output_dir}")

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete ETL pipeline.

        Returns:
            Tuple of (unified_items_df, unified_ratings_df)
        """
        logger.info("\n" + "=" * 80)
        logger.info("Starting MovieLens ETL Pipeline")
        logger.info("=" * 80)

        # Step 1: Load data
        logger.info("\n[Step 1/5] Loading data...")
        movies_df, ratings_df, tags_df, links_df = self._load_data()

        # Step 2: Clean data
        logger.info("\n[Step 2/5] Cleaning data...")
        movies_clean, ratings_clean, tags_clean = self._clean_data(
            movies_df, ratings_df, tags_df
        )

        # Step 3: Transform to unified schema
        logger.info("\n[Step 3/5] Transforming to unified schema...")
        unified_items, unified_ratings = self._transform_data(
            movies_clean, ratings_clean, tags_clean
        )

        # Step 4: Validate
        logger.info("\n[Step 4/5] Validating...")
        self._validate_data(unified_items, unified_ratings)

        # Step 5: Export
        logger.info("\n[Step 5/5] Exporting to Parquet...")
        self._export_data(unified_items, unified_ratings)

        logger.info("\n" + "=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 80 + "\n")

        return unified_items, unified_ratings

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all MovieLens data files."""
        with tqdm(total=4, desc="Loading files") as pbar:
            movies_df = self.loader.load_movies()
            pbar.update(1)

            ratings_df = self.loader.load_ratings(chunksize=1_000_000)
            pbar.update(1)

            tags_df = self.loader.load_tags()
            pbar.update(1)

            links_df = self.loader.load_links()
            pbar.update(1)

        logger.info(f"Loaded:")
        logger.info(f"  Movies: {len(movies_df):,}")
        logger.info(f"  Ratings: {len(ratings_df):,}")
        logger.info(f"  Tags: {len(tags_df):,}")
        logger.info(f"  Links: {len(links_df):,}")

        return movies_df, ratings_df, tags_df, links_df

    def _clean_data(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        tags_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Clean all data."""
        original_movies = len(movies_df)
        original_ratings = len(ratings_df)

        # Clean movies
        movies_clean = self.cleaner.clean_movies(movies_df)

        # Get valid movie IDs
        valid_movie_ids = set(movies_clean['movieId'].unique())

        # Clean ratings
        ratings_clean = self.cleaner.clean_ratings(
            ratings_df,
            valid_movie_ids=valid_movie_ids
        )

        # Get valid user IDs and update valid movie IDs
        valid_user_ids = set(ratings_clean['userId'].unique())
        valid_movie_ids = set(ratings_clean['movieId'].unique())

        # Filter movies to only those with ratings
        movies_clean = movies_clean[movies_clean['movieId'].isin(valid_movie_ids)]

        # Clean tags
        tags_clean = self.cleaner.clean_tags(
            tags_df,
            valid_movie_ids=valid_movie_ids,
            valid_user_ids=valid_user_ids
        )

        # Print cleaning report
        report = self.cleaner.get_cleaning_report(
            original_movies,
            original_ratings,
            len(movies_clean),
            len(ratings_clean)
        )
        logger.info(report)

        return movies_clean, ratings_clean, tags_clean

    def _transform_data(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        tags_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Transform data to unified schema."""
        unified_items = self.transformer.transform_movies(
            movies_df,
            ratings_df,
            tags_df
        )

        unified_ratings = self.transformer.transform_ratings(
            ratings_df,
            media_type='movie'
        )

        return unified_items, unified_ratings

    def _validate_data(
        self,
        unified_items: pd.DataFrame,
        unified_ratings: pd.DataFrame
    ) -> None:
        """Validate unified data."""
        self.transformer.validate_unified_schema(unified_items, unified_ratings)

        # Print summary
        summary = self.transformer.get_schema_summary(unified_items, unified_ratings)
        logger.info(summary)

    def _export_data(
        self,
        unified_items: pd.DataFrame,
        unified_ratings: pd.DataFrame
    ) -> None:
        """Export data to Parquet files."""
        items_path = self.output_dir / "movie_items.parquet"
        ratings_path = self.output_dir / "movie_ratings.parquet"

        logger.info(f"Exporting items to {items_path}")
        unified_items.to_parquet(items_path, index=False, compression='snappy')

        logger.info(f"Exporting ratings to {ratings_path}")
        unified_ratings.to_parquet(ratings_path, index=False, compression='snappy')

        # Verify files were created
        items_size = items_path.stat().st_size / (1024 * 1024)  # MB
        ratings_size = ratings_path.stat().st_size / (1024 * 1024)  # MB

        logger.info(f"Export complete:")
        logger.info(f"  movie_items.parquet: {items_size:.2f} MB")
        logger.info(f"  movie_ratings.parquet: {ratings_size:.2f} MB")


def run_movie_pipeline(
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    min_item_ratings: int = 10,
    min_user_ratings: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the MovieLens ETL pipeline.

    Args:
        data_dir: Path to MovieLens data directory
        output_dir: Path to output directory
        min_item_ratings: Minimum ratings per item
        min_user_ratings: Minimum ratings per user

    Returns:
        Tuple of (unified_items_df, unified_ratings_df)
    """
    pipeline = MovieLensPipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        min_item_ratings=min_item_ratings,
        min_user_ratings=min_user_ratings
    )

    return pipeline.run()


def run_anime_pipeline(
    output_dir: Optional[str] = None,
    min_item_ratings: int = 10,
    min_user_ratings: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run anime ETL pipeline."""
    logger.info("\n" + "=" * 80)
    logger.info("Starting Anime ETL Pipeline")
    logger.info("=" * 80)

    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "data" / "processed"

    # Load anime data
    logger.info("\n[Step 1/5] Loading anime data...")
    anime_loader = AnimeLoader()
    anime_df, ratings_df, users_df = anime_loader.load_all(load_ratings_chunksize=1_000_000)

    # Clean data
    logger.info("\n[Step 2/5] Cleaning anime data...")
    cleaner = DataCleaner(min_item_ratings=min_item_ratings, min_user_ratings=min_user_ratings, rating_scale=(1.0, 10.0))

    anime_clean = cleaner.clean_movies(anime_df.rename(columns={'anime_id': 'movieId', 'title': 'title'}))
    anime_clean = anime_clean.rename(columns={'movieId': 'anime_id'})

    valid_anime_ids = set(anime_clean['anime_id'].unique())
    ratings_clean = cleaner.clean_ratings(
        ratings_df.rename(columns={'anime_id': 'movieId', 'user_id': 'userId'}),
        valid_movie_ids=valid_anime_ids
    )
    ratings_clean = ratings_clean.rename(columns={'movieId': 'anime_id', 'userId': 'user_id'})

    # Transform to unified schema
    logger.info("\n[Step 3/5] Transforming to unified schema...")
    transformer = SchemaTransformer()

    # Transform anime items (rename for transformer compatibility)
    anime_for_transform = anime_clean.rename(columns={'anime_id': 'movieId'})
    ratings_for_transform = ratings_clean.rename(columns={'anime_id': 'movieId', 'user_id': 'userId'})

    unified_items = transformer.transform_movies(anime_for_transform, ratings_for_transform, None)
    unified_items['item_id'] = 'ani_' + anime_clean['anime_id'].astype(str)
    unified_items['media_type'] = 'anime'

    unified_ratings = transformer.transform_ratings(ratings_for_transform, media_type='anime')
    unified_ratings['item_id'] = 'ani_' + ratings_clean['anime_id'].astype(str)

    # Export
    logger.info("\n[Step 4/5] Exporting to Parquet...")
    items_path = Path(output_dir) / "anime_items.parquet"
    ratings_path = Path(output_dir) / "anime_ratings.parquet"

    unified_items.to_parquet(items_path, index=False, compression='snappy')
    unified_ratings.to_parquet(ratings_path, index=False, compression='snappy')

    logger.info(f"Export complete:")
    logger.info(f"  anime_items.parquet: {items_path.stat().st_size / (1024 * 1024):.2f} MB")
    logger.info(f"  anime_ratings.parquet: {ratings_path.stat().st_size / (1024 * 1024):.2f} MB")
    logger.info("=" * 80 + "\n")

    return unified_items, unified_ratings


def run_manga_pipeline(
    output_dir: Optional[str] = None,
    min_item_ratings: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run manga ETL pipeline."""
    logger.info("\n" + "=" * 80)
    logger.info("Starting Manga ETL Pipeline")
    logger.info("=" * 80)

    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "data" / "processed"

    # Load manga data
    logger.info("\n[Step 1/4] Loading manga data...")
    manga_loader = MangaLoader()
    manga_df, ratings_df = manga_loader.load_all(create_ratings=True)

    # Clean data
    logger.info("\n[Step 2/4] Cleaning manga data...")
    cleaner = DataCleaner(min_item_ratings=min_item_ratings, min_user_ratings=5, rating_scale=(1.0, 10.0))

    manga_clean = cleaner.clean_movies(manga_df.rename(columns={'manga_id': 'movieId'}))
    manga_clean = manga_clean.rename(columns={'movieId': 'manga_id'})

    if len(ratings_df) > 0:
        valid_manga_ids = set(manga_clean['manga_id'].unique())
        ratings_clean = cleaner.clean_ratings(
            ratings_df.rename(columns={'manga_id': 'movieId', 'user_id': 'userId'}),
            valid_movie_ids=valid_manga_ids
        )
        ratings_clean = ratings_clean.rename(columns={'movieId': 'manga_id', 'userId': 'user_id'})
    else:
        ratings_clean = ratings_df

    # Transform to unified schema
    logger.info("\n[Step 3/4] Transforming to unified schema...")
    transformer = SchemaTransformer()

    manga_for_transform = manga_clean.rename(columns={'manga_id': 'movieId'})
    ratings_for_transform = ratings_clean.rename(columns={'manga_id': 'movieId', 'user_id': 'userId'})

    unified_items = transformer.transform_movies(manga_for_transform, ratings_for_transform, None)
    unified_items['item_id'] = 'man_' + manga_clean['manga_id'].astype(str)
    unified_items['media_type'] = 'manga'

    unified_ratings = transformer.transform_ratings(ratings_for_transform, media_type='manga')
    unified_ratings['item_id'] = 'man_' + ratings_clean['manga_id'].astype(str)

    # Export
    logger.info("\n[Step 4/4] Exporting to Parquet...")
    items_path = Path(output_dir) / "manga_items.parquet"
    ratings_path = Path(output_dir) / "manga_ratings.parquet"

    unified_items.to_parquet(items_path, index=False, compression='snappy')
    unified_ratings.to_parquet(ratings_path, index=False, compression='snappy')

    logger.info(f"Export complete:")
    logger.info(f"  manga_items.parquet: {items_path.stat().st_size / (1024 * 1024):.2f} MB")
    logger.info(f"  manga_ratings.parquet: {ratings_path.stat().st_size / (1024 * 1024):.2f} MB")
    logger.info("=" * 80 + "\n")

    return unified_items, unified_ratings


def run_game_pipeline(
    output_dir: Optional[str] = None,
    min_item_ratings: int = 10,
    min_user_ratings: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run games ETL pipeline."""
    logger.info("\n" + "=" * 80)
    logger.info("Starting Games ETL Pipeline")
    logger.info("=" * 80)

    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "data" / "processed"

    # Load game data
    logger.info("\n[Step 1/5] Loading game data...")
    games_loader = GamesLoader()
    games_df, ratings_df, users_df = games_loader.load_all(load_ratings_chunksize=1_000_000)

    # Clean data
    logger.info("\n[Step 2/5] Cleaning game data...")
    cleaner = DataCleaner(min_item_ratings=min_item_ratings, min_user_ratings=min_user_ratings, rating_scale=(1.0, 10.0))

    games_clean = cleaner.clean_movies(games_df.rename(columns={'game_id': 'movieId'}))
    games_clean = games_clean.rename(columns={'movieId': 'game_id'})

    valid_game_ids = set(games_clean['game_id'].unique())
    ratings_clean = cleaner.clean_ratings(
        ratings_df.rename(columns={'game_id': 'movieId', 'user_id': 'userId'}),
        valid_movie_ids=valid_game_ids
    )
    ratings_clean = ratings_clean.rename(columns={'movieId': 'game_id', 'userId': 'user_id'})

    # Transform to unified schema
    logger.info("\n[Step 3/5] Transforming to unified schema...")
    transformer = SchemaTransformer()

    games_for_transform = games_clean.rename(columns={'game_id': 'movieId'})
    ratings_for_transform = ratings_clean.rename(columns={'game_id': 'movieId', 'user_id': 'userId'})

    unified_items = transformer.transform_movies(games_for_transform, ratings_for_transform, None)
    unified_items['item_id'] = 'gam_' + games_clean['game_id'].astype(str)
    unified_items['media_type'] = 'game'

    unified_ratings = transformer.transform_ratings(ratings_for_transform, media_type='game')
    unified_ratings['item_id'] = 'gam_' + ratings_clean['game_id'].astype(str)

    # Export
    logger.info("\n[Step 4/5] Exporting to Parquet...")
    items_path = Path(output_dir) / "game_items.parquet"
    ratings_path = Path(output_dir) / "game_ratings.parquet"

    unified_items.to_parquet(items_path, index=False, compression='snappy')
    unified_ratings.to_parquet(ratings_path, index=False, compression='snappy')

    logger.info(f"Export complete:")
    logger.info(f"  game_items.parquet: {items_path.stat().st_size / (1024 * 1024):.2f} MB")
    logger.info(f"  game_ratings.parquet: {ratings_path.stat().st_size / (1024 * 1024):.2f} MB")
    logger.info("=" * 80 + "\n")

    return unified_items, unified_ratings


def run_all_pipelines(
    output_dir: Optional[str] = None,
    min_item_ratings: int = 10,
    min_user_ratings: int = 10
) -> dict:
    """
    Run all media type pipelines.

    Returns:
        Dictionary with results for each media type
    """
    logger.info("\n" + "=" * 80)
    logger.info("Running ALL Media Type Pipelines")
    logger.info("=" * 80)

    results = {}

    # Run each pipeline
    try:
        logger.info("\n🎬 Processing MOVIES...")
        results['movies'] = run_movie_pipeline(output_dir=output_dir, min_item_ratings=min_item_ratings, min_user_ratings=min_user_ratings)
    except Exception as e:
        logger.error(f"Movie pipeline failed: {e}")
        results['movies'] = None

    try:
        logger.info("\n📺 Processing ANIME...")
        results['anime'] = run_anime_pipeline(output_dir=output_dir, min_item_ratings=min_item_ratings, min_user_ratings=min_user_ratings)
    except Exception as e:
        logger.error(f"Anime pipeline failed: {e}")
        results['anime'] = None

    try:
        logger.info("\n📚 Processing MANGA...")
        results['manga'] = run_manga_pipeline(output_dir=output_dir, min_item_ratings=min_item_ratings)
    except Exception as e:
        logger.error(f"Manga pipeline failed: {e}")
        results['manga'] = None

    try:
        logger.info("\n🎮 Processing GAMES...")
        results['games'] = run_game_pipeline(output_dir=output_dir, min_item_ratings=min_item_ratings, min_user_ratings=min_user_ratings)
    except Exception as e:
        logger.error(f"Games pipeline failed: {e}")
        results['games'] = None

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("All Pipelines Complete - Summary")
    logger.info("=" * 80)

    for media_type, result in results.items():
        if result:
            items, ratings = result
            logger.info(f"{media_type.upper()}: {len(items):,} items, {len(ratings):,} ratings ✅")
        else:
            logger.info(f"{media_type.upper()}: FAILED ❌")

    logger.info("=" * 80 + "\n")

    return results


if __name__ == "__main__":
    """Run pipeline when executed as script."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Media Recommendation System ETL Pipeline")
    parser.add_argument(
        "--media",
        type=str,
        choices=['movie', 'anime', 'manga', 'game', 'all'],
        default='all',
        help="Media type to process (default: all)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory (only for movies)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to output directory"
    )
    parser.add_argument(
        "--min-item-ratings",
        type=int,
        default=10,
        help="Minimum number of ratings per item"
    )
    parser.add_argument(
        "--min-user-ratings",
        type=int,
        default=10,
        help="Minimum number of ratings per user"
    )

    args = parser.parse_args()

    # Run pipeline based on media type
    try:
        if args.media == 'all':
            results = run_all_pipelines(
                output_dir=args.output_dir,
                min_item_ratings=args.min_item_ratings,
                min_user_ratings=args.min_user_ratings
            )
            print("\n✅ All pipelines completed!")

        elif args.media == 'movie':
            items_df, ratings_df = run_movie_pipeline(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                min_item_ratings=args.min_item_ratings,
                min_user_ratings=args.min_user_ratings
            )
            print(f"\n✅ Movie pipeline completed! Items: {len(items_df):,}, Ratings: {len(ratings_df):,}")

        elif args.media == 'anime':
            items_df, ratings_df = run_anime_pipeline(
                output_dir=args.output_dir,
                min_item_ratings=args.min_item_ratings,
                min_user_ratings=args.min_user_ratings
            )
            print(f"\n✅ Anime pipeline completed! Items: {len(items_df):,}, Ratings: {len(ratings_df):,}")

        elif args.media == 'manga':
            items_df, ratings_df = run_manga_pipeline(
                output_dir=args.output_dir,
                min_item_ratings=args.min_item_ratings
            )
            print(f"\n✅ Manga pipeline completed! Items: {len(items_df):,}, Ratings: {len(ratings_df):,}")

        elif args.media == 'game':
            items_df, ratings_df = run_game_pipeline(
                output_dir=args.output_dir,
                min_item_ratings=args.min_item_ratings,
                min_user_ratings=args.min_user_ratings
            )
            print(f"\n✅ Game pipeline completed! Items: {len(items_df):,}, Ratings: {len(ratings_df):,}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)
