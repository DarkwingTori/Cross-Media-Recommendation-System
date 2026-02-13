#!/usr/bin/env python3
"""
Model training script for Media Recommendation System.
Trains collaborative filtering model and evaluates performance.
"""

import sys
from pathlib import Path
import argparse
import logging
import pandas as pd

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from models.collaborative.item_based_cf import ItemBasedCF
from evaluation.evaluator import ModelEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / 'training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def train_movie_cf(
    k_neighbors: int = 50,
    test_size: float = 0.2,
    sample_users: int = None,
    save_model: bool = True
) -> dict:
    """
    Train and evaluate item-based collaborative filtering model for movies.

    Args:
        k_neighbors: Number of similar items to consider
        test_size: Fraction of data for testing
        sample_users: Number of users to sample for evaluation (None = all)
        save_model: Whether to save the trained model

    Returns:
        Dictionary with evaluation results
    """
    logger.info("\n" + "=" * 80)
    logger.info("Training Item-Based Collaborative Filtering Model")
    logger.info("=" * 80)

    # Paths
    data_dir = project_root / "data" / "processed"
    model_dir = project_root / "data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load processed data
    logger.info("\n[Step 1/4] Loading processed data...")
    ratings_path = data_dir / "movie_ratings.parquet"
    items_path = data_dir / "movie_items.parquet"

    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Processed ratings not found at {ratings_path}\n"
            f"Please run the data pipeline first: python src/data_pipeline/pipeline.py"
        )

    ratings_df = pd.read_parquet(ratings_path)
    items_df = pd.read_parquet(items_path)

    logger.info(f"Loaded:")
    logger.info(f"  Ratings: {len(ratings_df):,}")
    logger.info(f"  Items: {len(items_df):,}")
    logger.info(f"  Users: {ratings_df['user_id'].nunique():,}")
    logger.info(f"  Rating range: {ratings_df['rating'].min():.1f} - {ratings_df['rating'].max():.1f}")

    # Initialize evaluator
    logger.info("\n[Step 2/4] Preparing train/test split...")
    evaluator = ModelEvaluator(k_values=[5, 10, 20])

    # Split data
    train_df, test_df = evaluator.train_test_split(
        ratings_df,
        test_size=test_size,
        temporal_split=True,
        min_train_ratings=5
    )

    # Train model
    logger.info(f"\n[Step 3/4] Training ItemBasedCF model (k_neighbors={k_neighbors})...")
    model = ItemBasedCF(k_neighbors=k_neighbors)
    model.fit(train_df)

    logger.info(f"Model info:")
    logger.info(f"{model}")

    # Evaluate model
    logger.info(f"\n[Step 4/4] Evaluating model...")
    results = evaluator.evaluate_collaborative_filtering(
        model=model,
        test_ratings_df=test_df,
        train_ratings_df=train_df,
        sample_users=sample_users,
        top_n=20
    )

    # Print results
    evaluator.print_results("Item-Based CF Evaluation Results")

    # Save model
    if save_model:
        model_path = model_dir / "movie_similarity.npz"
        logger.info(f"\nSaving model to {model_path}")
        model.save_model(str(model_path))
        logger.info("Model saved successfully")

        # Save evaluation results
        results_path = model_dir / "evaluation_results.csv"
        evaluator.export_results(str(results_path))
        logger.info(f"Evaluation results saved to {results_path}")

    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80 + "\n")

    return results


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Train collaborative filtering model for movie recommendations"
    )

    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=50,
        help="Number of similar items to keep per item (default: 50)"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2)"
    )

    parser.add_argument(
        "--sample-users",
        type=int,
        default=None,
        help="Number of users to sample for evaluation (default: all users)"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the trained model"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: sample 1000 users for fast evaluation"
    )

    args = parser.parse_args()

    # Quick mode
    if args.quick:
        logger.info("Quick mode enabled: sampling 1000 users")
        args.sample_users = 1000

    # Train model
    try:
        results = train_movie_cf(
            k_neighbors=args.k_neighbors,
            test_size=args.test_size,
            sample_users=args.sample_users,
            save_model=not args.no_save
        )

        # Success criteria check
        precision_10 = results.get('precision@10', 0)
        if precision_10 >= 0.30:
            print("\n✅ SUCCESS: Model meets performance criteria (Precision@10 >= 0.30)")
            return 0
        else:
            print(f"\n⚠️  WARNING: Model below target (Precision@10 = {precision_10:.4f}, target >= 0.30)")
            print("Consider adjusting hyperparameters or increasing data quality")
            return 0  # Still return success, just a warning

    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}")
        print(f"\n❌ ERROR: {e}")
        return 1

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print(f"\n❌ ERROR: Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
