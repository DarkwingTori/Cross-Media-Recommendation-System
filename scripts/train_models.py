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
from data_pipeline.transform.embedding_generator import EmbeddingGenerator
from models.cross_domain.domain_bridge import DomainBridge
from sklearn.feature_extraction.text import TfidfVectorizer

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


def train_cross_domain_bridges(
    k_similar: int = 50,
    max_features: int = 500
) -> dict:
    """
    Train cross-domain bridge matrices for all media type pairs.

    Args:
        k_similar: Number of similar items to keep per source item
        max_features: Maximum TF-IDF features

    Returns:
        Dictionary with bridge statistics
    """
    logger.info("\n" + "=" * 80)
    logger.info("Training Cross-Domain Bridges")
    logger.info("=" * 80)

    # Paths
    data_dir = project_root / "data" / "processed"
    mappings_dir = project_root / "data" / "mappings"
    model_dir = project_root / "data" / "models"

    mappings_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load all processed items
    logger.info("\n[Step 1/3] Loading processed items...")
    items = {}
    for media_type in ['movie', 'anime', 'manga']:
        items_path = data_dir / f"{media_type}_items.parquet"
        if items_path.exists():
            items[media_type] = pd.read_parquet(items_path)
            logger.info(f"  {media_type}: {len(items[media_type]):,} items")
        else:
            logger.warning(f"  {media_type}: not found, skipping")

    if len(items) < 2:
        logger.error("Need at least 2 media types to build bridges")
        return {}

    # Generate embeddings for each media type
    logger.info("\n[Step 2/3] Generating TF-IDF embeddings...")

    # CRITICAL: Fit vectorizer on ALL media types combined to create shared vocabulary
    logger.info("Fitting vectorizer on combined corpus...")
    all_items = pd.concat([items_df for items_df in items.values()], ignore_index=True)

    generator = EmbeddingGenerator(max_features=max_features)

    # Fit on all items to create shared vocabulary
    all_items['text'] = all_items.apply(generator._create_text_representation, axis=1)
    generator.vectorizer = generator.vectorizer or TfidfVectorizer(
        max_features=max_features,
        ngram_range=generator.ngram_range,
        min_df=generator.min_df,
        lowercase=True
    )
    generator.vectorizer.fit(all_items['text'])
    logger.info(f"Shared vocabulary: {len(generator.vectorizer.vocabulary_):,} features")

    # Now transform each media type using the shared vocabulary
    embeddings = {}

    for media_type, items_df in items.items():
        logger.info(f"\nGenerating embeddings for {media_type}...")
        emb, item_to_idx, idx_to_item = generator.generate_embeddings(
            items_df,
            media_type,
            fit_new=False  # Use the shared vocabulary we just fitted
        )

        embeddings[media_type] = {
            'embeddings': emb,
            'item_to_idx': item_to_idx,
            'idx_to_item': idx_to_item
        }

        # Save embeddings
        emb_path = mappings_dir / f"{media_type}_embeddings.pkl"
        generator.save_embeddings(emb, item_to_idx, idx_to_item, str(emb_path))

    # Build bridges from movie to other media types
    logger.info("\n[Step 3/3] Building cross-domain bridges...")
    bridges_built = {}

    source_media = 'movie'
    if source_media not in embeddings:
        logger.error("Movie embeddings required as source")
        return {}

    for target_media in ['anime', 'manga']:
        if target_media not in embeddings:
            logger.warning(f"Skipping {source_media}→{target_media} (target not available)")
            continue

        logger.info(f"\nBuilding bridge: {source_media} → {target_media}")

        bridge = DomainBridge(
            source_embeddings=embeddings[source_media]['embeddings'],
            target_embeddings=embeddings[target_media]['embeddings'],
            source_items=items[source_media],
            target_items=items[target_media],
            source_media=source_media,
            target_media=target_media,
            k_similar=k_similar
        )

        bridge.build_bridge_matrix()

        # Save bridge
        bridge_path = model_dir / f"bridge_{source_media}_to_{target_media}.npz"
        bridge.save_bridge(str(bridge_path))

        # Store stats
        bridges_built[f"{source_media}→{target_media}"] = bridge.get_bridge_stats()

        logger.info(f"Bridge {source_media}→{target_media} complete:")
        logger.info(f"  {bridge}")

    logger.info("\n" + "=" * 80)
    logger.info("Cross-Domain Bridges Training Complete!")
    logger.info("=" * 80)
    logger.info(f"\nBridges built: {len(bridges_built)}")
    for key in bridges_built.keys():
        logger.info(f"  ✅ {key}")
    logger.info("=" * 80 + "\n")

    return bridges_built


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Train collaborative filtering and cross-domain models"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=['collaborative', 'cross-domain', 'both'],
        default='collaborative',
        help="Training mode (default: collaborative)"
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

    # Train based on mode
    try:
        if args.mode == 'collaborative':
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
            else:
                print(f"\n⚠️  WARNING: Model below target (Precision@10 = {precision_10:.4f}, target >= 0.30)")

        elif args.mode == 'cross-domain':
            results = train_cross_domain_bridges(
                k_similar=args.k_neighbors,
                max_features=500
            )
            print(f"\n✅ Cross-domain bridges trained successfully!")
            print(f"Bridges built: {len(results)}")

        elif args.mode == 'both':
            # Train collaborative filtering first
            logger.info("Training collaborative filtering...")
            train_movie_cf(
                k_neighbors=args.k_neighbors,
                test_size=args.test_size,
                sample_users=args.sample_users,
                save_model=not args.no_save
            )

            # Then train cross-domain bridges
            logger.info("\nTraining cross-domain bridges...")
            train_cross_domain_bridges(
                k_similar=args.k_neighbors,
                max_features=500
            )

            print("\n✅ All models trained successfully!")

        return 0

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
