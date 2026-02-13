"""
Model evaluation orchestrator.
Handles train/test splitting and comprehensive evaluation of recommendation models.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.metrics.accuracy import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
    aggregate_metrics,
    MetricsCalculator
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates recommendation models using standard metrics."""

    def __init__(self, k_values: List[int] = [5, 10, 20]):
        """
        Initialize evaluator.

        Args:
            k_values: List of K values for @K metrics
        """
        self.k_values = k_values
        self.metrics_calc = MetricsCalculator()
        self.results = {}

        logger.info(f"Initialized ModelEvaluator with K values: {k_values}")

    def train_test_split(
        self,
        ratings_df: pd.DataFrame,
        test_size: float = 0.2,
        temporal_split: bool = True,
        min_train_ratings: int = 5
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split ratings into train and test sets.

        Args:
            ratings_df: DataFrame with user_id, item_id, rating, timestamp
            test_size: Fraction of data to use for testing (0.0 to 1.0)
            temporal_split: If True, use temporal split (most recent ratings as test)
            min_train_ratings: Minimum ratings per user in training set

        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info(f"Splitting {len(ratings_df):,} ratings into train/test")
        logger.info(f"  test_size: {test_size}")
        logger.info(f"  temporal_split: {temporal_split}")

        if temporal_split:
            # Temporal split: most recent ratings per user go to test
            train_rows = []
            test_rows = []

            for user_id, group in ratings_df.groupby('user_id'):
                # Sort by timestamp
                sorted_group = group.sort_values('timestamp')

                # Split point
                n = len(sorted_group)
                n_test = max(1, int(n * test_size))
                n_train = n - n_test

                # Ensure minimum train ratings
                if n_train < min_train_ratings:
                    # If not enough ratings, put all in train
                    train_rows.append(sorted_group)
                else:
                    train_rows.append(sorted_group.iloc[:n_train])
                    test_rows.append(sorted_group.iloc[n_train:])

            train_df = pd.concat(train_rows, ignore_index=True)
            test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame()

        else:
            # Random split
            train_df = ratings_df.sample(frac=1-test_size, random_state=42)
            test_df = ratings_df.drop(train_df.index)

        logger.info(f"Split complete:")
        logger.info(f"  Train: {len(train_df):,} ratings from {train_df['user_id'].nunique():,} users")
        logger.info(f"  Test: {len(test_df):,} ratings from {test_df['user_id'].nunique():,} users")

        return train_df, test_df

    def evaluate_collaborative_filtering(
        self,
        model,
        test_ratings_df: pd.DataFrame,
        train_ratings_df: pd.DataFrame,
        sample_users: Optional[int] = None,
        top_n: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate a collaborative filtering model.

        Args:
            model: Fitted collaborative filtering model (with recommend() method)
            test_ratings_df: Test ratings DataFrame
            train_ratings_df: Train ratings DataFrame (for user history)
            sample_users: If set, evaluate only this many random users
            top_n: Number of recommendations to generate

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating collaborative filtering model")

        # Get unique test users
        test_users = test_ratings_df['user_id'].unique()

        # Sample users if requested
        if sample_users and len(test_users) > sample_users:
            test_users = np.random.choice(test_users, sample_users, replace=False)
            logger.info(f"Evaluating on {sample_users} sampled users")
        else:
            logger.info(f"Evaluating on {len(test_users):,} users")

        # Collect metrics for each user
        all_metrics = []
        skipped_users = 0

        for user_id in tqdm(test_users, desc="Evaluating users"):
            # Get user's train ratings (their rating history)
            user_train = train_ratings_df[train_ratings_df['user_id'] == user_id]

            if len(user_train) == 0:
                # Skip users with no training data
                skipped_users += 1
                continue

            # Convert to dict for model
            user_ratings = dict(zip(user_train['item_id'], user_train['rating']))

            # Get user's test items (ground truth)
            user_test = test_ratings_df[test_ratings_df['user_id'] == user_id]
            true_items = set(user_test['item_id'])

            if len(true_items) == 0:
                continue

            # Generate recommendations
            try:
                recommendations = model.recommend(
                    user_ratings,
                    top_n=max(self.k_values),
                    exclude_rated=True
                )

                # Extract item IDs
                rec_item_ids = [item_id for item_id, score in recommendations]

                # Calculate metrics
                user_metrics = self.metrics_calc.calculate_all(
                    true_items,
                    rec_item_ids,
                    k_values=self.k_values
                )

                all_metrics.append(user_metrics)

            except Exception as e:
                logger.debug(f"Error generating recommendations for user {user_id}: {e}")
                skipped_users += 1
                continue

        if skipped_users > 0:
            logger.warning(f"Skipped {skipped_users} users due to errors or missing data")

        # Aggregate metrics
        if not all_metrics:
            logger.error("No metrics collected!")
            return {}

        aggregated = aggregate_metrics(all_metrics)

        # Store results
        self.results = aggregated

        logger.info(f"Evaluation complete on {len(all_metrics)} users")

        return aggregated

    def calculate_coverage(
        self,
        recommendations_list: List[List[Tuple[str, float]]],
        catalog_items: set
    ) -> Dict[str, float]:
        """
        Calculate catalog coverage metrics.

        Args:
            recommendations_list: List of recommendation lists
            catalog_items: Set of all available items

        Returns:
            Dictionary with coverage metrics
        """
        # Collect all recommended items
        recommended_items = set()
        for recs in recommendations_list:
            for item_id, score in recs:
                recommended_items.add(item_id)

        # Calculate coverage
        coverage = len(recommended_items) / len(catalog_items) if catalog_items else 0.0

        return {
            'catalog_coverage': coverage,
            'num_recommended_items': len(recommended_items),
            'num_catalog_items': len(catalog_items)
        }

    def print_results(self, title: str = "Evaluation Results") -> None:
        """
        Print evaluation results in a formatted table.

        Args:
            title: Title for the results table
        """
        if not self.results:
            print("No results to display. Run evaluation first.")
            return

        print("\n" + "=" * 80)
        print(f"{title:^80}")
        print("=" * 80)

        # Group metrics by type
        for k in self.k_values:
            print(f"\n📊 Metrics @ K={k}:")
            print(f"  Precision@{k:2d}: {self.results.get(f'precision@{k}', 0):.4f}")
            print(f"  Recall@{k:2d}:    {self.results.get(f'recall@{k}', 0):.4f}")
            print(f"  F1@{k:2d}:        {self.results.get(f'f1@{k}', 0):.4f}")
            print(f"  NDCG@{k:2d}:      {self.results.get(f'ndcg@{k}', 0):.4f}")
            print(f"  Hit Rate@{k:2d}:  {self.results.get(f'hit_rate@{k}', 0):.4f}")

        print(f"\n📈 Ranking Metrics:")
        print(f"  MRR: {self.results.get('mrr', 0):.4f}")

        # Check success criteria
        print("\n✅ Success Criteria:")
        precision_10 = self.results.get('precision@10', 0)
        if precision_10 >= 0.30:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        print(f"  Precision@10 > 0.30: {precision_10:.4f} {status}")

        print("=" * 80 + "\n")

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get results as a DataFrame.

        Returns:
            DataFrame with metrics
        """
        if not self.results:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame([self.results]).T
        df.columns = ['Value']
        df.index.name = 'Metric'

        return df

    def export_results(self, filepath: str) -> None:
        """
        Export results to CSV.

        Args:
            filepath: Path to save CSV file
        """
        df = self.get_results_dataframe()
        df.to_csv(filepath)
        logger.info(f"Results exported to {filepath}")


# Convenience function
def evaluate_model(
    model,
    ratings_df: pd.DataFrame,
    test_size: float = 0.2,
    k_values: List[int] = [5, 10, 20],
    sample_users: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate a recommendation model with train/test split.

    Args:
        model: Model to evaluate (must have fit() and recommend() methods)
        ratings_df: DataFrame with ratings
        test_size: Fraction for test set
        k_values: K values for metrics
        sample_users: Number of users to sample for evaluation

    Returns:
        Dictionary with metrics
    """
    evaluator = ModelEvaluator(k_values=k_values)

    # Split data
    train_df, test_df = evaluator.train_test_split(
        ratings_df,
        test_size=test_size,
        temporal_split=True
    )

    # Train model
    logger.info("Training model...")
    model.fit(train_df)

    # Evaluate
    results = evaluator.evaluate_collaborative_filtering(
        model,
        test_df,
        train_df,
        sample_users=sample_users
    )

    # Print results
    evaluator.print_results()

    return results
