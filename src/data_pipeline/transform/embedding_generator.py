"""
Embedding generator for cross-domain recommendations.
Generates TF-IDF embeddings from genres and themes to enable semantic similarity.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import pickle
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate TF-IDF embeddings for items based on genres and themes."""

    def __init__(
        self,
        max_features: int = 500,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2
    ):
        """
        Initialize embedding generator.

        Args:
            max_features: Maximum number of features for TF-IDF
            ngram_range: N-gram range for TF-IDF (1,2) captures bigrams like "Action Adventure"
            min_df: Minimum document frequency
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer = None

        logger.info(f"Initialized EmbeddingGenerator:")
        logger.info(f"  max_features: {max_features}")
        logger.info(f"  ngram_range: {ngram_range}")
        logger.info(f"  min_df: {min_df}")

    def generate_embeddings(
        self,
        items_df: pd.DataFrame,
        media_type: str,
        fit_new: bool = True
    ) -> Tuple[csr_matrix, Dict[str, int], Dict[int, str]]:
        """
        Generate TF-IDF embeddings for items.

        Args:
            items_df: DataFrame with 'genres' and 'themes' columns
            media_type: Type of media ('movie', 'anime', 'manga', 'game')
            fit_new: If True, fit a new vectorizer. If False, use existing.

        Returns:
            Tuple of:
            - embeddings: Sparse matrix (num_items × num_features)
            - item_to_idx: Dict mapping item_id to matrix row index
            - idx_to_item: Dict mapping matrix row index to item_id
        """
        logger.info(f"Generating embeddings for {len(items_df):,} {media_type} items")

        # Create text representation from genres + themes
        items_df = items_df.copy()
        items_df['text'] = items_df.apply(self._create_text_representation, axis=1)

        logger.info("Sample text representations:")
        for i in range(min(3, len(items_df))):
            logger.info(f"  {items_df.iloc[i]['item_id']}: {items_df.iloc[i]['text']}")

        # Create or use vectorizer
        if fit_new or self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                lowercase=True,
                stop_words=None  # Keep all words (genres are important)
            )
            embeddings = self.vectorizer.fit_transform(items_df['text'])
            logger.info(f"Fitted new vectorizer with {len(self.vectorizer.vocabulary_):,} features")
        else:
            embeddings = self.vectorizer.transform(items_df['text'])
            logger.info(f"Used existing vectorizer")

        # Create mappings
        item_to_idx = {item_id: idx for idx, item_id in enumerate(items_df['item_id'])}
        idx_to_item = {idx: item_id for item_id, idx in item_to_idx.items()}

        logger.info(f"Generated embeddings:")
        logger.info(f"  Shape: {embeddings.shape}")
        logger.info(f"  Sparsity: {1.0 - (embeddings.nnz / (embeddings.shape[0] * embeddings.shape[1])):.4%}")
        logger.info(f"  Non-zero entries: {embeddings.nnz:,}")

        return embeddings, item_to_idx, idx_to_item

    def save_embeddings(
        self,
        embeddings: csr_matrix,
        item_to_idx: Dict[str, int],
        idx_to_item: Dict[int, str],
        filepath: str
    ) -> None:
        """
        Save embeddings and mappings to pickle file.

        Args:
            embeddings: Sparse embedding matrix
            item_to_idx: Item ID to index mapping
            idx_to_item: Index to item ID mapping
            filepath: Path to save file
        """
        logger.info(f"Saving embeddings to {filepath}")

        data = {
            'embeddings': embeddings,
            'item_to_idx': item_to_idx,
            'idx_to_item': idx_to_item,
            'vectorizer': self.vectorizer,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        file_size = Path(filepath).stat().st_size / (1024 * 1024)
        logger.info(f"Embeddings saved: {file_size:.2f} MB")

    def load_embeddings(
        self,
        filepath: str
    ) -> Tuple[csr_matrix, Dict[str, int], Dict[int, str]]:
        """
        Load embeddings and mappings from pickle file.

        Args:
            filepath: Path to pickle file

        Returns:
            Tuple of (embeddings, item_to_idx, idx_to_item)
        """
        logger.info(f"Loading embeddings from {filepath}")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        embeddings = data['embeddings']
        item_to_idx = data['item_to_idx']
        idx_to_item = data['idx_to_item']
        self.vectorizer = data['vectorizer']

        logger.info(f"Loaded embeddings: shape {embeddings.shape}, {len(item_to_idx):,} items")

        return embeddings, item_to_idx, idx_to_item

    @staticmethod
    def _create_text_representation(row) -> str:
        """
        Create text representation from genres and themes.

        Args:
            row: DataFrame row with 'genres' and 'themes' columns

        Returns:
            Space-separated text of genres and themes
        """
        text_parts = []

        # Add genres
        if isinstance(row['genres'], (list, np.ndarray)):
            text_parts.extend([str(g).lower() for g in row['genres'] if g])
        elif isinstance(row['genres'], str) and row['genres']:
            # Handle string representation of list
            import ast
            try:
                genres = ast.literal_eval(row['genres'])
                text_parts.extend([str(g).lower() for g in genres if g])
            except:
                # Fallback: treat as single genre
                text_parts.append(row['genres'].lower())

        # Add themes
        if isinstance(row['themes'], (list, np.ndarray)):
            text_parts.extend([str(t).lower() for t in row['themes'] if t])
        elif isinstance(row['themes'], str) and row['themes']:
            import ast
            try:
                themes = ast.literal_eval(row['themes'])
                text_parts.extend([str(t).lower() for t in themes if t])
            except:
                text_parts.append(row['themes'].lower())

        # Join with spaces
        text = ' '.join(text_parts) if text_parts else 'unknown'

        return text

    def get_embedding_stats(self, embeddings: csr_matrix) -> Dict:
        """
        Get statistics about embeddings.

        Args:
            embeddings: Embedding matrix

        Returns:
            Dictionary with statistics
        """
        stats = {
            'shape': embeddings.shape,
            'num_items': embeddings.shape[0],
            'num_features': embeddings.shape[1],
            'nnz': embeddings.nnz,
            'sparsity': 1.0 - (embeddings.nnz / (embeddings.shape[0] * embeddings.shape[1])),
            'density': embeddings.nnz / (embeddings.shape[0] * embeddings.shape[1]),
            'avg_features_per_item': embeddings.nnz / embeddings.shape[0] if embeddings.shape[0] > 0 else 0
        }

        if embeddings.nnz > 0:
            stats['min_value'] = embeddings.data.min()
            stats['max_value'] = embeddings.data.max()
            stats['mean_value'] = embeddings.data.mean()

        return stats


# Convenience function
def generate_and_save_embeddings(
    items_df: pd.DataFrame,
    media_type: str,
    output_filepath: str,
    max_features: int = 500
) -> Tuple[csr_matrix, Dict[str, int], Dict[int, str]]:
    """
    Generate and save embeddings in one step.

    Args:
        items_df: Items DataFrame
        media_type: Media type name
        output_filepath: Path to save pickle file
        max_features: Maximum TF-IDF features

    Returns:
        Tuple of (embeddings, item_to_idx, idx_to_item)
    """
    generator = EmbeddingGenerator(max_features=max_features)
    embeddings, item_to_idx, idx_to_item = generator.generate_embeddings(
        items_df,
        media_type
    )

    generator.save_embeddings(embeddings, item_to_idx, idx_to_item, output_filepath)

    return embeddings, item_to_idx, idx_to_item
