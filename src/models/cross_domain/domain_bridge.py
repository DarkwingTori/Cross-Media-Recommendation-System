"""
Cross-domain bridge for preference transfer across media types.
Builds similarity matrices between different media using genre/theme embeddings.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.sparse_utils import save_sparse_matrix, load_sparse_matrix, keep_top_k_per_row

logger = logging.getLogger(__name__)


class DomainBridge:
    """Build and manage cross-domain similarity bridges."""

    def __init__(
        self,
        source_embeddings: csr_matrix,
        target_embeddings: csr_matrix,
        source_items: pd.DataFrame,
        target_items: pd.DataFrame,
        source_media: str,
        target_media: str,
        k_similar: int = 50
    ):
        """
        Initialize domain bridge.

        Args:
            source_embeddings: Embeddings for source media (num_source_items × num_features)
            target_embeddings: Embeddings for target media (num_target_items × num_features)
            source_items: Source items DataFrame with item_id
            target_items: Target items DataFrame with item_id
            source_media: Source media type ('movie', 'anime', etc.)
            target_media: Target media type
            k_similar: Number of similar items to keep per source item
        """
        self.source_embeddings = source_embeddings
        self.target_embeddings = target_embeddings
        self.source_items = source_items
        self.target_items = target_items
        self.source_media = source_media
        self.target_media = target_media
        self.k_similar = k_similar

        # Create item mappings
        self.source_item_to_idx = {item_id: idx for idx, item_id in enumerate(source_items['item_id'])}
        self.source_idx_to_item = {idx: item_id for item_id, idx in self.source_item_to_idx.items()}
        self.target_item_to_idx = {item_id: idx for idx, item_id in enumerate(target_items['item_id'])}
        self.target_idx_to_item = {idx: item_id for item_id, idx in self.target_item_to_idx.items()}

        self.bridge_matrix = None

        logger.info(f"Initialized DomainBridge: {source_media} → {target_media}")
        logger.info(f"  Source items: {len(source_items):,}")
        logger.info(f"  Target items: {len(target_items):,}")
        logger.info(f"  k_similar: {k_similar}")

    def build_bridge_matrix(self) -> 'DomainBridge':
        """
        Build cross-domain similarity matrix.

        Returns:
            Self for method chaining
        """
        logger.info(f"Building bridge matrix: {self.source_media} → {self.target_media}")

        # Compute cosine similarity
        logger.info("Computing cosine similarity...")
        self.bridge_matrix = cosine_similarity(
            self.source_embeddings,
            self.target_embeddings,
            dense_output=False
        )

        logger.info(f"Initial bridge matrix:")
        logger.info(f"  Shape: {self.bridge_matrix.shape}")
        logger.info(f"  Non-zero entries: {self.bridge_matrix.nnz:,}")

        # Keep only top-k similar items per source item
        if self.k_similar > 0:
            logger.info(f"Keeping top-{self.k_similar} similar items per source item...")
            self.bridge_matrix = keep_top_k_per_row(self.bridge_matrix, self.k_similar)

        logger.info("Bridge matrix built successfully")

        return self

    def get_similar_items(
        self,
        source_item_id: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get most similar target items for a source item.

        Args:
            source_item_id: Source item ID
            top_k: Number of similar items to return

        Returns:
            List of (target_item_id, similarity_score) tuples
        """
        if self.bridge_matrix is None:
            raise ValueError("Bridge matrix not built. Call build_bridge_matrix() first.")

        if source_item_id not in self.source_item_to_idx:
            logger.warning(f"Source item {source_item_id} not found")
            return []

        source_idx = self.source_item_to_idx[source_item_id]

        # Get similarity scores for this source item
        similarity_row = self.bridge_matrix.getrow(source_idx)

        # Get top-k items
        scores = similarity_row.toarray().flatten()
        top_indices = np.argsort(scores)[-top_k:][::-1]
        top_scores = scores[top_indices]

        # Convert to item IDs
        similar_items = [
            (self.target_idx_to_item[idx], float(score))
            for idx, score in zip(top_indices, top_scores)
            if score > 0
        ]

        return similar_items

    def save_bridge(self, filepath: str) -> None:
        """
        Save bridge matrix to NPZ file.

        Args:
            filepath: Path to save file
        """
        if self.bridge_matrix is None:
            raise ValueError("Bridge matrix not built. Call build_bridge_matrix() first.")

        logger.info(f"Saving bridge to {filepath}")

        # Save bridge matrix
        save_sparse_matrix(self.bridge_matrix, filepath)

        # Save mappings
        mappings_path = Path(filepath).with_suffix('.mappings.npz')
        np.savez(
            mappings_path,
            source_item_to_idx=self.source_item_to_idx,
            source_idx_to_item=self.source_idx_to_item,
            target_item_to_idx=self.target_item_to_idx,
            target_idx_to_item=self.target_idx_to_item,
            source_media=self.source_media,
            target_media=self.target_media,
            k_similar=self.k_similar
        )

        logger.info("Bridge saved successfully")

    @classmethod
    def load_bridge(cls, filepath: str) -> 'DomainBridge':
        """
        Load a saved bridge from file.

        Args:
            filepath: Path to bridge file

        Returns:
            DomainBridge instance
        """
        logger.info(f"Loading bridge from {filepath}")

        # Load bridge matrix
        bridge_matrix = load_sparse_matrix(filepath)

        # Load mappings
        mappings_path = Path(filepath).with_suffix('.mappings.npz')
        data = np.load(mappings_path, allow_pickle=True)

        # Create instance (without embeddings since we don't need them after building)
        instance = object.__new__(cls)
        instance.bridge_matrix = bridge_matrix
        instance.source_item_to_idx = data['source_item_to_idx'].item()
        instance.source_idx_to_item = data['source_idx_to_item'].item()
        instance.target_item_to_idx = data['target_item_to_idx'].item()
        instance.target_idx_to_item = data['target_idx_to_item'].item()
        instance.source_media = str(data['source_media'])
        instance.target_media = str(data['target_media'])
        instance.k_similar = int(data['k_similar'])

        # These won't be available after loading
        instance.source_embeddings = None
        instance.target_embeddings = None
        instance.source_items = None
        instance.target_items = None

        logger.info(f"Bridge loaded: {instance.source_media} → {instance.target_media}")
        logger.info(f"  Shape: {bridge_matrix.shape}")
        logger.info(f"  Source items: {len(instance.source_item_to_idx):,}")
        logger.info(f"  Target items: {len(instance.target_item_to_idx):,}")

        return instance

    def get_bridge_stats(self) -> Dict:
        """
        Get statistics about the bridge matrix.

        Returns:
            Dictionary with statistics
        """
        if self.bridge_matrix is None:
            return {'built': False}

        stats = {
            'built': True,
            'source_media': self.source_media,
            'target_media': self.target_media,
            'shape': self.bridge_matrix.shape,
            'num_source_items': self.bridge_matrix.shape[0],
            'num_target_items': self.bridge_matrix.shape[1],
            'nnz': self.bridge_matrix.nnz,
            'sparsity': 1.0 - (self.bridge_matrix.nnz / (self.bridge_matrix.shape[0] * self.bridge_matrix.shape[1])),
            'avg_connections_per_source': self.bridge_matrix.nnz / self.bridge_matrix.shape[0] if self.bridge_matrix.shape[0] > 0 else 0,
            'k_similar': self.k_similar
        }

        if self.bridge_matrix.nnz > 0:
            stats['min_similarity'] = self.bridge_matrix.data.min()
            stats['max_similarity'] = self.bridge_matrix.data.max()
            stats['mean_similarity'] = self.bridge_matrix.data.mean()

        return stats

    def __repr__(self) -> str:
        stats = self.get_bridge_stats()

        if not stats['built']:
            return f"DomainBridge({self.source_media} → {self.target_media}, not built)"

        return (
            f"DomainBridge(\n"
            f"  {self.source_media} → {self.target_media}\n"
            f"  source_items={stats['num_source_items']:,},\n"
            f"  target_items={stats['num_target_items']:,},\n"
            f"  avg_connections={stats['avg_connections_per_source']:.1f}\n"
            f")"
        )


# Convenience function
def build_bridge(
    source_embeddings: csr_matrix,
    target_embeddings: csr_matrix,
    source_items: pd.DataFrame,
    target_items: pd.DataFrame,
    source_media: str,
    target_media: str,
    k_similar: int = 50
) -> DomainBridge:
    """
    Build a cross-domain bridge.

    Args:
        source_embeddings: Source embeddings
        target_embeddings: Target embeddings
        source_items: Source items DataFrame
        target_items: Target items DataFrame
        source_media: Source media type
        target_media: Target media type
        k_similar: Number of similar items to keep

    Returns:
        Built DomainBridge
    """
    bridge = DomainBridge(
        source_embeddings,
        target_embeddings,
        source_items,
        target_items,
        source_media,
        target_media,
        k_similar
    )

    bridge.build_bridge_matrix()

    return bridge
