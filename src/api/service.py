"""
Recommendation service layer.
Orchestrates model loading and recommendation generation for the API.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.collaborative.item_based_cf import ItemBasedCF
from models.content_based.genre_based import GenreBasedRecommender
from models.cross_domain.domain_bridge import DomainBridge
from models.cross_domain.preference_transfer import PreferenceTransfer
from models.hybrid.weighted_hybrid import WeightedHybridRecommender
from models.cross_domain.cold_start_handler import ColdStartHandler
from data_pipeline.transform.embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


class RecommendationService:
    """Core recommendation service managing models and serving requests."""

    def __init__(self):
        """Initialize service (models loaded via initialize())."""
        self.cf_model = None
        self.content_model = None
        self.preference_transfer = None
        self.hybrid_model = None
        self.cold_start_handler = None
        self.items_cache = {}
        self.all_items_df = None
        self.initialized = False

        logger.info("RecommendationService created (awaiting initialization)")

    async def initialize(self):
        """
        Load all models and data.
        Called once at application startup.
        """
        logger.info("=" * 80)
        logger.info("Initializing Recommendation Service")
        logger.info("=" * 80)

        project_root = Path(__file__).parent.parent.parent

        # Step 1: Load processed items
        logger.info("\n[1/5] Loading processed items...")
        data_dir = project_root / "data" / "processed"

        for media_type in ['movie', 'anime', 'manga']:
            items_path = data_dir / f"{media_type}_items.parquet"
            if items_path.exists():
                self.items_cache[media_type] = pd.read_parquet(items_path)
                logger.info(f"  {media_type}: {len(self.items_cache[media_type]):,} items")
            else:
                logger.warning(f"  {media_type}: not found")

        # Combine all items
        self.all_items_df = pd.concat(self.items_cache.values(), ignore_index=True)
        logger.info(f"  Total: {len(self.all_items_df):,} items")

        # Step 2: Load collaborative filtering model
        logger.info("\n[2/5] Loading collaborative filtering model...")
        cf_path = project_root / "data" / "models" / "movie_similarity.npz"

        if cf_path.exists():
            self.cf_model = ItemBasedCF()
            self.cf_model.load_model(str(cf_path))
            logger.info(f"  ✅ CF model loaded: {self.cf_model.get_model_info()['num_items']:,} items")
        else:
            logger.warning("  ❌ CF model not found")

        # Step 3: Load content-based model
        logger.info("\n[3/5] Loading content-based model...")
        emb_path = project_root / "data" / "mappings" / "movie_embeddings.pkl"

        if emb_path.exists() and 'movie' in self.items_cache:
            gen = EmbeddingGenerator()
            embeddings, item_to_idx, idx_to_item = gen.load_embeddings(str(emb_path))
            self.content_model = GenreBasedRecommender(
                self.items_cache['movie'],
                embeddings,
                item_to_idx,
                idx_to_item
            )
            logger.info(f"  ✅ Content model loaded: {len(item_to_idx):,} items")
        else:
            logger.warning("  ❌ Content model embeddings not found")

        # Step 4: Load cross-domain bridges
        logger.info("\n[4/5] Loading cross-domain bridges...")
        bridges = {}
        model_dir = project_root / "data" / "models"

        for source_media in ['movie']:
            for target_media in ['anime', 'manga']:
                bridge_path = model_dir / f"bridge_{source_media}_to_{target_media}.npz"

                if bridge_path.exists():
                    bridge = DomainBridge.load_bridge(str(bridge_path))
                    bridges[f"{source_media}→{target_media}"] = bridge
                    logger.info(f"  ✅ Bridge loaded: {source_media}→{target_media}")
                else:
                    logger.warning(f"  ❌ Bridge not found: {source_media}→{target_media}")

        if bridges:
            self.preference_transfer = PreferenceTransfer(bridges)
        else:
            logger.warning("  No bridges loaded")

        # Step 5: Build hybrid model and cold start handler
        logger.info("\n[5/5] Building hybrid model...")
        self.hybrid_model = WeightedHybridRecommender(
            cf_model=self.cf_model,
            content_model=self.content_model,
            preference_transfer=self.preference_transfer,
            weights={'cf': 0.5, 'content': 0.3, 'cross_domain': 0.2}
        )
        logger.info(f"  {self.hybrid_model}")

        self.cold_start_handler = ColdStartHandler(
            self.all_items_df,
            self.hybrid_model,
            cold_start_threshold=10
        )
        logger.info(f"  {self.cold_start_handler}")

        self.initialized = True

        logger.info("\n" + "=" * 80)
        logger.info("✅ Recommendation Service Initialized Successfully")
        logger.info("=" * 80 + "\n")

    async def get_recommendations(
        self,
        user_ratings: Dict[str, float],
        target_media: str = 'movie',
        top_n: int = 10
    ) -> List[Dict]:
        """
        Get personalized recommendations.

        Args:
            user_ratings: Dict mapping item_id to rating
            target_media: Target media type
            top_n: Number of recommendations

        Returns:
            List of recommendation dicts with item details
        """
        if not self.initialized:
            raise RuntimeError("Service not initialized")

        # Use cold start handler (adaptively chooses strategy)
        recs = self.cold_start_handler.recommend(
            user_ratings,
            target_media=target_media,
            top_n=top_n,
            exclude_rated=True
        )

        # Enrich with item details
        enriched_recs = []
        for item_id, score in recs:
            item_dict = self._get_item_details(item_id, target_media)
            if item_dict:
                item_dict['score'] = float(score)
                enriched_recs.append(item_dict)

        return enriched_recs

    async def get_item_details(self, item_id: str) -> Optional[Dict]:
        """
        Get details for a specific item.

        Args:
            item_id: Item ID

        Returns:
            Item details dict or None if not found
        """
        if not self.initialized:
            raise RuntimeError("Service not initialized")

        # Determine media type from prefix
        if item_id.startswith('mov_'):
            media_type = 'movie'
        elif item_id.startswith('ani_'):
            media_type = 'anime'
        elif item_id.startswith('man_'):
            media_type = 'manga'
        else:
            return None

        return self._get_item_details(item_id, media_type)

    async def search_items(
        self,
        query: str,
        media_type: Optional[str] = None,
        limit: int = 20
    ) -> Dict:
        """
        Search for items by title.

        Args:
            query: Search string
            media_type: Optional media type filter
            limit: Maximum results

        Returns:
            Dict with results and total count
        """
        if not self.initialized:
            raise RuntimeError("Service not initialized")

        query_lower = query.lower()

        # Filter by media type if specified
        if media_type and media_type in self.items_cache:
            search_df = self.items_cache[media_type]
        else:
            search_df = self.all_items_df

        # Search by title (case-insensitive partial match)
        matches = search_df[
            search_df['title'].str.lower().str.contains(query_lower, na=False, regex=False)
        ]

        # Limit results
        results = matches.head(limit)

        # Convert to dict
        results_list = []
        for _, row in results.iterrows():
            item_dict = self._row_to_dict(row)
            results_list.append(item_dict)

        return {
            'results': results_list,
            'total_results': len(matches)
        }

    async def health_check(self) -> Dict:
        """
        Check service health.

        Returns:
            Health status dict
        """
        items_count = {
            media_type: len(items_df)
            for media_type, items_df in self.items_cache.items()
        }

        return {
            'status': 'healthy' if self.initialized else 'initializing',
            'models_loaded': self.initialized,
            'items_count': items_count,
            'version': '1.0.0'
        }

    def _get_item_details(self, item_id: str, media_type: str) -> Optional[Dict]:
        """
        Get item details from cache.

        Args:
            item_id: Item ID
            media_type: Media type

        Returns:
            Item details dict or None
        """
        if media_type not in self.items_cache:
            return None

        items = self.items_cache[media_type]
        item_match = items[items['item_id'] == item_id]

        if len(item_match) == 0:
            return None

        return self._row_to_dict(item_match.iloc[0])

    @staticmethod
    def _row_to_dict(row) -> Dict:
        """Convert DataFrame row to dict for API response."""
        return {
            'item_id': row['item_id'],
            'title': row['title'],
            'media_type': row['media_type'],
            'genres': row['genres'].tolist() if isinstance(row['genres'], np.ndarray) else list(row['genres']),
            'themes': row['themes'].tolist() if isinstance(row['themes'], np.ndarray) else list(row['themes']),
            'avg_rating': float(row['avg_rating']),
            'rating_count': int(row['rating_count']),
            'year': int(row['year']) if pd.notna(row['year']) else None
        }


# Global service instance (singleton)
_service = None


def get_service() -> RecommendationService:
    """Get or create the global service instance."""
    global _service
    if _service is None:
        _service = RecommendationService()
    return _service
