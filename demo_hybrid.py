#!/usr/bin/env python3
"""
Hybrid Recommendation System Demo
Demonstrates content-based, hybrid, and cold start capabilities.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from models.collaborative.item_based_cf import ItemBasedCF
from models.content_based.genre_based import GenreBasedRecommender
from models.cross_domain.domain_bridge import DomainBridge
from models.cross_domain.preference_transfer import PreferenceTransfer
from models.hybrid.weighted_hybrid import WeightedHybridRecommender
from models.cross_domain.cold_start_handler import ColdStartHandler
from data_pipeline.transform.embedding_generator import EmbeddingGenerator


def main():
    """Demo hybrid recommendation system."""
    print("\n" + "=" * 80)
    print("Hybrid Recommendation System Demo")
    print("=" * 80)

    # Load data
    print("\n[1/4] Loading data and models...")
    movie_items = pd.read_parquet('data/processed/movie_items.parquet')
    anime_items = pd.read_parquet('data/processed/anime_items.parquet')
    all_items = pd.concat([movie_items, anime_items], ignore_index=True)

    # Load CF model
    print("  Loading collaborative filtering model...")
    cf_model = ItemBasedCF()
    cf_model.load_model('data/models/movie_similarity.npz')

    # Load content-based model
    print("  Loading content-based model...")
    gen = EmbeddingGenerator()
    embeddings, item_to_idx, idx_to_item = gen.load_embeddings('data/mappings/movie_embeddings.pkl')
    content_model = GenreBasedRecommender(movie_items, embeddings, item_to_idx, idx_to_item)

    # Load cross-domain bridges
    print("  Loading cross-domain bridges...")
    bridge_movie_anime = DomainBridge.load_bridge('data/models/bridge_movie_to_anime.npz')
    preference_transfer = PreferenceTransfer({'movie→anime': bridge_movie_anime})

    print("  ✅ All models loaded")

    # Create hybrid model
    print("\n[2/4] Building hybrid recommender...")
    hybrid = WeightedHybridRecommender(
        cf_model=cf_model,
        content_model=content_model,
        preference_transfer=preference_transfer,
        weights={'cf': 0.5, 'content': 0.3, 'cross_domain': 0.2}
    )
    print(f"  {hybrid}")

    # Create cold start handler
    print("\n[3/4] Building cold start handler...")
    cold_start_handler = ColdStartHandler(all_items, hybrid, cold_start_threshold=10)
    print(f"  {cold_start_handler}")

    # Demo scenarios
    print("\n[4/4] Running demo scenarios...")
    print("=" * 80)

    # Scenario 1: Cold Start User (3 ratings)
    print("\n🆕 Scenario 1: Cold Start User (3 ratings)")
    print("-" * 80)

    cold_start_ratings = {
        'mov_1': 5.0,      # Toy Story (Adventure/Comedy/Fantasy)
        'mov_2': 4.5,      # Jumanji (Adventure/Comedy/Fantasy)
        'mov_318': 5.0     # Shawshank Redemption (Drama)
    }

    print("User's ratings:")
    for item_id, rating in cold_start_ratings.items():
        item = movie_items[movie_items['item_id'] == item_id].iloc[0]
        print(f"  ⭐ {rating} - {item['title']} {item['genres']}")

    print("\nRecommendations (genre-based popularity):")
    cold_recs = cold_start_handler.recommend(cold_start_ratings, target_media='movie', top_n=5)

    for i, (item_id, score) in enumerate(cold_recs, 1):
        item = movie_items[movie_items['item_id'] == item_id].iloc[0]
        print(f"  {i}. [{score:.1f}] {item['title']}")
        print(f"      Genres: {item['genres']} | ⭐ {item['avg_rating']:.2f}")

    # Scenario 2: Warmstart User (15 ratings) - Within-Domain
    print("\n\n🔥 Scenario 2: Warmstart User (15 ratings) - Movie Recommendations")
    print("-" * 80)

    # Create warmstart user with diverse tastes
    warmstart_ratings = {
        **cold_start_ratings,
        'mov_260': 5.0,    # Star Wars
        'mov_296': 4.0,    # Pulp Fiction
        'mov_356': 5.0,    # Forrest Gump
        'mov_480': 4.5,    # Jurassic Park
        'mov_527': 5.0,    # Schindler's List
        'mov_589': 4.0,    # Terminator 2
        'mov_593': 5.0,    # Silence of the Lambs
        'mov_780': 4.5,    # Independence Day
        'mov_1196': 5.0,   # Star Wars V
        'mov_1210': 4.0,   # Star Wars VI
        'mov_2571': 5.0,   # Matrix
        'mov_4993': 5.0,   # LOTR
    }

    print(f"User has {len(warmstart_ratings)} ratings (Action/Sci-Fi/Drama mix)")
    print("\nHybrid Recommendations (CF + Content):")
    warm_recs = cold_start_handler.recommend(warmstart_ratings, target_media='movie', top_n=5)

    for i, (item_id, score) in enumerate(warm_recs, 1):
        item = movie_items[movie_items['item_id'] == item_id].iloc[0]
        print(f"  {i}. [{score:.3f}] {item['title']}")
        print(f"      Genres: {item['genres']} | ⭐ {item['avg_rating']:.2f}")

    # Scenario 3: Cross-Domain (Movie → Anime)
    print("\n\n🌐 Scenario 3: Cross-Domain Transfer (Movie → Anime)")
    print("-" * 80)

    print("Same user wants anime recommendations...")
    anime_recs = hybrid.recommend(warmstart_ratings, target_media='anime', top_n=5)

    print("\nAnime Recommendations:")
    for i, (item_id, score) in enumerate(anime_recs, 1):
        item = anime_items[anime_items['item_id'] == item_id].iloc[0]
        print(f"  {i}. [{score:.3f}] {item['title']}")
        print(f"      Genres: {item['genres']} | ⭐ {item['avg_rating']:.2f}")

    print("\n" + "=" * 80)
    print("✅ Hybrid System Demonstration Complete!")
    print("=" * 80)
    print("\nKey Capabilities Demonstrated:")
    print("  ✅ Cold start handling (3 ratings → genre-based popularity)")
    print("  ✅ Warmstart hybrid (15 ratings → CF + content combined)")
    print("  ✅ Cross-domain transfer (movie ratings → anime recommendations)")
    print("  ✅ Adaptive strategy based on user history")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
