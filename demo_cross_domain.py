#!/usr/bin/env python3
"""
Cross-Domain Recommendation Demo
Demonstrates preference transfer from movies to anime/manga.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from models.cross_domain.domain_bridge import DomainBridge
from models.cross_domain.preference_transfer import PreferenceTransfer


def main():
    """Demo cross-domain recommendations."""
    print("\n" + "=" * 80)
    print("Cross-Domain Recommendation System Demo")
    print("=" * 80)

    # Load processed items
    print("\nLoading data...")
    movie_items = pd.read_parquet('data/processed/movie_items.parquet')
    anime_items = pd.read_parquet('data/processed/anime_items.parquet')

    # Load bridges
    print("Loading cross-domain bridges...")
    bridge_movie_anime = DomainBridge.load_bridge('data/models/bridge_movie_to_anime.npz')

    # Create preference transfer
    transfer = PreferenceTransfer({
        'movie→anime': bridge_movie_anime
    })

    print(f"✅ Loaded bridge: movie → anime ({bridge_movie_anime.bridge_matrix.shape})")

    # Demo user who likes Action/Sci-Fi movies
    print("\n" + "=" * 80)
    print("Demo: User who loves Action/Sci-Fi Movies")
    print("=" * 80)

    # Find some action/sci-fi movies
    action_scifi_movies = movie_items[
        movie_items['genres'].apply(lambda g: 'Action' in g or 'Sci-Fi' in g)
    ].head(10)

    print("\n📽️  User's Movie Ratings:")
    user_ratings = {}
    for idx, row in action_scifi_movies.iterrows():
        rating = 5.0  # User loves these movies
        user_ratings[row['item_id']] = rating
        print(f"  ⭐ {rating:.1f} - {row['title']} ({row['genres']})")

    # Transfer to anime recommendations
    print("\n🔄 Transferring preferences to anime...")
    anime_recs = transfer.transfer_ratings(
        user_ratings,
        source_media='movie',
        target_media='anime',
        top_n=10
    )

    print("\n📺 Recommended Anime:")
    for i, (anime_id, score) in enumerate(anime_recs, 1):
        anime_row = anime_items[anime_items['item_id'] == anime_id].iloc[0]
        print(f"  {i}. [{score:.3f}] {anime_row['title']}")
        print(f"      Genres: {anime_row['genres']}")
        print(f"      Avg Rating: {anime_row['avg_rating']:.2f} ({anime_row['rating_count']:,} ratings)")

    print("\n" + "=" * 80)
    print("✅ Cross-domain recommendation system working successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
