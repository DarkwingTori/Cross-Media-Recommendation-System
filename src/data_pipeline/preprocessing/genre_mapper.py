"""
Genre mapping utilities for cross-domain recommendation.
Maps native genre classifications to a unified taxonomy across all media types.
"""

import json
import logging
from pathlib import Path
from typing import List, Set, Dict
import re

logger = logging.getLogger(__name__)


class GenreMapper:
    """Maps native genres to universal genre taxonomy."""

    def __init__(self, taxonomy_path: str = None):
        """
        Initialize genre mapper.

        Args:
            taxonomy_path: Path to genre_taxonomy.json file.
                          Defaults to data/mappings/genre_taxonomy.json
        """
        if taxonomy_path is None:
            # Default to project mappings directory
            project_root = Path(__file__).parent.parent.parent.parent
            taxonomy_path = project_root / "data" / "mappings" / "genre_taxonomy.json"

        self.taxonomy_path = Path(taxonomy_path)
        logger.info(f"Loading genre taxonomy from {self.taxonomy_path}")

        try:
            with open(self.taxonomy_path, 'r') as f:
                self.taxonomy = json.load(f)

            self.universal_genres = self.taxonomy['universal_genres']
            self.genre_mappings = self.taxonomy['genre_mappings']
            self.theme_keywords = self.taxonomy['theme_keywords']

            logger.info(f"Loaded taxonomy with {len(self.universal_genres)} universal genres")
            logger.info(f"Media types: {list(self.genre_mappings.keys())}")

        except FileNotFoundError:
            logger.error(f"Genre taxonomy file not found: {self.taxonomy_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in taxonomy file: {e}")
            raise

    def map_genres(self, native_genres: str, media_type: str) -> List[str]:
        """
        Map native genres to universal genres.

        Args:
            native_genres: Genre string from source data (e.g., "Action|Adventure|Sci-Fi")
            media_type: Type of media ('movie', 'anime', 'manga', 'game')

        Returns:
            List of universal genre strings
        """
        if not native_genres or pd.isna(native_genres):
            logger.debug(f"No genres provided for {media_type}")
            return []

        # Check media type is supported
        if media_type not in self.genre_mappings:
            logger.warning(f"Unsupported media type: {media_type}. Returning empty list.")
            return []

        # Parse native genres (split on pipe or other delimiters)
        native_genre_list = self._parse_genre_string(native_genres)

        # Map to universal genres
        universal_genres = set()
        mapping = self.genre_mappings[media_type]

        for native_genre in native_genre_list:
            # Clean the genre name
            clean_genre = native_genre.strip()

            if clean_genre in mapping:
                # Add mapped universal genres
                mapped = mapping[clean_genre]
                universal_genres.update(mapped)
            else:
                logger.debug(f"No mapping found for {media_type} genre: {clean_genre}")

        return sorted(list(universal_genres))

    def extract_themes(self, tags: List[str], media_type: str) -> List[str]:
        """
        Extract theme keywords from tags.

        Args:
            tags: List of tag strings
            media_type: Type of media ('movie', 'anime', 'manga', 'game')

        Returns:
            List of matching theme keywords
        """
        if not tags:
            return []

        if media_type not in self.theme_keywords:
            logger.warning(f"No theme keywords defined for {media_type}")
            return []

        # Get theme keywords for this media type
        theme_list = self.theme_keywords[media_type]

        # Find matching themes in tags
        matching_themes = set()
        for tag in tags:
            tag_lower = tag.lower().strip()
            for theme in theme_list:
                if theme in tag_lower or tag_lower in theme:
                    matching_themes.add(theme)

        return sorted(list(matching_themes))

    def get_universal_genres(self) -> List[str]:
        """Get list of all universal genres."""
        return self.universal_genres.copy()

    def get_supported_media_types(self) -> List[str]:
        """Get list of supported media types."""
        return list(self.genre_mappings.keys())

    def get_native_genres(self, media_type: str) -> List[str]:
        """
        Get list of native genres for a media type.

        Args:
            media_type: Type of media

        Returns:
            List of native genre names
        """
        if media_type not in self.genre_mappings:
            return []
        return list(self.genre_mappings[media_type].keys())

    def validate_genre(self, genre: str) -> bool:
        """
        Check if a genre is a valid universal genre.

        Args:
            genre: Genre name to validate

        Returns:
            True if valid universal genre
        """
        return genre in self.universal_genres

    @staticmethod
    def _parse_genre_string(genre_string: str) -> List[str]:
        """
        Parse genre string into list of genres.

        Handles various delimiters: pipe (|), comma (,), semicolon (;)

        Args:
            genre_string: Genre string (e.g., "Action|Adventure|Sci-Fi")

        Returns:
            List of individual genre strings
        """
        # Replace common delimiters with pipe
        normalized = genre_string.replace(',', '|').replace(';', '|')

        # Split and clean
        genres = [g.strip() for g in normalized.split('|') if g.strip()]

        return genres

    def get_genre_statistics(self, media_type: str) -> Dict:
        """
        Get statistics about genre mappings.

        Args:
            media_type: Type of media

        Returns:
            Dictionary with mapping statistics
        """
        if media_type not in self.genre_mappings:
            return {}

        mapping = self.genre_mappings[media_type]
        native_genres = list(mapping.keys())

        # Count how many universal genres each native genre maps to
        mapping_counts = [len(mapping[ng]) for ng in native_genres]

        stats = {
            'media_type': media_type,
            'native_genre_count': len(native_genres),
            'universal_genre_count': len(self.universal_genres),
            'avg_mappings_per_native': sum(mapping_counts) / len(mapping_counts) if mapping_counts else 0,
            'min_mappings': min(mapping_counts) if mapping_counts else 0,
            'max_mappings': max(mapping_counts) if mapping_counts else 0,
        }

        return stats


# Import pandas for isna check
import pandas as pd


# Convenience function
def map_genres_from_df(df: pd.DataFrame, genre_col: str, media_type: str, mapper: GenreMapper = None) -> pd.Series:
    """
    Apply genre mapping to a DataFrame column.

    Args:
        df: DataFrame containing genre data
        genre_col: Name of genre column
        media_type: Type of media
        mapper: GenreMapper instance (creates new one if None)

    Returns:
        Series with mapped universal genres (as lists)
    """
    if mapper is None:
        mapper = GenreMapper()

    return df[genre_col].apply(lambda x: mapper.map_genres(x, media_type))
