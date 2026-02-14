#!/usr/bin/env python3
"""
Dataset downloader for Media Recommendation System.
Downloads anime, manga, and game datasets from Kaggle.
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_kaggle_credentials():
    """Check if Kaggle credentials are set up."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    kaggle_token = os.environ.get('KAGGLE_API_TOKEN')

    if not kaggle_json.exists() and not kaggle_token:
        logger.error("Kaggle credentials not found!")
        logger.error("Please either:")
        logger.error("  1. Set KAGGLE_API_TOKEN environment variable")
        logger.error("  2. Place kaggle.json in ~/.kaggle/")
        return False

    if kaggle_token:
        logger.info("✅ Using KAGGLE_API_TOKEN environment variable")
    else:
        logger.info("✅ Using kaggle.json file")

    return True


def download_dataset(dataset_name, output_path, unzip=True):
    """
    Download a Kaggle dataset.

    Args:
        dataset_name: Kaggle dataset identifier (e.g., 'username/dataset-name')
        output_path: Directory to save the dataset
        unzip: Whether to automatically unzip the dataset

    Returns:
        True if successful, False otherwise
    """
    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Build kaggle command
    cmd = ['kaggle', 'datasets', 'download', '-d', dataset_name, '--path', output_path]
    if unzip:
        cmd.append('--unzip')

    logger.info(f"Downloading {dataset_name} to {output_path}...")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"✅ Successfully downloaded {dataset_name}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to download {dataset_name}")
        logger.error(f"Error: {e.stderr}")
        return False

    except FileNotFoundError:
        logger.error("❌ Kaggle CLI not found. Please install it: pip install kaggle")
        return False


def main():
    """Main function to download all datasets."""
    logger.info("=" * 80)
    logger.info("Media Recommendation System - Dataset Downloader")
    logger.info("=" * 80)

    # Check credentials
    if not check_kaggle_credentials():
        sys.exit(1)

    # Get project root
    project_root = Path(__file__).parent.parent
    data_raw = project_root / "data" / "raw"

    # Datasets to download
    datasets = [
        {
            'name': 'azathoth42/myanimelist',
            'path': data_raw / 'anime',
            'description': 'MyAnimeList Dataset (Anime)'
        },
        {
            'name': 'duongtruongbinh/manga-and-anime-dataset',
            'path': data_raw / 'manga',
            'description': 'Manga & Anime 2024'
        },
        {
            'name': 'fronkongames/steam-games-dataset',
            'path': data_raw / 'games' / 'steam-games',
            'description': 'Steam Games Dataset 2025'
        },
        {
            'name': 'antonkozyriev/game-recommendations-on-steam',
            'path': data_raw / 'games' / 'steam-recommendations',
            'description': 'Steam Game Recommendations'
        }
    ]

    # Download each dataset
    logger.info(f"\nStarting download of {len(datasets)} datasets...\n")

    success_count = 0
    failed = []

    for i, dataset in enumerate(datasets, 1):
        logger.info(f"[{i}/{len(datasets)}] {dataset['description']}")
        success = download_dataset(dataset['name'], str(dataset['path']), unzip=True)

        if success:
            success_count += 1
        else:
            failed.append(dataset['description'])

        logger.info("")  # Blank line for readability

    # Summary
    logger.info("=" * 80)
    logger.info("Download Summary")
    logger.info("=" * 80)
    logger.info(f"Total datasets: {len(datasets)}")
    logger.info(f"Successfully downloaded: {success_count}")
    logger.info(f"Failed: {len(failed)}")

    if failed:
        logger.error(f"\nFailed downloads:")
        for name in failed:
            logger.error(f"  - {name}")
        sys.exit(1)
    else:
        logger.info("\n✅ All datasets downloaded successfully!")
        logger.info("\nNext steps:")
        logger.info("  1. Verify downloaded files in data/raw/")
        logger.info("  2. Run data pipeline: python src/data_pipeline/pipeline.py --media all")


if __name__ == "__main__":
    main()
