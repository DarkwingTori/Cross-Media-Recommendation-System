#!/usr/bin/env python3
"""
Setup script for Media Recommendation System
Creates directory structure and initializes Python packages
"""

import os
import subprocess
from pathlib import Path


def create_directory_structure():
    """Create all project directories"""
    base_dir = Path(__file__).parent.parent

    directories = [
        # Data directories
        "data/raw/movies",
        "data/raw/anime",
        "data/raw/manga",
        "data/raw/games",
        "data/processed",
        "data/mappings",
        "data/models",

        # Source directories
        "src/config",
        "src/data_pipeline/ingest",
        "src/data_pipeline/preprocessing",
        "src/data_pipeline/transform",
        "src/models/collaborative",
        "src/models/content_based",
        "src/models/hybrid",
        "src/models/cross_domain",
        "src/evaluation/metrics",
        "src/evaluation/validators",
        "src/api",
        "src/utils",

        # UI directories
        "ui/streamlit_app/pages",
        "ui/streamlit_app/components",
        "ui/future_figma",

        # Notebook directories
        "notebooks",

        # Test directories
        "tests/unit",
        "tests/integration",
        "tests/fixtures",

        # Script directories
        "scripts",

        # Documentation
        "docs",

        # Logs
        "logs",
    ]

    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {dir_path}")

        # Create __init__.py for Python packages
        if directory.startswith("src/") or directory.startswith("tests/") or directory.startswith("ui/"):
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                print(f"  Created: {init_file}")


def create_symlink_to_movielens():
    """Create symlink to existing MovieLens dataset"""
    base_dir = Path(__file__).parent.parent
    movielens_src = Path.home() / "Downloads" / "ml-32m"
    movielens_dst = base_dir / "data" / "raw" / "movies" / "ml-32m"

    if movielens_src.exists():
        if not movielens_dst.exists():
            try:
                movielens_dst.symlink_to(movielens_src)
                print(f"\nSymlink created: {movielens_dst} -> {movielens_src}")
            except Exception as e:
                print(f"\nWarning: Could not create symlink to MovieLens data: {e}")
                print(f"You can manually copy the data from {movielens_src}")
        else:
            print(f"\nMovieLens symlink already exists: {movielens_dst}")
    else:
        print(f"\nWarning: MovieLens dataset not found at {movielens_src}")
        print("Please ensure the dataset is downloaded")


def create_venv():
    """Create virtual environment"""
    base_dir = Path(__file__).parent.parent
    venv_path = base_dir / "venv"

    if not venv_path.exists():
        print(f"\nCreating virtual environment at {venv_path}...")
        try:
            subprocess.run(["python3", "-m", "venv", str(venv_path)], check=True)
            print(f"Virtual environment created successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e}")
            print("You can create it manually with: python3 -m venv venv")
    else:
        print(f"\nVirtual environment already exists at {venv_path}")


def main():
    """Main setup function"""
    print("=" * 60)
    print("Media Recommendation System - Environment Setup")
    print("=" * 60)

    # Create directory structure
    print("\n1. Creating directory structure...")
    create_directory_structure()

    # Create symlink to MovieLens data
    print("\n2. Setting up MovieLens dataset...")
    create_symlink_to_movielens()

    # Create virtual environment
    print("\n3. Creating virtual environment...")
    create_venv()

    # Print next steps
    base_dir = Path(__file__).parent.parent
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print(f"1. cd {base_dir}")
    print("2. source venv/bin/activate")
    print("3. pip install -r requirements.txt")
    print("4. python scripts/download_datasets.py")
    print("\nProject initialized successfully! 🚀")


if __name__ == "__main__":
    main()
