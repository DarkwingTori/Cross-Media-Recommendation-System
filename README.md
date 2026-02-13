# Media Recommendation System

A cross-media recommendation engine that suggests anime, manga, and games based on movie preferences using collaborative filtering, cosine similarity, and sparse matrix optimization.

## Overview

This project implements a sophisticated recommendation system that bridges different media types (movies, anime, manga, games) to provide personalized content discovery. The system leverages:

- **Collaborative Filtering**: Item-based CF with cosine similarity on sparse matrices
- **Cross-Domain Transfer**: Genre-based bridge architecture for multi-media recommendations
- **Hybrid Approach**: Combines collaborative filtering, content-based filtering, and cross-domain signals
- **Scalable Pipeline**: End-to-end ML pipeline from data ingestion to deployment

## Architecture

### Three-Layer Design

1. **Data Layer**: Unified schema across all media types with genre taxonomy mapping
2. **Model Layer**: Item-based collaborative filtering + cross-domain bridge for transfer learning
3. **API Layer**: Backend service supporting Streamlit UI and future frontends

### Key Innovation

Uses genre/theme embeddings as a "bridge" between media types, enabling preference transfer:
- Movies → Anime recommendations
- Movies → Manga recommendations
- Movies → Game recommendations

## Project Structure

```
media-recommendation-system/
├── data/                           # Data storage
│   ├── raw/                        # Original datasets
│   ├── processed/                  # Cleaned data (Parquet)
│   ├── mappings/                   # Genre taxonomy, embeddings
│   └── models/                     # Trained models (NPZ)
├── src/                            # Source code
│   ├── data_pipeline/              # ETL pipeline
│   ├── models/                     # Recommendation algorithms
│   ├── evaluation/                 # Metrics and evaluation
│   ├── api/                        # FastAPI backend
│   └── utils/                      # Utilities
├── ui/streamlit_app/               # Streamlit interface
├── notebooks/                      # Jupyter notebooks for EDA
├── tests/                          # Unit and integration tests
└── scripts/                        # Setup and training scripts
```

## Setup

### 1. Initialize Project

```bash
cd ~/Desktop/media-recommendation-system
python scripts/setup_environment.py
```

This creates the directory structure and virtual environment.

### 2. Install Dependencies

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Download Datasets

The MovieLens 32M dataset should already be linked. Download additional datasets from Kaggle:

- **Anime**: [MyAnimeList Dataset](https://www.kaggle.com/datasets/azathoth42/myanimelist)
- **Manga**: [Manga & Anime 2024](https://www.kaggle.com/datasets/duongtruongbinh/manga-and-anime-dataset)
- **Games**: [Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)
- **Game Ratings**: [Steam Recommendations](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)

Place downloaded datasets in the appropriate `data/raw/` subdirectories.

### 4. Run Data Pipeline

```bash
python src/data_pipeline/pipeline.py
```

This processes raw data into unified format.

### 5. Train Models

```bash
python scripts/train_models.py
```

Trains collaborative filtering models and builds cross-domain bridges.

### 6. Start API Server

```bash
uvicorn src.api.endpoints:app --reload
```

API will be available at `http://localhost:8000`

### 7. Launch Streamlit UI

```bash
streamlit run ui/streamlit_app/app.py
```

## Datasets

### MovieLens 32M
- **Size**: 32 million ratings, 87K movies, 200K users
- **Location**: `~/Downloads/ml-32m/` (symlinked)
- **Source**: [MovieLens](https://grouplens.org/datasets/movielens/)

### Anime/Manga (MyAnimeList)
- **Size**: 300K users, 14K anime, 80M ratings
- **Features**: Genres, themes, studios, scores
- **Download**: Kaggle (see links above)

### Games (Steam)
- **Size**: 110K+ games with metadata
- **Features**: Genres, tags, categories, reviews
- **Download**: Kaggle (see links above)

## Algorithms

### Item-Based Collaborative Filtering

Uses cosine similarity on sparse user-item matrices:

```python
# Build sparse matrix
user_item_matrix = csr_matrix((ratings, (user_idx, item_idx)))

# Compute item-item similarity
item_similarity = cosine_similarity(user_item_matrix.T, dense_output=False)

# Generate recommendations
scores = item_similarity[rated_items, :] @ user_ratings
```

### Cross-Domain Bridge

Connects media types via genre embeddings:

```python
# TF-IDF embeddings from genres+themes
movie_embeddings = TfidfVectorizer().fit_transform(movie_genres)
anime_embeddings = TfidfVectorizer().fit_transform(anime_genres)

# Bridge matrix for transfer
bridge = cosine_similarity(movie_embeddings, anime_embeddings)
```

### Hybrid Scoring

Combines multiple signals:

```
final_score = 0.5 * CF_score + 0.3 * content_score + 0.2 * cross_domain_score
```

## Evaluation Metrics

- **Accuracy**: Precision@10, Recall@10, F1@10, NDCG@10
- **Ranking**: MRR, MAP
- **Coverage**: Catalog coverage, genre diversity
- **Diversity**: Intra-list diversity, serendipity
- **Performance**: API response time, throughput

### Success Criteria

- ✅ Within-domain Precision@10 > 0.30
- ✅ Cross-domain Precision@10 > 0.20
- ✅ Cold start Precision@10 > 0.15
- ✅ API response time < 200ms (p95)
- ✅ Catalog coverage > 60%

## API Endpoints

### POST /recommend
Get cross-media recommendations

```json
{
  "user_ratings": [
    {"item_id": "mov_123", "rating": 4.5, "media_type": "movie"}
  ],
  "target_media": "anime",
  "top_n": 10
}
```

### GET /items/{item_id}
Get item details

### GET /items/search
Search items by query

## Development

### Run Tests

```bash
pytest tests/ -v --cov=src
```

### Explore Data (Jupyter)

```bash
jupyter notebook notebooks/
```

### Format Code

```bash
black src/ tests/
```

## Implementation Phases

- [x] Phase 1: Foundation & Data Pipeline
- [ ] Phase 2: Within-Domain Models
- [ ] Phase 3: Cross-Domain Bridge
- [ ] Phase 4: Hybrid System & Cold Start
- [ ] Phase 5: API & Backend Service
- [ ] Phase 6: Streamlit UI
- [ ] Phase 7: Evaluation & Optimization

## Tech Stack

- **Python 3.10+**
- **Data**: pandas, numpy, scipy
- **ML**: scikit-learn
- **API**: FastAPI, uvicorn
- **UI**: Streamlit
- **Storage**: Parquet (data), NPZ (models)
- **Testing**: pytest

## Contributing

This is an individual project for AI/ML development portfolio.

## License

MIT License

## Contact

Built by Torien Mitchell | Fall 2025
