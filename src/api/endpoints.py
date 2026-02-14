"""
FastAPI endpoints for Media Recommendation API.
REST API serving personalized cross-media recommendations.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
import logging
import time
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from api.models import (
    RecommendRequest,
    RecommendationResponse,
    ItemResponse,
    SearchResponse,
    HealthResponse,
    ErrorResponse
)
from api.service import get_service

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Media Recommendation API",
    description="""
    Cross-media recommendation engine with hybrid algorithms.

    Features:
    - Personalized recommendations across movies, anime, and manga
    - Hybrid algorithm combining collaborative filtering, content-based, and cross-domain signals
    - Cold start handling for new users
    - Fast response times (< 200ms)

    Built with: Item-based CF, TF-IDF embeddings, genre bridges, weighted hybrid scoring
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production (restrict to specific domains)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Get service instance
service = get_service()


@app.on_event("startup")
async def startup_event():
    """Initialize service on application startup."""
    logger.info("Starting up application...")
    await service.initialize()
    logger.info("Application startup complete ✅")


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Media Recommendation API",
        "version": "1.0.0",
        "description": "Cross-media recommendation engine",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns service status and basic statistics.
    """
    return await service.health_check()


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def recommend(request: RecommendRequest):
    """
    Get personalized recommendations.

    Uses adaptive strategy:
    - **Cold start** (< 10 ratings): Genre-based popularity
    - **Warmstart** (>= 10 ratings): Hybrid (CF + content + cross-domain)

    Cross-domain support:
    - Rate movies → get anime/manga recommendations
    - Leverages genre/theme embeddings for preference transfer
    """
    start_time = time.time()

    try:
        # Convert ratings to dict
        user_ratings_dict = {r.item_id: r.rating for r in request.user_ratings}

        # Get recommendations
        recommendations = await service.get_recommendations(
            user_ratings_dict,
            target_media=request.target_media,
            top_n=request.top_n
        )

        # Determine strategy used
        num_ratings = len(user_ratings_dict)
        strategy = "coldstart" if num_ratings < 10 else "hybrid"

        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000

        return {
            "recommendations": recommendations,
            "num_ratings": num_ratings,
            "strategy": strategy,
            "response_time_ms": response_time_ms
        }

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")


@app.get("/items/search", response_model=SearchResponse, tags=["Items"])
async def search_items(
    query: str = Query(..., min_length=1, description="Search query"),
    media_type: Optional[str] = Query(None, pattern="^(movie|anime|manga)$", description="Filter by media type"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results to return")
):
    """
    Search for items by title.

    Performs case-insensitive partial matching on item titles.
    """
    try:
        results = await service.search_items(query, media_type, limit)
        return results

    except Exception as e:
        logger.error(f"Error searching items: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/items/{item_id}", response_model=ItemResponse, tags=["Items"])
async def get_item(item_id: str):
    """
    Get details for a specific item.

    Item IDs are prefixed:
    - `mov_*` for movies
    - `ani_*` for anime
    - `man_*` for manga
    """
    try:
        item_details = await service.get_item_details(item_id)

        if not item_details:
            raise HTTPException(
                status_code=404,
                detail=f"Item {item_id} not found"
            )

        return item_details

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching item: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch item: {str(e)}")


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    """Handle validation errors."""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "error_code": "VALIDATION_ERROR"}
    )


@app.exception_handler(Exception)
async def general_error_handler(request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error_code": "INTERNAL_ERROR"}
    )


if __name__ == "__main__":
    """Run with uvicorn when executed directly."""
    import uvicorn

    uvicorn.run(
        "endpoints:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
