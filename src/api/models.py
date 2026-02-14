"""
Pydantic models for API request/response schemas.
Type-safe contracts for the Media Recommendation API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum


class MediaType(str, Enum):
    """Valid media types."""
    MOVIE = "movie"
    ANIME = "anime"
    MANGA = "manga"


class UserRating(BaseModel):
    """Single user rating for an item."""
    item_id: str = Field(
        ...,
        description="Item ID (e.g., mov_1, ani_123, man_456)",
        examples=["mov_1", "ani_100", "man_50"]
    )
    rating: float = Field(
        ...,
        ge=0.5,
        le=10.0,
        description="Rating value (0.5-10.0 scale)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "item_id": "mov_1",
                "rating": 5.0
            }
        }


class RecommendRequest(BaseModel):
    """Request for personalized recommendations."""
    user_ratings: List[UserRating] = Field(
        ...,
        min_length=1,
        description="User's item ratings (at least 1 required)"
    )
    target_media: MediaType = Field(
        default=MediaType.MOVIE,
        description="Target media type for recommendations"
    )
    top_n: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of recommendations to return (1-100)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_ratings": [
                    {"item_id": "mov_1", "rating": 5.0},
                    {"item_id": "mov_260", "rating": 4.5}
                ],
                "target_media": "movie",
                "top_n": 10
            }
        }


class RecommendationItem(BaseModel):
    """Single recommendation with full item details."""
    item_id: str = Field(..., description="Unique item identifier")
    title: str = Field(..., description="Item title")
    media_type: str = Field(..., description="Media type (movie/anime/manga)")
    genres: List[str] = Field(..., description="List of genres")
    avg_rating: float = Field(..., description="Average user rating")
    year: Optional[int] = Field(None, description="Release/publication year")
    score: float = Field(..., description="Recommendation score (0.0-1.0)")

    class Config:
        json_schema_extra = {
            "example": {
                "item_id": "mov_2571",
                "title": "The Matrix",
                "media_type": "movie",
                "genres": ["Action", "Sci-Fi"],
                "avg_rating": 4.19,
                "year": 1999,
                "score": 0.87
            }
        }


class RecommendationResponse(BaseModel):
    """Response containing personalized recommendations."""
    recommendations: List[RecommendationItem] = Field(
        ...,
        description="List of recommended items"
    )
    num_ratings: int = Field(..., description="Number of ratings provided by user")
    strategy: str = Field(..., description="Recommendation strategy used (coldstart/hybrid)")
    response_time_ms: float = Field(..., description="API response time in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "recommendations": [
                    {
                        "item_id": "mov_2571",
                        "title": "The Matrix",
                        "media_type": "movie",
                        "genres": ["Action", "Sci-Fi"],
                        "avg_rating": 4.19,
                        "year": 1999,
                        "score": 0.87
                    }
                ],
                "num_ratings": 2,
                "strategy": "coldstart",
                "response_time_ms": 145.2
            }
        }


class ItemResponse(BaseModel):
    """Item details response."""
    item_id: str
    title: str
    media_type: str
    genres: List[str]
    themes: List[str]
    avg_rating: float
    rating_count: int
    year: Optional[int]

    class Config:
        json_schema_extra = {
            "example": {
                "item_id": "mov_1",
                "title": "Toy Story",
                "media_type": "movie",
                "genres": ["Adventure", "Comedy", "Fantasy"],
                "themes": ["buddy-cop", "superhero"],
                "avg_rating": 3.92,
                "rating_count": 81847,
                "year": 1995
            }
        }


class SearchResponse(BaseModel):
    """Search results response."""
    results: List[ItemResponse] = Field(..., description="List of matching items")
    total_results: int = Field(..., description="Total number of matches")

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "item_id": "mov_2571",
                        "title": "The Matrix",
                        "media_type": "movie",
                        "genres": ["Action", "Sci-Fi"],
                        "themes": ["sci-fi", "cyberpunk"],
                        "avg_rating": 4.19,
                        "rating_count": 67946,
                        "year": 1999
                    }
                ],
                "total_results": 1
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status (healthy/unhealthy)")
    models_loaded: bool = Field(..., description="Whether all models are loaded")
    items_count: Dict[str, int] = Field(..., description="Item count per media type")
    version: str = Field(default="1.0.0", description="API version")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "models_loaded": True,
                "items_count": {
                    "movie": 31961,
                    "anime": 14478,
                    "manga": 10000
                },
                "version": "1.0.0"
            }
        }


class ErrorResponse(BaseModel):
    """Error response."""
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code for debugging")

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Item not found",
                "error_code": "ITEM_NOT_FOUND"
            }
        }
