#!/usr/bin/env python3
"""
API server startup script.
Runs the FastAPI recommendation service.
"""

import uvicorn
import sys
import argparse
from pathlib import Path


def main():
    """Start the FastAPI server."""
    parser = argparse.ArgumentParser(description="Start Media Recommendation API")

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development mode)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (production mode)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Media Recommendation API Server")
    print("=" * 80)
    print(f"Starting server on http://{args.host}:{args.port}")
    print(f"OpenAPI docs: http://{args.host}:{args.port}/docs")
    print(f"ReDoc: http://{args.host}:{args.port}/redoc")
    print("=" * 80 + "\n")

    # Run server
    uvicorn.run(
        "src.api.endpoints:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,  # reload doesn't work with workers
        log_level="info"
    )


if __name__ == "__main__":
    main()
