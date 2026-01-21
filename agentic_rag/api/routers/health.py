"""Health check and statistics endpoints."""

from fastapi import APIRouter, Depends
from datetime import datetime

from ..models import HealthResponse, IndexStats
from ..dependencies import get_pipeline, PipelineState

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API is healthy and get current index statistics.",
)
async def health_check(pipeline: PipelineState = Depends(get_pipeline)):
    """
    Returns the API health status and index statistics.

    Use this endpoint to:
    - Verify the API is running
    - Check how many documents are indexed
    - Monitor chunk counts (text vs table)
    """
    stats = pipeline.index.get_stats()

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        index_stats=IndexStats(
            total_chunks=stats["total_chunks"],
            text_chunks=stats["text_chunks"],
            table_chunks=stats["table_chunks"],
            tables_with_data=stats["tables_with_data"],
            indexed_documents=list(pipeline.indexed_documents.keys())
        )
    )


@router.get(
    "/stats",
    response_model=IndexStats,
    summary="Get Stats",
    description="Get detailed statistics about the document index.",
)
async def get_stats(pipeline: PipelineState = Depends(get_pipeline)):
    """
    Returns detailed index statistics.

    Statistics include:
    - Total number of chunks in the index
    - Breakdown by chunk type (text vs table)
    - List of all indexed document filenames
    """
    stats = pipeline.index.get_stats()
    return IndexStats(
        total_chunks=stats["total_chunks"],
        text_chunks=stats["text_chunks"],
        table_chunks=stats["table_chunks"],
        tables_with_data=stats["tables_with_data"],
        indexed_documents=list(pipeline.indexed_documents.keys())
    )
