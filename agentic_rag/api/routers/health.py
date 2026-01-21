"""Health check and statistics endpoints."""

from fastapi import APIRouter, Depends
from datetime import datetime

from ..models import HealthResponse, IndexStats
from ..dependencies import get_pipeline, PipelineState

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(pipeline: PipelineState = Depends(get_pipeline)):
    """Check API health and return index stats."""
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


@router.get("/stats", response_model=IndexStats)
async def get_stats(pipeline: PipelineState = Depends(get_pipeline)):
    """Get detailed index statistics."""
    stats = pipeline.index.get_stats()
    return IndexStats(
        total_chunks=stats["total_chunks"],
        text_chunks=stats["text_chunks"],
        table_chunks=stats["table_chunks"],
        tables_with_data=stats["tables_with_data"],
        indexed_documents=list(pipeline.indexed_documents.keys())
    )
