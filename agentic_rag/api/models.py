"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime


# ============ Request Models ============

class QueryRequest(BaseModel):
    """Request body for query endpoint."""
    query: str = Field(..., min_length=1, max_length=1000)
    n_chunks: int = Field(default=5, ge=1, le=20)
    apply_guardrails: bool = True


# ============ Response Models ============

class Citation(BaseModel):
    """Citation reference in response."""
    id: int
    source_file: str
    page: int
    chunk_type: str
    relevance: float
    excerpt: str


class RiskFlag(BaseModel):
    """Risk flag from guardrails."""
    metric: str = ""
    value: str = ""
    limit: str = ""
    severity: str = ""
    message: str = ""


class ExtractedValue(BaseModel):
    """Extracted numerical value."""
    value: Any
    unit: str = ""
    source: int | None = None


class QueryResponse(BaseModel):
    """Response for query endpoint."""
    query: str
    summary: str
    key_findings: list[str]
    extracted_data: dict[str, Any]
    risk_flags: list[RiskFlag]
    citations: list[Citation]
    confidence_score: float
    metadata: dict[str, Any]


class DocumentInfo(BaseModel):
    """Document information."""
    filename: str
    total_pages: int
    pages_with_tables: int
    chunks_indexed: int
    indexed_at: datetime


class UploadResponse(BaseModel):
    """Response for document upload."""
    success: bool
    filename: str
    message: str
    document_info: DocumentInfo | None = None


class DocumentListItem(BaseModel):
    """Document in list response."""
    filename: str
    total_pages: int
    pages_with_tables: int


class DocumentListResponse(BaseModel):
    """Response for document list endpoint."""
    documents: list[DocumentListItem]


class IndexStats(BaseModel):
    """Index statistics."""
    total_chunks: int
    text_chunks: int
    table_chunks: int
    tables_with_data: int
    indexed_documents: list[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    index_stats: IndexStats | None = None
