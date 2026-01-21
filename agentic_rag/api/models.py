"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime


# ============ Request Models ============

class QueryRequest(BaseModel):
    """Request body for querying indexed documents."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question to ask about your documents",
        json_schema_extra={"example": "What is the maximum operating temperature?"}
    )
    n_chunks: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of relevant document chunks to retrieve for answering"
    )
    apply_guardrails: bool = Field(
        default=True,
        description="Apply safety guardrails to check extracted values against policy limits"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What is the maximum operating temperature?",
                    "n_chunks": 5,
                    "apply_guardrails": True
                }
            ]
        }
    }


# ============ Response Models ============

class Citation(BaseModel):
    """A citation referencing a source document chunk."""

    id: int = Field(description="Citation ID referenced in the answer as [1], [2], etc.")
    source_file: str = Field(description="Name of the source PDF file")
    page: int = Field(description="Page number in the source document")
    chunk_type: str = Field(description="Type of chunk: 'text' or 'table'")
    relevance: float = Field(description="Relevance score from 0.0 to 1.0")
    excerpt: str = Field(description="Brief excerpt from the source content")


class RiskFlag(BaseModel):
    """A policy violation or safety warning from guardrails."""

    metric: str = Field(default="", description="Name of the metric that triggered the flag")
    value: str = Field(default="", description="Actual value found in the document")
    limit: str = Field(default="", description="Policy limit that was exceeded or approached")
    severity: str = Field(default="", description="Severity level: 'info', 'warning', or 'critical'")
    message: str = Field(default="", description="Human-readable description of the risk")


class ExtractedValue(BaseModel):
    """A numerical value extracted from the documents."""

    value: Any = Field(description="The extracted value")
    unit: str = Field(default="", description="Unit of measurement (e.g., '°C', 'V', 'PSI')")
    source: int | None = Field(default=None, description="Citation ID for this value")


class QueryResponse(BaseModel):
    """Response containing the RAG pipeline analysis results."""

    query: str = Field(description="The original query")
    summary: str = Field(description="Executive summary answering the query")
    key_findings: list[str] = Field(description="Bullet points of key findings")
    extracted_data: dict[str, Any] = Field(description="Structured data extracted from documents")
    risk_flags: list[RiskFlag] = Field(description="Policy violations or safety warnings")
    citations: list[Citation] = Field(description="Source references for the answer")
    confidence_score: float = Field(description="Confidence score from 0.0 to 1.0")
    metadata: dict[str, Any] = Field(description="Processing metadata (model, timestamps, etc.)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What is the maximum operating temperature?",
                    "summary": "The maximum operating temperature is 80°C. Operating above this limit may cause damage.",
                    "key_findings": [
                        "Maximum operating temperature: 80°C",
                        "Automatic shutdown triggers at 85°C",
                        "Current temperature is 72°C (within limits)"
                    ],
                    "extracted_data": {
                        "max_temperature": {"value": "80", "unit": "°C", "source": 1}
                    },
                    "risk_flags": [],
                    "citations": [
                        {
                            "id": 1,
                            "source_file": "technical_report.pdf",
                            "page": 1,
                            "chunk_type": "text",
                            "relevance": 0.95,
                            "excerpt": "Maximum Temperature: 80°C (CRITICAL)..."
                        }
                    ],
                    "confidence_score": 0.92,
                    "metadata": {
                        "model": "gpt-4o",
                        "chunks_retrieved": 5
                    }
                }
            ]
        }
    }


class DocumentInfo(BaseModel):
    """Information about an indexed document."""

    filename: str = Field(description="Name of the uploaded PDF file")
    total_pages: int = Field(description="Total number of pages in the document")
    pages_with_tables: int = Field(description="Number of pages containing tables")
    chunks_indexed: int = Field(description="Number of chunks created and indexed")
    indexed_at: datetime = Field(description="Timestamp when the document was indexed")


class UploadResponse(BaseModel):
    """Response after uploading and indexing a document."""

    success: bool = Field(description="Whether the upload and indexing succeeded")
    filename: str = Field(description="Name of the uploaded file")
    message: str = Field(description="Status message")
    document_info: DocumentInfo | None = Field(
        default=None,
        description="Document details (only present on success)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "filename": "technical_report.pdf",
                    "message": "Successfully indexed 5 chunks",
                    "document_info": {
                        "filename": "technical_report.pdf",
                        "total_pages": 2,
                        "pages_with_tables": 1,
                        "chunks_indexed": 5,
                        "indexed_at": "2026-01-21T10:30:00Z"
                    }
                }
            ]
        }
    }


class DocumentListItem(BaseModel):
    """Summary of an indexed document."""

    filename: str = Field(description="Name of the PDF file")
    total_pages: int = Field(description="Total number of pages")
    pages_with_tables: int = Field(description="Number of pages containing tables")


class DocumentListResponse(BaseModel):
    """List of all indexed documents."""

    documents: list[DocumentListItem] = Field(description="List of indexed documents")


class IndexStats(BaseModel):
    """Statistics about the document index."""

    total_chunks: int = Field(description="Total number of indexed chunks")
    text_chunks: int = Field(description="Number of text chunks")
    table_chunks: int = Field(description="Number of table chunks")
    tables_with_data: int = Field(description="Number of tables with structured data")
    indexed_documents: list[str] = Field(description="List of indexed document filenames")


class HealthResponse(BaseModel):
    """API health check response."""

    status: str = Field(description="Health status: 'healthy' or 'unhealthy'")
    timestamp: datetime = Field(description="Current server timestamp")
    index_stats: IndexStats | None = Field(
        default=None,
        description="Index statistics"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "timestamp": "2026-01-21T10:30:00Z",
                    "index_stats": {
                        "total_chunks": 15,
                        "text_chunks": 12,
                        "table_chunks": 3,
                        "tables_with_data": 3,
                        "indexed_documents": ["technical_report.pdf", "product_spec.pdf"]
                    }
                }
            ]
        }
    }
