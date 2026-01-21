"""Query endpoint for RAG pipeline."""

from fastapi import APIRouter, Depends, HTTPException

from ..models import QueryRequest, QueryResponse, Citation, RiskFlag
from ..dependencies import get_pipeline, PipelineState

router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    pipeline: PipelineState = Depends(get_pipeline),
):
    """
    Query indexed documents using the full RAG pipeline.

    Pipeline: Retrieve -> Reason -> Synthesize
    """
    # Check if index has data
    stats = pipeline.index.get_stats()
    if stats["total_chunks"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Please upload a PDF first."
        )

    try:
        # Phase 5: Retrieve (internal to reasoning)
        # Phase 6: Reason
        reasoning_result = pipeline.reasoning_agent.reason(
            query=request.query,
            n_chunks=request.n_chunks,
            apply_guardrails=request.apply_guardrails,
        )

        # Get retrieval result for metadata
        retrieval_result = pipeline.retriever.retrieve(
            query=request.query,
            n_results=request.n_chunks,
        )

        # Phase 7: Synthesize
        output = pipeline.synthesizer.synthesize(
            query=request.query,
            reasoning_output=reasoning_result,
            retrieval_result=retrieval_result,
        )

        # Convert to response model using existing to_dict()
        output_dict = output.to_dict()

        # Build citations list
        citations = []
        for c in output_dict.get("citations", []):
            citations.append(Citation(
                id=c.get("id", 0),
                source_file=c.get("source_file", ""),
                page=c.get("page", 0),
                chunk_type=c.get("chunk_type", "text"),
                relevance=c.get("relevance", 0.0),
                excerpt=c.get("excerpt", "")
            ))

        # Build risk flags list
        risk_flags = []
        for rf in output_dict.get("risk_flags", []):
            risk_flags.append(RiskFlag(
                metric=rf.get("metric", ""),
                value=str(rf.get("value", "")),
                limit=str(rf.get("limit", "")),
                severity=rf.get("severity", ""),
                message=rf.get("message", "")
            ))

        return QueryResponse(
            query=output_dict["query"],
            summary=output_dict["summary"],
            key_findings=output_dict["key_findings"],
            extracted_data=output_dict["extracted_data"],
            risk_flags=risk_flags,
            citations=citations,
            confidence_score=output_dict["confidence_score"],
            metadata=output_dict["metadata"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
