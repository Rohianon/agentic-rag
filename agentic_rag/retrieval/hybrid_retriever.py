"""Hybrid retrieval with explainability and relevance filtering."""

from dataclasses import dataclass, field
from enum import Enum

from openai import OpenAI

from agentic_rag.indexing.hybrid_index import HybridIndex
from agentic_rag.indexing.chunker import ChunkType


class RetrievalReason(Enum):
    """Why a chunk was retrieved."""

    SEMANTIC_MATCH = "semantic_similarity"
    KEYWORD_MATCH = "keyword_match"
    TABLE_DATA = "contains_relevant_table"
    NUMERIC_DATA = "contains_numeric_data"


@dataclass
class RetrievedChunk:
    """A retrieved chunk with explanation."""

    id: str
    content: str
    relevance_score: float
    source_file: str
    page_num: int
    chunk_type: str
    retrieval_reasons: list[RetrievalReason] = field(default_factory=list)
    explanation: str = ""
    structured_data: dict | None = None


@dataclass
class RetrievalResult:
    """Complete retrieval result with context."""

    query: str
    chunks: list[RetrievedChunk]
    total_retrieved: int
    filtered_count: int
    context_window: str = ""


class HybridRetriever:
    """
    Retrieval system with explainability and relevance filtering.

    Features:
    1. Hybrid search: semantic + metadata filtering
    2. Relevance scoring and thresholding
    3. Retrieval explanations (why each chunk was selected)
    4. Context assembly for LLM consumption
    """

    RELEVANCE_PROMPT = """Analyze why this document chunk is relevant to the query.

Query: {query}

Chunk content:
{content}

Chunk metadata:
- Source: {source_file}, Page {page_num}
- Type: {chunk_type}

Provide a brief (1-2 sentence) explanation of:
1. Why this chunk is relevant to the query
2. What specific information it contains that helps answer the query

If the chunk is NOT relevant, say "NOT_RELEVANT" and explain why.

Response format:
RELEVANT: [explanation]
or
NOT_RELEVANT: [reason]"""

    def __init__(
        self,
        index: HybridIndex,
        openai_client: OpenAI | None = None,
        relevance_threshold: float = 0.3,
        explain_retrievals: bool = True,
        model: str = "gpt-4o-mini",
    ):
        self.index = index
        self.client = openai_client or OpenAI()
        self.relevance_threshold = relevance_threshold
        self.explain_retrievals = explain_retrievals
        self.model = model

    def retrieve(
        self,
        query: str,
        n_results: int = 10,
        filter_irrelevant: bool = True,
        chunk_type: ChunkType | None = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks with explanations.

        Process:
        1. Vector search for initial candidates
        2. Score and filter by relevance threshold
        3. Generate explanations for why each chunk was retrieved
        4. Assemble context window for LLM
        """
        # Initial retrieval
        raw_results = self.index.search(
            query=query,
            n_results=n_results * 2,  # Over-fetch for filtering
            chunk_type=chunk_type,
        )

        # Convert to RetrievedChunk objects
        chunks = []
        for result in raw_results:
            chunk = RetrievedChunk(
                id=result["id"],
                content=result["content"],
                relevance_score=result["relevance_score"],
                source_file=result["metadata"]["source_file"],
                page_num=result["metadata"]["page_num"],
                chunk_type=result["metadata"]["chunk_type"],
                structured_data=result.get("structured_data"),
            )

            # Determine retrieval reasons
            chunk.retrieval_reasons = self._determine_reasons(chunk, query)

            chunks.append(chunk)

        # Filter by relevance threshold
        filtered_chunks = chunks
        if filter_irrelevant:
            filtered_chunks = [
                c for c in chunks
                if c.relevance_score >= self.relevance_threshold
            ]

        # Generate explanations
        if self.explain_retrievals:
            filtered_chunks = self._add_explanations(query, filtered_chunks)
            # Remove chunks marked as NOT_RELEVANT
            filtered_chunks = [
                c for c in filtered_chunks
                if not c.explanation.startswith("NOT_RELEVANT")
            ]

        # Limit to requested count
        final_chunks = filtered_chunks[:n_results]

        # Build context window
        context = self._build_context(query, final_chunks)

        return RetrievalResult(
            query=query,
            chunks=final_chunks,
            total_retrieved=len(raw_results),
            filtered_count=len(raw_results) - len(final_chunks),
            context_window=context,
        )

    def _determine_reasons(
        self, chunk: RetrievedChunk, query: str
    ) -> list[RetrievalReason]:
        """Determine why this chunk was retrieved."""
        reasons = []

        # Always semantic match (that's how we retrieved it)
        reasons.append(RetrievalReason.SEMANTIC_MATCH)

        # Check for keyword overlap
        query_words = set(query.lower().split())
        content_words = set(chunk.content.lower().split())
        if query_words & content_words:
            reasons.append(RetrievalReason.KEYWORD_MATCH)

        # Table data
        if chunk.chunk_type == "table":
            reasons.append(RetrievalReason.TABLE_DATA)

        # Numeric data
        if any(c.isdigit() for c in chunk.content):
            reasons.append(RetrievalReason.NUMERIC_DATA)

        return reasons

    def _add_explanations(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Add LLM-generated explanations to chunks."""
        for chunk in chunks:
            prompt = self.RELEVANCE_PROMPT.format(
                query=query,
                content=chunk.content[:1000],  # Truncate for efficiency
                source_file=chunk.source_file,
                page_num=chunk.page_num,
                chunk_type=chunk.chunk_type,
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0,
            )

            chunk.explanation = response.choices[0].message.content.strip()

        return chunks

    def _build_context(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> str:
        """Build context window for LLM consumption."""
        context_parts = [f"Query: {query}\n", "Retrieved Context:\n"]

        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"--- Source {i}: {chunk.source_file} (Page {chunk.page_num}) ---")
            context_parts.append(f"Type: {chunk.chunk_type}")
            context_parts.append(f"Relevance: {chunk.relevance_score:.2f}")
            context_parts.append(f"Content:\n{chunk.content}")

            if chunk.structured_data:
                context_parts.append(f"Structured Data: {chunk.structured_data}")

            context_parts.append("")

        return "\n".join(context_parts)

    def get_retrieval_summary(self, result: RetrievalResult) -> dict:
        """Generate a summary of the retrieval process."""
        return {
            "query": result.query,
            "chunks_retrieved": len(result.chunks),
            "chunks_filtered": result.filtered_count,
            "sources": list({c.source_file for c in result.chunks}),
            "pages": list({c.page_num for c in result.chunks}),
            "chunk_types": list({c.chunk_type for c in result.chunks}),
            "avg_relevance": (
                sum(c.relevance_score for c in result.chunks) / len(result.chunks)
                if result.chunks else 0
            ),
            "retrieval_reasons": [
                {
                    "chunk_id": c.id,
                    "reasons": [r.value for r in c.retrieval_reasons],
                    "explanation": c.explanation,
                }
                for c in result.chunks
            ],
        }
