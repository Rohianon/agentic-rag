"""Shared dependencies for API endpoints."""

from dataclasses import dataclass, field
from openai import OpenAI
from fastapi import Request

from agentic_rag.ingestion.pdf_parser import PDFParser, ParsedDocument
from agentic_rag.ingestion.table_extractor import TableExtractor
from agentic_rag.indexing.chunker import DocumentChunker
from agentic_rag.indexing.hybrid_index import HybridIndex
from agentic_rag.retrieval.hybrid_retriever import HybridRetriever
from agentic_rag.agent.guardrails import create_manufacturing_guardrail
from agentic_rag.agent.reasoning import ReasoningAgent
from agentic_rag.output.synthesizer import OutputSynthesizer


@dataclass
class PipelineState:
    """Holds initialized pipeline components."""
    client: OpenAI | None = None
    parser: PDFParser | None = None
    table_extractor: TableExtractor | None = None
    chunker: DocumentChunker | None = None
    index: HybridIndex | None = None
    retriever: HybridRetriever | None = None
    reasoning_agent: ReasoningAgent | None = None
    synthesizer: OutputSynthesizer | None = None

    # Track indexed documents
    indexed_documents: dict[str, ParsedDocument] = field(default_factory=dict)

    def initialize(self):
        """Initialize all pipeline components."""
        self.client = OpenAI()
        self.parser = PDFParser(extract_images=True, image_dpi=150)
        self.table_extractor = TableExtractor(client=self.client, model="gpt-4o")
        self.chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
        self.index = HybridIndex(
            collection_name="web_documents",
            openai_client=self.client,
            embedding_model="text-embedding-3-small"
        )
        self.retriever = HybridRetriever(
            index=self.index,
            openai_client=self.client,
            relevance_threshold=0.3,
            explain_retrievals=False
        )
        guardrail = create_manufacturing_guardrail()
        self.reasoning_agent = ReasoningAgent(
            retriever=self.retriever,
            guardrail=guardrail,
            openai_client=self.client,
            model="gpt-4o"
        )
        self.synthesizer = OutputSynthesizer(openai_client=self.client)


def get_pipeline(request: Request) -> PipelineState:
    """Dependency to get pipeline state from app."""
    return request.app.state.pipeline
