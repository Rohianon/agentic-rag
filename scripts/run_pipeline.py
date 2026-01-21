"""Run the complete RAG pipeline to verify functionality."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import json
from openai import OpenAI

from agentic_rag.ingestion.pdf_parser import PDFParser
from agentic_rag.ingestion.table_extractor import TableExtractor
from agentic_rag.indexing.chunker import DocumentChunker, ChunkType
from agentic_rag.indexing.hybrid_index import HybridIndex
from agentic_rag.retrieval.hybrid_retriever import HybridRetriever
from agentic_rag.agent.guardrails import create_manufacturing_guardrail
from agentic_rag.agent.reasoning import ReasoningAgent
from agentic_rag.output.synthesizer import OutputSynthesizer


def main():
    print("=" * 60)
    print("AGENTIC RAG PIPELINE TEST RUN")
    print("=" * 60)

    # Initialize OpenAI client
    client = OpenAI()

    # Phase 1: Parse PDFs
    print("\n[Phase 1] Parsing PDFs...")
    parser = PDFParser(extract_images=True, image_dpi=150)
    pdf_dir = Path(__file__).parent.parent / "data" / "pdfs"
    documents = parser.parse_directory(pdf_dir)

    print(f"  Parsed {len(documents)} documents:")
    for doc in documents:
        pages_with_tables = len(doc.get_pages_with_tables())
        print(f"    - {doc.filename}: {doc.total_pages} pages, {pages_with_tables} with tables")

    # Phase 2: Extract tables (skip for speed, use text detection)
    print("\n[Phase 2] Extracting tables with Vision LLM...")
    table_extractor = TableExtractor(client=client, model="gpt-4o")

    # Only extract from first document with tables for demo
    all_tables = []
    for doc in documents[:1]:  # Limit for speed
        tables = table_extractor.extract_from_pages(doc.pages, only_table_pages=True)
        all_tables.extend(tables)
        print(f"  {doc.filename}: Extracted {len(tables)} tables")

    # Phase 3: Chunking
    print("\n[Phase 3] Chunking documents...")
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
    all_chunks = []

    for doc in documents:
        for page in doc.pages:
            chunks = chunker.chunk_text(
                text=page.text,
                source_file=doc.filename,
                page_num=page.page_num,
                chunk_id_prefix=f"{doc.filename}_p{page.page_num}"
            )
            all_chunks.extend(chunks)

    for i, table in enumerate(all_tables):
        chunk = chunker.chunk_table(
            table_json=table.table_json,
            table_summary=table.table_summary,
            source_file=table.source_file,
            page_num=table.page_num,
            chunk_id=f"table_{i}"
        )
        all_chunks.append(chunk)

    text_chunks = sum(1 for c in all_chunks if c.chunk_type == ChunkType.TEXT)
    table_chunks = sum(1 for c in all_chunks if c.chunk_type == ChunkType.TABLE)
    print(f"  Total chunks: {len(all_chunks)} (text: {text_chunks}, tables: {table_chunks})")

    # Phase 4: Indexing
    print("\n[Phase 4] Building hybrid index...")
    index = HybridIndex(
        collection_name="test_run",
        openai_client=client,
        embedding_model="text-embedding-3-small"
    )
    index.clear()
    added = index.add_chunks(all_chunks)
    print(f"  Added {added} chunks to index")
    print(f"  Stats: {index.get_stats()}")

    # Phase 5: Retrieval
    print("\n[Phase 5] Testing retrieval...")
    retriever = HybridRetriever(
        index=index,
        openai_client=client,
        relevance_threshold=0.3,
        explain_retrievals=False  # Skip for speed
    )

    test_query = "What is the maximum operating temperature?"
    result = retriever.retrieve(test_query, n_results=3)
    print(f"  Query: '{test_query}'")
    print(f"  Retrieved {len(result.chunks)} chunks")
    for chunk in result.chunks:
        print(f"    - {chunk.source_file} p{chunk.page_num}: {chunk.relevance_score:.2f}")

    # Phase 6: Reasoning
    print("\n[Phase 6] Running reasoning agent...")
    guardrail = create_manufacturing_guardrail()
    agent = ReasoningAgent(
        retriever=retriever,
        guardrail=guardrail,
        openai_client=client,
        model="gpt-4o"
    )

    query = "What is the current operating temperature and is it within safe limits?"
    reasoning_result = agent.reason(query, n_chunks=5)

    print(f"  Query: '{query}'")
    print(f"  Confidence: {reasoning_result.confidence:.0%}")
    print(f"  Answer: {reasoning_result.answer[:200]}...")
    if reasoning_result.risk_flags:
        print(f"  RISK FLAGS: {reasoning_result.risk_flags}")

    # Phase 7: Output synthesis
    print("\n[Phase 7] Synthesizing output...")
    synthesizer = OutputSynthesizer(openai_client=client)
    output = synthesizer.synthesize(
        query=query,
        reasoning_output=reasoning_result,
        retrieval_result=result
    )

    print("\n" + "=" * 60)
    print("STRUCTURED OUTPUT")
    print("=" * 60)
    print(synthesizer.format_for_display(output))

    print("\n" + "=" * 60)
    print("JSON OUTPUT")
    print("=" * 60)
    print(output.to_json())

    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETE")
    print("=" * 60)

    return output


if __name__ == "__main__":
    main()
