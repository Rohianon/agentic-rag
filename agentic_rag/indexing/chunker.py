"""Document chunking strategies optimized for RAG retrieval."""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import tiktoken


class ChunkType(Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE_DESCRIPTION = "image_description"


@dataclass
class Chunk:
    """Represents a document chunk for indexing."""

    id: str
    content: str
    chunk_type: ChunkType
    metadata: dict = field(default_factory=dict)
    token_count: int = 0

    # Source tracking for citations
    source_file: str = ""
    page_num: int = 0

    # For tables: store structured data separately
    structured_data: dict | None = None


class DocumentChunker:
    """
    Implements chunking strategies optimized for hybrid RAG.

    Key design decisions:
    1. Tables are chunked as single units (never split mid-table)
    2. Text uses semantic paragraph boundaries when possible
    3. Each chunk includes overlap for context continuity
    4. Metadata preserves source location for citations
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        tokenizer: str = "cl100k_base",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding(tokenizer)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    def chunk_text(
        self,
        text: str,
        source_file: str = "",
        page_num: int = 0,
        chunk_id_prefix: str | None = None,
    ) -> list[Chunk]:
        """
        Chunk text content with semantic awareness.

        Strategy:
        1. Split on paragraph boundaries (double newlines)
        2. Merge small paragraphs up to chunk_size
        3. Split large paragraphs at sentence boundaries
        4. Add overlap between chunks
        """
        if not text.strip():
            return []

        # Generate unique prefix if none provided
        if chunk_id_prefix is None:
            unique_suffix = uuid.uuid4().hex[:8]
            if source_file:
                chunk_id_prefix = f"{source_file}_p{page_num}_{unique_suffix}"
            else:
                chunk_id_prefix = f"chunk_{unique_suffix}"

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_idx = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # If single paragraph exceeds chunk size, split it
            if para_tokens > self.chunk_size:
                # Flush current chunk first
                if current_chunk:
                    chunks.append(
                        self._create_chunk(
                            content="\n\n".join(current_chunk),
                            chunk_type=ChunkType.TEXT,
                            source_file=source_file,
                            page_num=page_num,
                            chunk_id=f"{chunk_id_prefix}_{chunk_idx}",
                        )
                    )
                    chunk_idx += 1
                    current_chunk = []
                    current_tokens = 0

                # Split large paragraph by sentences
                sentence_chunks = self._split_by_sentences(para)
                for sent_chunk in sentence_chunks:
                    chunks.append(
                        self._create_chunk(
                            content=sent_chunk,
                            chunk_type=ChunkType.TEXT,
                            source_file=source_file,
                            page_num=page_num,
                            chunk_id=f"{chunk_id_prefix}_{chunk_idx}",
                        )
                    )
                    chunk_idx += 1
                continue

            # Check if adding paragraph exceeds limit
            if current_tokens + para_tokens > self.chunk_size:
                # Create chunk from accumulated content
                chunks.append(
                    self._create_chunk(
                        content="\n\n".join(current_chunk),
                        chunk_type=ChunkType.TEXT,
                        source_file=source_file,
                        page_num=page_num,
                        chunk_id=f"{chunk_id_prefix}_{chunk_idx}",
                    )
                )
                chunk_idx += 1

                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_chunk)
                current_chunk = [overlap_text] if overlap_text else []
                current_tokens = self.count_tokens(overlap_text) if overlap_text else 0

            current_chunk.append(para)
            current_tokens += para_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(
                self._create_chunk(
                    content="\n\n".join(current_chunk),
                    chunk_type=ChunkType.TEXT,
                    source_file=source_file,
                    page_num=page_num,
                    chunk_id=f"{chunk_id_prefix}_{chunk_idx}",
                )
            )

        return chunks

    def chunk_table(
        self,
        table_json: dict,
        table_summary: str,
        source_file: str = "",
        page_num: int = 0,
        chunk_id: str = "tbl_0",
    ) -> Chunk:
        """
        Create a chunk for a table.

        Strategy: Tables are NEVER split. Instead, we create a single chunk
        with the summary text for semantic search, and store the full
        structured data in metadata for precise retrieval.
        """
        # Create searchable text representation
        headers = table_json.get("headers", [])
        rows = table_json.get("rows", [])

        # Build text representation for embedding
        text_repr = f"Table: {table_summary}\n"
        text_repr += f"Columns: {', '.join(headers)}\n"

        # Include first few rows as context
        for i, row in enumerate(rows[:3]):
            text_repr += f"Row {i+1}: {' | '.join(str(v) for v in row)}\n"

        if len(rows) > 3:
            text_repr += f"... and {len(rows) - 3} more rows"

        return self._create_chunk(
            content=text_repr,
            chunk_type=ChunkType.TABLE,
            source_file=source_file,
            page_num=page_num,
            chunk_id=chunk_id,
            structured_data=table_json,
            extra_metadata={
                "table_summary": table_summary,
                "column_count": len(headers),
                "row_count": len(rows),
            },
        )

    def _create_chunk(
        self,
        content: str,
        chunk_type: ChunkType,
        source_file: str,
        page_num: int,
        chunk_id: str,
        structured_data: dict | None = None,
        extra_metadata: dict | None = None,
    ) -> Chunk:
        """Create a Chunk instance with all metadata."""
        metadata = {
            "chunk_type": chunk_type.value,
            "source_file": source_file,
            "page_num": page_num,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return Chunk(
            id=chunk_id,
            content=content,
            chunk_type=chunk_type,
            metadata=metadata,
            token_count=self.count_tokens(content),
            source_file=source_file,
            page_num=page_num,
            structured_data=structured_data,
        )

    def _split_by_sentences(self, text: str) -> list[str]:
        """Split text by sentences, respecting chunk size."""
        # Simple sentence splitting (could use nltk for better results)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self.count_tokens(sent)

            if current_tokens + sent_tokens > self.chunk_size:
                if current:
                    chunks.append(" ".join(current))
                current = [sent]
                current_tokens = sent_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _get_overlap(self, paragraphs: list[str]) -> str:
        """Get overlap text from the end of previous chunk."""
        if not paragraphs:
            return ""

        # Take last paragraph if it fits in overlap budget
        last_para = paragraphs[-1]
        if self.count_tokens(last_para) <= self.chunk_overlap:
            return last_para

        # Otherwise take last N tokens worth of text
        words = last_para.split()
        overlap_words = []
        token_count = 0

        for word in reversed(words):
            word_tokens = self.count_tokens(word)
            if token_count + word_tokens > self.chunk_overlap:
                break
            overlap_words.insert(0, word)
            token_count += word_tokens

        return " ".join(overlap_words)
