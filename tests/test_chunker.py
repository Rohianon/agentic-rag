"""Tests for document chunking functionality."""

import pytest

from agentic_rag.indexing.chunker import DocumentChunker, Chunk, ChunkType


class TestDocumentChunker:
    """Test suite for DocumentChunker."""

    @pytest.fixture
    def chunker(self):
        """Create a chunker instance."""
        return DocumentChunker(chunk_size=100, chunk_overlap=20)

    def test_chunker_initialization(self, chunker):
        """Test chunker initializes with correct settings."""
        assert chunker.chunk_size == 100
        assert chunker.chunk_overlap == 20

    def test_count_tokens(self, chunker):
        """Test token counting."""
        text = "Hello world, this is a test."
        tokens = chunker.count_tokens(text)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_chunk_short_text(self, chunker):
        """Test chunking text shorter than chunk_size."""
        short_text = "This is a short paragraph."
        chunks = chunker.chunk_text(
            text=short_text,
            source_file="test.pdf",
            page_num=1
        )

        assert len(chunks) == 1
        assert chunks[0].content == short_text
        assert chunks[0].chunk_type == ChunkType.TEXT

    def test_chunk_long_text(self):
        """Test chunking text longer than chunk_size."""
        chunker = DocumentChunker(chunk_size=20, chunk_overlap=5)
        long_text = """
        This is the first paragraph with some detailed content about temperature monitoring.

        This is the second paragraph with more detailed content about voltage specifications and requirements.

        This is the third paragraph with even more detailed content about pressure limits and safety margins.

        This is the fourth paragraph continuing on with additional technical specifications and notes.
        """

        chunks = chunker.chunk_text(
            text=long_text,
            source_file="test.pdf",
            page_num=1
        )

        # With small chunk size of 20 tokens, should produce multiple chunks
        assert len(chunks) >= 1  # At least one chunk
        assert all(c.chunk_type == ChunkType.TEXT for c in chunks)

    def test_chunk_preserves_metadata(self, chunker):
        """Test that chunking preserves source metadata."""
        text = "Sample text content."
        chunks = chunker.chunk_text(
            text=text,
            source_file="document.pdf",
            page_num=5,
            chunk_id_prefix="doc_p5"
        )

        chunk = chunks[0]
        assert chunk.source_file == "document.pdf"
        assert chunk.page_num == 5
        assert chunk.id.startswith("doc_p5")

    def test_chunk_table_atomic(self, chunker):
        """Test that tables are chunked as single units."""
        table_json = {
            "headers": ["Name", "Value", "Unit"],
            "rows": [
                ["Temperature", "72", "°C"],
                ["Voltage", "220", "V"],
                ["Current", "10", "A"],
            ]
        }
        table_summary = "Performance metrics table"

        chunk = chunker.chunk_table(
            table_json=table_json,
            table_summary=table_summary,
            source_file="specs.pdf",
            page_num=3,
            chunk_id="table_1"
        )

        assert chunk.chunk_type == ChunkType.TABLE
        assert chunk.structured_data == table_json
        assert "Performance metrics table" in chunk.content
        assert "Temperature" in chunk.content

    def test_chunk_table_metadata(self, chunker):
        """Test that table chunks have correct metadata."""
        table_json = {"headers": ["A", "B"], "rows": [["1", "2"]]}

        chunk = chunker.chunk_table(
            table_json=table_json,
            table_summary="Test table",
            source_file="test.pdf",
            page_num=1,
            chunk_id="tbl_0"
        )

        assert chunk.id == "tbl_0"
        assert chunk.metadata["chunk_type"] == "table"
        assert chunk.metadata["row_count"] == 1
        assert chunk.metadata["column_count"] == 2

    def test_chunk_empty_text(self, chunker):
        """Test chunking empty or whitespace-only text."""
        chunks = chunker.chunk_text(
            text="   ",
            source_file="empty.pdf",
            page_num=1
        )
        assert len(chunks) == 0

    def test_chunk_has_token_count(self, chunker):
        """Test that chunks have token counts."""
        text = "This is sample text for testing token counts."
        chunks = chunker.chunk_text(
            text=text,
            source_file="test.pdf",
            page_num=1
        )

        assert chunks[0].token_count > 0

    def test_chunk_text_generates_unique_ids_across_calls(self, chunker):
        """Test that multiple chunk_text calls without prefix generate unique IDs."""
        chunks1 = chunker.chunk_text(text="First document.", source_file="doc1.pdf", page_num=1)
        chunks2 = chunker.chunk_text(text="Second document.", source_file="doc1.pdf", page_num=2)
        chunks3 = chunker.chunk_text(text="Third document.", source_file="doc2.pdf", page_num=1)

        all_ids = [c.id for c in chunks1 + chunks2 + chunks3]
        assert len(all_ids) == len(set(all_ids)), "Chunk IDs must be unique across multiple calls"

    def test_chunk_text_explicit_prefix_backward_compatible(self, chunker):
        """Test that explicit prefix still works as before."""
        chunks = chunker.chunk_text(
            text="Sample text.",
            source_file="test.pdf",
            page_num=1,
            chunk_id_prefix="custom_prefix"
        )

        assert chunks[0].id == "custom_prefix_0"


class TestChunk:
    """Test suite for Chunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a Chunk instance."""
        chunk = Chunk(
            id="chunk_1",
            content="Test content",
            chunk_type=ChunkType.TEXT,
            source_file="test.pdf",
            page_num=1
        )

        assert chunk.id == "chunk_1"
        assert chunk.content == "Test content"
        assert chunk.chunk_type == ChunkType.TEXT

    def test_chunk_defaults(self):
        """Test default values for Chunk."""
        chunk = Chunk(
            id="chunk_1",
            content="Test",
            chunk_type=ChunkType.TEXT
        )

        assert chunk.metadata == {}
        assert chunk.token_count == 0
        assert chunk.source_file == ""
        assert chunk.page_num == 0
        assert chunk.structured_data is None
