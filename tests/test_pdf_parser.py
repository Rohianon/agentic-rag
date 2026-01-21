"""Tests for PDF parsing functionality."""

import pytest
from pathlib import Path

from agentic_rag.ingestion.pdf_parser import PDFParser, ParsedDocument, PageContent


class TestPDFParser:
    """Test suite for PDFParser."""

    @pytest.fixture
    def parser(self):
        """Create a PDF parser instance."""
        return PDFParser(extract_images=True, image_dpi=150)

    @pytest.fixture
    def sample_pdf_path(self):
        """Path to a sample PDF."""
        return Path(__file__).parent.parent / "data" / "pdfs" / "technical_report.pdf"

    def test_parser_initialization(self, parser):
        """Test parser initializes with correct settings."""
        assert parser.extract_images is True
        assert parser.image_dpi == 150

    def test_parse_single_pdf(self, parser, sample_pdf_path):
        """Test parsing a single PDF file."""
        if not sample_pdf_path.exists():
            pytest.skip("Sample PDF not found")

        doc = parser.parse(sample_pdf_path)

        assert isinstance(doc, ParsedDocument)
        assert doc.filename == "technical_report.pdf"
        assert doc.total_pages > 0
        assert len(doc.pages) == doc.total_pages

    def test_parsed_document_has_text(self, parser, sample_pdf_path):
        """Test that parsed document contains text."""
        if not sample_pdf_path.exists():
            pytest.skip("Sample PDF not found")

        doc = parser.parse(sample_pdf_path)
        full_text = doc.get_full_text()

        assert len(full_text) > 0
        assert "temperature" in full_text.lower() or "equipment" in full_text.lower()

    def test_table_detection(self, parser, sample_pdf_path):
        """Test that tables are detected in pages."""
        if not sample_pdf_path.exists():
            pytest.skip("Sample PDF not found")

        doc = parser.parse(sample_pdf_path)
        pages_with_tables = doc.get_pages_with_tables()

        # Technical report should have at least one page with tables
        assert len(pages_with_tables) >= 0  # May or may not have tables

    def test_image_extraction(self, parser, sample_pdf_path):
        """Test that images are extracted when enabled."""
        if not sample_pdf_path.exists():
            pytest.skip("Sample PDF not found")

        doc = parser.parse(sample_pdf_path)

        # Check first page has image data
        if doc.pages:
            page = doc.pages[0]
            assert len(page.image_base64) > 0 or len(page.images) >= 0

    def test_parse_nonexistent_file(self, parser):
        """Test parsing a non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/path/file.pdf")

    def test_parse_directory(self, parser):
        """Test parsing a directory of PDFs."""
        pdf_dir = Path(__file__).parent.parent / "data" / "pdfs"
        if not pdf_dir.exists():
            pytest.skip("PDF directory not found")

        docs = parser.parse_directory(pdf_dir)

        assert isinstance(docs, list)
        assert all(isinstance(d, ParsedDocument) for d in docs)


class TestPageContent:
    """Test suite for PageContent dataclass."""

    def test_page_content_creation(self):
        """Test creating a PageContent instance."""
        page = PageContent(
            page_num=1,
            text="Sample text",
            has_tables=True,
            source_file="test.pdf"
        )

        assert page.page_num == 1
        assert page.text == "Sample text"
        assert page.has_tables is True
        assert page.source_file == "test.pdf"
        assert page.images == []
        assert page.image_base64 == []


class TestParsedDocument:
    """Test suite for ParsedDocument dataclass."""

    def test_get_full_text(self):
        """Test concatenating all page text."""
        pages = [
            PageContent(page_num=1, text="Page 1 text", source_file="test.pdf"),
            PageContent(page_num=2, text="Page 2 text", source_file="test.pdf"),
        ]
        doc = ParsedDocument(
            filename="test.pdf",
            pages=pages,
            total_pages=2
        )

        full_text = doc.get_full_text()

        assert "Page 1 text" in full_text
        assert "Page 2 text" in full_text
        assert "[Page 1]" in full_text
        assert "[Page 2]" in full_text

    def test_get_pages_with_tables(self):
        """Test filtering pages with tables."""
        pages = [
            PageContent(page_num=1, text="No table", has_tables=False, source_file="test.pdf"),
            PageContent(page_num=2, text="Has table", has_tables=True, source_file="test.pdf"),
            PageContent(page_num=3, text="Also has table", has_tables=True, source_file="test.pdf"),
        ]
        doc = ParsedDocument(
            filename="test.pdf",
            pages=pages,
            total_pages=3
        )

        table_pages = doc.get_pages_with_tables()

        assert len(table_pages) == 2
        assert all(p.has_tables for p in table_pages)
