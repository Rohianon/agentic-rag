"""PDF parsing module for extracting text, images, and detecting tables."""

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import fitz  # pymupdf


@dataclass
class PageContent:
    """Represents extracted content from a single PDF page."""

    page_num: int
    text: str
    images: list[bytes] = field(default_factory=list)
    image_base64: list[str] = field(default_factory=list)
    has_tables: bool = False
    source_file: str = ""


@dataclass
class ParsedDocument:
    """Represents a fully parsed PDF document."""

    filename: str
    pages: list[PageContent]
    total_pages: int
    metadata: dict = field(default_factory=dict)

    def get_full_text(self) -> str:
        """Concatenate all page text."""
        return "\n\n".join(f"[Page {p.page_num}]\n{p.text}" for p in self.pages)

    def get_pages_with_tables(self) -> list[PageContent]:
        """Return pages that likely contain tables."""
        return [p for p in self.pages if p.has_tables]


class PDFParser:
    """
    Extracts text, images, and detects table regions from PDF documents.

    Uses PyMuPDF (fitz) for efficient extraction without OCR dependency.
    Table detection uses heuristics (grid lines, consistent spacing).
    """

    def __init__(self, extract_images: bool = True, image_dpi: int = 150):
        self.extract_images = extract_images
        self.image_dpi = image_dpi

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        """Parse a PDF file and extract all content."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        pages = list(self._extract_pages(doc, pdf_path.name))

        metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "creator": doc.metadata.get("creator", ""),
        }

        doc.close()

        return ParsedDocument(
            filename=pdf_path.name,
            pages=pages,
            total_pages=len(pages),
            metadata=metadata,
        )

    def _extract_pages(
        self, doc: fitz.Document, filename: str
    ) -> Iterator[PageContent]:
        """Extract content from each page."""
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")

            # Detect if page likely has tables (heuristic)
            has_tables = self._detect_tables(page, text)

            images = []
            images_b64 = []

            if self.extract_images:
                images, images_b64 = self._extract_page_images(page)

            yield PageContent(
                page_num=page_num,
                text=text.strip(),
                images=images,
                image_base64=images_b64,
                has_tables=has_tables,
                source_file=filename,
            )

    def _detect_tables(self, page: fitz.Page, text: str) -> bool:
        """
        Heuristic table detection based on:
        1. Presence of drawing paths (grid lines)
        2. Text alignment patterns
        3. Consistent column spacing
        """
        # Check for vector drawings (table borders)
        drawings = page.get_drawings()
        has_grid_lines = len(drawings) > 10  # Threshold for table-like structure

        # Check for tab-separated or consistently spaced content
        lines = text.split("\n")
        tabular_lines = sum(1 for line in lines if "\t" in line or "  " in line)
        has_tabular_text = tabular_lines > 3

        # Check for numeric columns (common in tables)
        numeric_pattern_count = sum(
            1 for line in lines
            if sum(c.isdigit() for c in line) > len(line) * 0.3
        )
        has_numeric_data = numeric_pattern_count > 2

        return has_grid_lines or (has_tabular_text and has_numeric_data)

    def _extract_page_images(
        self, page: fitz.Page
    ) -> tuple[list[bytes], list[str]]:
        """Extract images from a page and convert to base64."""
        images = []
        images_b64 = []

        # Render page as image (useful for vision models)
        mat = fitz.Matrix(self.image_dpi / 72, self.image_dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

        images.append(img_bytes)
        images_b64.append(base64.b64encode(img_bytes).decode("utf-8"))

        return images, images_b64

    def parse_directory(self, dir_path: str | Path) -> list[ParsedDocument]:
        """Parse all PDFs in a directory."""
        dir_path = Path(dir_path)
        pdfs = list(dir_path.glob("*.pdf"))
        return [self.parse(pdf) for pdf in pdfs]
