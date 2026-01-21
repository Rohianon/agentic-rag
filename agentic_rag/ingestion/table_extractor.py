"""Table extraction using Vision LLM (OpenAI GPT-4V or compatible)."""

import json
from dataclasses import dataclass

from openai import OpenAI

from agentic_rag.ingestion.pdf_parser import PageContent


@dataclass
class ExtractedTable:
    """Represents a table extracted from a document page."""

    page_num: int
    source_file: str
    table_json: dict | list
    table_summary: str
    headers: list[str]
    row_count: int
    confidence: float


class TableExtractor:
    """
    Uses Vision LLM to extract structured table data from page images.

    Converts visual tables to JSON with headers, rows, and a natural
    language summary for semantic search.
    """

    EXTRACTION_PROMPT = """Analyze this document page image and extract any tables present.

For each table found, provide:
1. A JSON representation with headers and rows
2. A brief natural language summary (1-2 sentences) describing what the table contains
3. Key numerical values or findings

If there are no tables, respond with {"tables": []}.

Respond in this exact JSON format:
{
    "tables": [
        {
            "headers": ["Column1", "Column2", ...],
            "rows": [["val1", "val2", ...], ...],
            "summary": "This table shows...",
            "key_values": {"metric_name": "value", ...}
        }
    ]
}

Be precise with numbers. Preserve units (%, $, °C, etc.)."""

    def __init__(self, client: OpenAI | None = None, model: str = "gpt-4o"):
        self.client = client or OpenAI()
        self.model = model

    def extract_tables(self, page: PageContent) -> list[ExtractedTable]:
        """Extract tables from a single page using vision model."""
        if not page.image_base64:
            return []

        # Use the page image for vision analysis
        image_b64 = page.image_base64[0]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.EXTRACTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=4096,
            response_format={"type": "json_object"},
        )

        try:
            result = json.loads(response.choices[0].message.content)
            tables = result.get("tables", [])
        except (json.JSONDecodeError, AttributeError):
            return []

        extracted = []
        for table_data in tables:
            headers = table_data.get("headers", [])
            rows = table_data.get("rows", [])

            extracted.append(
                ExtractedTable(
                    page_num=page.page_num,
                    source_file=page.source_file,
                    table_json={"headers": headers, "rows": rows},
                    table_summary=table_data.get("summary", ""),
                    headers=headers,
                    row_count=len(rows),
                    confidence=0.9 if headers and rows else 0.5,
                )
            )

        return extracted

    def extract_from_pages(
        self, pages: list[PageContent], only_table_pages: bool = True
    ) -> list[ExtractedTable]:
        """Extract tables from multiple pages."""
        all_tables = []

        for page in pages:
            # Optionally skip pages without detected tables
            if only_table_pages and not page.has_tables:
                continue

            tables = self.extract_tables(page)
            all_tables.extend(tables)

        return all_tables
