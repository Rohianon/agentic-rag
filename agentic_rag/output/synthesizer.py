"""Output synthesis for structured JSON generation with citations."""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

from openai import OpenAI

from agentic_rag.agent.reasoning import ReasoningOutput
from agentic_rag.retrieval.hybrid_retriever import RetrievalResult


@dataclass
class StructuredOutput:
    """Final structured output from the pipeline."""

    query: str
    summary: str
    key_findings: list[str]
    extracted_data: dict[str, Any]
    risk_flags: list[dict]
    citations: list[dict]
    confidence_score: float
    metadata: dict

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), indent=indent, default=str)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class OutputSynthesizer:
    """
    Synthesizes final structured output from reasoning results.

    Produces:
    1. Executive summary
    2. Key findings with citations
    3. Extracted numerical data
    4. Risk flags and warnings
    5. Full citation trail
    """

    SYNTHESIS_PROMPT = """Based on the analysis below, create an executive summary and key findings.

Original Query: {query}

Analysis Result:
{answer}

Extracted Values:
{extracted_values}

Risk Flags:
{risk_flags}

Generate:
1. A concise executive summary (2-3 sentences)
2. 3-5 key findings as bullet points
3. Any domain-specific insights

Respond in JSON format:
{{
    "summary": "Executive summary here",
    "key_findings": ["Finding 1", "Finding 2", ...],
    "domain_insights": ["Insight 1", ...]
}}"""

    def __init__(
        self,
        openai_client: OpenAI | None = None,
        model: str = "gpt-4o-mini",
    ):
        self.client = openai_client or OpenAI()
        self.model = model

    def synthesize(
        self,
        query: str,
        reasoning_output: ReasoningOutput,
        retrieval_result: RetrievalResult | None = None,
    ) -> StructuredOutput:
        """
        Synthesize final structured output.

        Args:
            query: Original user query
            reasoning_output: Output from reasoning agent
            retrieval_result: Optional retrieval details for metadata

        Returns:
            StructuredOutput with all components
        """
        # Generate summary and findings
        synthesis = self._generate_synthesis(query, reasoning_output)

        # Build citation trail
        citations = self._build_citations(reasoning_output.citations)

        # Compile metadata
        metadata = self._build_metadata(reasoning_output, retrieval_result)

        return StructuredOutput(
            query=query,
            summary=synthesis.get("summary", reasoning_output.answer[:200]),
            key_findings=synthesis.get("key_findings", []),
            extracted_data=reasoning_output.extracted_values,
            risk_flags=reasoning_output.risk_flags,
            citations=citations,
            confidence_score=reasoning_output.confidence,
            metadata=metadata,
        )

    def _generate_synthesis(
        self,
        query: str,
        reasoning_output: ReasoningOutput,
    ) -> dict:
        """Generate executive summary and key findings."""
        prompt = self.SYNTHESIS_PROMPT.format(
            query=query,
            answer=reasoning_output.answer,
            extracted_values=json.dumps(reasoning_output.extracted_values, indent=2),
            risk_flags=json.dumps(reasoning_output.risk_flags, indent=2),
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0,
            response_format={"type": "json_object"},
        )

        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {
                "summary": reasoning_output.answer[:200],
                "key_findings": [],
            }

    def _build_citations(self, citations: list[dict]) -> list[dict]:
        """Build standardized citation format."""
        standardized = []

        for i, citation in enumerate(citations, 1):
            standardized.append({
                "id": i,
                "source_file": citation.get("file", "unknown"),
                "page": citation.get("page", 0),
                "chunk_type": citation.get("chunk_type", "text"),
                "relevance": citation.get("relevance_score", 0),
                "excerpt": citation.get("quote", "")[:200],
            })

        return standardized

    def _build_metadata(
        self,
        reasoning_output: ReasoningOutput,
        retrieval_result: RetrievalResult | None,
    ) -> dict:
        """Build output metadata."""
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "model": self.model,
            "reasoning_steps": len(reasoning_output.reasoning_trace),
        }

        if retrieval_result:
            metadata.update({
                "chunks_retrieved": len(retrieval_result.chunks),
                "chunks_filtered": retrieval_result.filtered_count,
                "sources_used": list({c.source_file for c in retrieval_result.chunks}),
            })

        return metadata

    def format_for_display(self, output: StructuredOutput) -> str:
        """Format output for human-readable display."""
        lines = [
            "=" * 60,
            "ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Query: {output.query}",
            "",
            "SUMMARY",
            "-" * 40,
            output.summary,
            "",
            "KEY FINDINGS",
            "-" * 40,
        ]

        for i, finding in enumerate(output.key_findings, 1):
            lines.append(f"  {i}. {finding}")

        if output.extracted_data:
            lines.extend([
                "",
                "EXTRACTED DATA",
                "-" * 40,
            ])
            for key, value in output.extracted_data.items():
                if isinstance(value, dict):
                    lines.append(f"  {key}: {value.get('value')} {value.get('unit', '')}")
                else:
                    lines.append(f"  {key}: {value}")

        if output.risk_flags:
            lines.extend([
                "",
                "⚠️  RISK FLAGS",
                "-" * 40,
            ])
            for flag in output.risk_flags:
                lines.append(f"  [{flag['severity'].upper()}] {flag['metric']}: {flag['message']}")

        lines.extend([
            "",
            "CITATIONS",
            "-" * 40,
        ])
        for citation in output.citations:
            lines.append(
                f"  [{citation['id']}] {citation['source_file']}, "
                f"Page {citation['page']}"
            )

        lines.extend([
            "",
            f"Confidence: {output.confidence_score:.0%}",
            f"Generated: {output.metadata.get('generated_at', 'N/A')}",
            "=" * 60,
        ])

        return "\n".join(lines)
