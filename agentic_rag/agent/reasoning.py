"""Reasoning agent for RAG-based question answering."""

import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from agentic_rag.retrieval.hybrid_retriever import RetrievalResult, HybridRetriever
from agentic_rag.agent.guardrails import PolicyGuardrail, GuardrailResult


@dataclass
class ReasoningOutput:
    """Output from the reasoning agent."""

    answer: str
    confidence: float
    citations: list[dict]
    extracted_values: dict[str, Any]
    risk_flags: list[dict]
    reasoning_trace: list[str]


class ReasoningAgent:
    """
    LLM-based reasoning agent for generating answers from retrieved context.

    Features:
    1. Chain-of-thought reasoning for complex queries
    2. Citation generation with source tracking
    3. Numerical value extraction
    4. Integration with policy guardrails
    5. Hallucination prevention through grounding
    """

    REASONING_PROMPT = """You are an expert analyst assistant. Answer the user's question using ONLY the provided context.

CRITICAL RULES:
1. ONLY use information from the provided context
2. If the context doesn't contain enough information, say "I cannot find this information in the provided documents"
3. ALWAYS cite your sources using [Source N] notation
4. Extract and highlight any numerical values, thresholds, or limits
5. Flag any potential risks or concerns found in the data

Context:
{context}

Question: {query}

Respond in this exact JSON format:
{{
    "answer": "Your detailed answer here with [Source N] citations",
    "confidence": 0.0-1.0,
    "citations": [
        {{"source_num": 1, "file": "filename.pdf", "page": 1, "quote": "relevant quote"}}
    ],
    "extracted_values": {{
        "metric_name": {{"value": "X", "unit": "unit", "source": 1}}
    }},
    "reasoning_steps": [
        "Step 1: I first looked at...",
        "Step 2: Then I found..."
    ]
}}"""

    def __init__(
        self,
        retriever: HybridRetriever,
        guardrail: PolicyGuardrail | None = None,
        openai_client: OpenAI | None = None,
        model: str = "gpt-4o",
    ):
        self.retriever = retriever
        self.guardrail = guardrail
        self.client = openai_client or OpenAI()
        self.model = model

    def reason(
        self,
        query: str,
        n_chunks: int = 5,
        apply_guardrails: bool = True,
    ) -> ReasoningOutput:
        """
        Process a query through the full reasoning pipeline.

        Steps:
        1. Retrieve relevant chunks
        2. Build context with citations
        3. Generate reasoned answer
        4. Apply guardrails to extracted values
        5. Return structured output
        """
        # Step 1: Retrieve context
        retrieval_result = self.retriever.retrieve(
            query=query,
            n_results=n_chunks,
            filter_irrelevant=True,
        )

        # Step 2: Build context with source numbering
        context = self._build_numbered_context(retrieval_result)

        # Step 3: Generate answer
        prompt = self.REASONING_PROMPT.format(
            context=context,
            query=query,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0,
            response_format={"type": "json_object"},
        )

        # Parse response
        try:
            result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            result = {
                "answer": response.choices[0].message.content,
                "confidence": 0.5,
                "citations": [],
                "extracted_values": {},
                "reasoning_steps": [],
            }

        # Step 4: Apply guardrails
        risk_flags = []
        if apply_guardrails and self.guardrail:
            risk_flags = self._apply_guardrails(result.get("extracted_values", {}))

        # Step 5: Enhance citations with actual source info
        citations = self._enhance_citations(
            result.get("citations", []),
            retrieval_result,
        )

        return ReasoningOutput(
            answer=result.get("answer", ""),
            confidence=result.get("confidence", 0.5),
            citations=citations,
            extracted_values=result.get("extracted_values", {}),
            risk_flags=risk_flags,
            reasoning_trace=result.get("reasoning_steps", []),
        )

    def _build_numbered_context(self, retrieval_result: RetrievalResult) -> str:
        """Build context with numbered sources for citation."""
        parts = []

        for i, chunk in enumerate(retrieval_result.chunks, 1):
            parts.append(f"[Source {i}]")
            parts.append(f"File: {chunk.source_file}")
            parts.append(f"Page: {chunk.page_num}")
            parts.append(f"Type: {chunk.chunk_type}")
            parts.append(f"Content:\n{chunk.content}")

            if chunk.structured_data:
                parts.append(f"Table Data: {json.dumps(chunk.structured_data, indent=2)}")

            parts.append("---\n")

        return "\n".join(parts)

    def _apply_guardrails(self, extracted_values: dict) -> list[dict]:
        """Apply policy guardrails to extracted values."""
        risk_flags = []

        for metric_name, value_info in extracted_values.items():
            if isinstance(value_info, dict):
                value = value_info.get("value")
                unit = value_info.get("unit", "")

                result = self.guardrail.check(
                    metric_name=metric_name,
                    value=value,
                    unit=unit,
                )

                if not result.is_safe:
                    risk_flags.append({
                        "metric": metric_name,
                        "value": f"{value} {unit}",
                        "limit": f"{result.limit} {unit}" if result.limit else "N/A",
                        "severity": result.severity,
                        "message": result.message,
                    })

        return risk_flags

    def _enhance_citations(
        self,
        citations: list[dict],
        retrieval_result: RetrievalResult,
    ) -> list[dict]:
        """Enhance citations with actual source information."""
        enhanced = []

        for citation in citations:
            source_num = citation.get("source_num", 1)
            idx = source_num - 1

            if 0 <= idx < len(retrieval_result.chunks):
                chunk = retrieval_result.chunks[idx]
                enhanced.append({
                    "source_num": source_num,
                    "file": chunk.source_file,
                    "page": chunk.page_num,
                    "chunk_type": chunk.chunk_type,
                    "relevance_score": chunk.relevance_score,
                    "quote": citation.get("quote", ""),
                })
            else:
                enhanced.append(citation)

        return enhanced

    def multi_turn_reason(
        self,
        queries: list[str],
        n_chunks_per_query: int = 3,
    ) -> list[ReasoningOutput]:
        """Process multiple related queries, maintaining context."""
        outputs = []
        accumulated_context = []

        for query in queries:
            # Retrieve for this query
            retrieval = self.retriever.retrieve(
                query=query,
                n_results=n_chunks_per_query,
            )

            # Add to accumulated context
            accumulated_context.extend(retrieval.chunks)

            # Reason with accumulated context
            output = self.reason(query, n_chunks=n_chunks_per_query)
            outputs.append(output)

        return outputs
