# generation/insight_generator.py

from __future__ import annotations

import json
import re
from typing import Any, List

from config import settings
from llm.llm_client import get_llm_client, load_llm_config_from_settings
from generation.context_packer import pack_context
from prompts.executive_report_prompt import build_executive_report_prompt
from core.schemas import Chunk, ExecutiveReportResult


def _extract_json_text(raw_response: str) -> str:
    """
    Extract JSON from an LLM response that may contain markdown fences
    or a little extra surrounding text.
    """
    if not raw_response:
        raise ValueError("Empty LLM response")

    text = raw_response.strip()

    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No valid JSON object found in LLM response")

    return text[start:end + 1]


def _parse_executive_report(raw_response: str) -> ExecutiveReportResult:
    json_text = _extract_json_text(raw_response)
    parsed = json.loads(json_text)
    return ExecutiveReportResult.model_validate(parsed)


def render_executive_report(report: ExecutiveReportResult) -> str:
    """
    Convert structured executive report schema into readable markdown-style text.
    """
    lines: List[str] = []

    sections = [
        ("Executive Summary", report.executive_summary),
        ("Top Risks", report.top_risks),
        ("Top Opportunities", report.top_opportunities),
        ("Recommended Actions", report.recommended_actions),
        ("Supporting Evidence", report.supporting_evidence),
    ]

    for title, bullets in sections:
        if not bullets:
            continue

        lines.append(title)
        for bullet in bullets:
            citation_str = f" {' '.join(bullet.citations)}" if bullet.citations else ""
            lines.append(f"- {bullet.text}{citation_str}")
        lines.append("")

    return "\n".join(lines).strip()


def generate_insight(
    query: str,
    retrieved_chunks: List[Chunk],
    mode: str = "brief",
    return_structured: bool = False,
) -> str | ExecutiveReportResult:
    """
    Generate an executive-style report from retrieved RAG chunks.

    Args:
        query: user query
        retrieved_chunks: top retrieved Chunk schema objects
        mode: "brief" or "detailed"
        return_structured: if True, return ExecutiveReportResult instead of rendered string
    """

    if not query or not query.strip():
        empty_report = ExecutiveReportResult()
        return empty_report if return_structured else "Query is empty. Unable to generate executive insight."

    if not retrieved_chunks:
        empty_report = ExecutiveReportResult()
        return empty_report if return_structured else "No relevant evidence was retrieved, so no executive report could be generated."

    if mode == "brief":
        packed_context = pack_context(
            retrieved_chunks,
            max_chars=3500,
            max_chunks_per_section=2,
        )
    else:
        packed_context = pack_context(
            retrieved_chunks,
            max_chars=7000,
            max_chunks_per_section=4,
        )

    system_prompt, user_prompt = build_executive_report_prompt(
        query=query.strip(),
        packed_context=packed_context,
    )

    llm_config = load_llm_config_from_settings(settings)
    llm_client = get_llm_client(llm_config)

    try:
        raw_response = llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )
    except Exception as exc:
        empty_report = ExecutiveReportResult()
        if return_structured:
            return empty_report
        return f"LLM generation failed: {exc}"

    if not raw_response or not str(raw_response).strip():
        empty_report = ExecutiveReportResult()
        return empty_report if return_structured else "LLM returned an empty response."

    try:
        structured_report = _parse_executive_report(str(raw_response).strip())
    except Exception:
        # fallback: return raw response if JSON parsing fails
        if return_structured:
            return ExecutiveReportResult()
        return str(raw_response).strip()

    if return_structured:
        return structured_report

    return render_executive_report(structured_report)