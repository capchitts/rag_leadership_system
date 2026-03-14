# generation/insight_generator.py

from __future__ import annotations

from typing import Any, Dict, List

from config import settings
from llm.llm_client import get_llm_client, load_llm_config_from_settings
from generation.context_builder import build_context
from generation.context_packer import pack_context
from prompts.executive_report_prompt import build_executive_report_prompt


def generate_insight(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Generates an executive-style report from retrieved RAG chunks.

    Expected flow:
    1. Normalize retrieved chunks into prompt-ready context items
    2. Pack context within token/character budget
    3. Build executive report prompts
    4. Load configured LLM provider from settings
    5. Generate final leadership insight
    """

    if not query or not query.strip():
        return "Query is empty. Unable to generate executive insight."

    if not retrieved_chunks:
        return "No relevant evidence was retrieved, so no executive report could be generated."

    context_items = build_context(retrieved_chunks)
    packed_context = pack_context(retrieved_chunks)

    system_prompt, user_prompt = build_executive_report_prompt(
        query=query.strip(),
        packed_context=packed_context,
    )

    llm_config = load_llm_config_from_settings(settings)
    llm_client = get_llm_client(llm_config)

    try:
        response = llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )
    except Exception as exc:
        return f"LLM generation failed: {exc}"

    if not response or not str(response).strip():
        return "LLM returned an empty response."

    return str(response).strip()