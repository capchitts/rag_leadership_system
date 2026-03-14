# generation/context_packer.py

from __future__ import annotations

from collections import defaultdict
from typing import List, Dict, Tuple

from core.schemas import Chunk


def pack_context(
    chunks: List[Chunk],
    max_chars: int = 3500,
    max_chunks_per_section: int = 2,
) -> str:
    """
    Convert retrieved Chunk schema objects into structured prompt-ready context.

    Features:
    - schema-safe attribute access
    - groups by section
    - attaches citation labels
    - limits context size
    - skips empty chunks
    """

    section_map: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    used_chars = 0
    structured_context = ""

    for chunk in chunks:
        text = chunk.text.strip()
        if not text:
            continue

        metadata = chunk.metadata
        section = metadata.section or "general"
        source = metadata.source or metadata.source_file or "unknown"
        page = metadata.page if metadata.page is not None else "unknown"
        chunk_id = chunk.chunk_id or metadata.chunk_id or "unknown_chunk"

        citation_label = f"[{chunk_id} | {source} | p.{page} | {section}]"
        section_map[section].append((citation_label, text))

    for section, items in section_map.items():
        section_header = f"\n=== {section.upper()} ===\n"

        if used_chars + len(section_header) > max_chars:
            break

        structured_context += section_header
        used_chars += len(section_header)

        for citation_label, text in items[:max_chunks_per_section]:
            bullet = f"- {citation_label}\n{text}\n\n"

            if used_chars + len(bullet) > max_chars:
                return structured_context.strip()

            structured_context += bullet
            used_chars += len(bullet)

    return structured_context.strip()