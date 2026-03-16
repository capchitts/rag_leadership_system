import re
from typing import List

from core.schemas import Chunk, DocumentUnit, make_chunk_id
from processing.metadata_extractor import detect_section


def split_into_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_paragraphs(paragraphs: List[str], max_chars: int = 800, overlap_paragraphs: int = 1) -> List[str]:
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        if current and current_len + para_len + 2 > max_chars:
            chunks.append("\n\n".join(current).strip())

            overlap = current[-overlap_paragraphs:] if overlap_paragraphs > 0 else []
            current = overlap[:]
            current_len = sum(len(p) for p in current) + (2 * len(current))

        current.append(para)
        current_len += para_len + 2

    if current:
        chunks.append("\n\n".join(current).strip())

    return [c for c in chunks if c.strip()]


def chunk_table_text(table_text: str, max_chars: int = 1200, overlap_rows: int = 1) -> List[str]:
    rows = [r.strip() for r in table_text.split("\n") if r.strip()]

    chunks = []
    current = []
    current_len = 0

    for row in rows:
        row_len = len(row)

        if current and current_len + row_len + 1 > max_chars:
            chunks.append("\n".join(current).strip())

            overlap = current[-overlap_rows:] if overlap_rows > 0 else []
            current = overlap[:]
            current_len = sum(len(r) + 1 for r in current)

        current.append(row)
        current_len += row_len + 1

    if current:
        chunks.append("\n".join(current).strip())

    return [c for c in chunks if c.strip()]


def hierarchical_chunk_documents(
    documents: List[DocumentUnit],
    text_max_chars: int = 800,
    text_overlap_paragraphs: int = 1,
    table_max_chars: int = 1200,
    table_overlap_rows: int = 1,
) -> List[Chunk]:
    chunks: List[Chunk] = []

    for doc in documents:
        text = doc.text.strip()
        if not text:
            continue

        metadata = doc.metadata
        doc_type = metadata.content_type or metadata.type

        if doc_type == "table":
            section = metadata.section or "table"
            pieces = chunk_table_text(
                text,
                max_chars=table_max_chars,
                overlap_rows=table_overlap_rows,
            )
        else:
            section = detect_section(text)
            paragraphs = split_into_paragraphs(text)
            pieces = chunk_paragraphs(
                paragraphs,
                max_chars=text_max_chars,
                overlap_paragraphs=text_overlap_paragraphs,
            )

        for idx, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue

            chunk_id = make_chunk_id(metadata.source, metadata.page, idx)

            chunk_metadata = metadata.model_copy(deep=True)
            chunk_metadata.chunk_id = chunk_id
            chunk_metadata.chunk_index = idx
            chunk_metadata.section = section
            chunk_metadata.char_length = len(piece)
            chunk_metadata.word_count = len(re.findall(r"\w+", piece))

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=piece,
                    metadata=chunk_metadata,
                )
            )

    return chunks