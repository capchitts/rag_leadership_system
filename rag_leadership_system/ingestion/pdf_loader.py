import os
import uuid
from datetime import datetime

from ingestion.text_extractor import extract_text
from ingestion.table_extractor import extract_tables
from core.schemas import DocumentUnit, ChunkMetadata


def is_valid_text(text: str, min_chars: int = 30) -> bool:
    if not text:
        return False
    cleaned = text.strip()
    return len(cleaned) >= min_chars


def load_pdf_documents(folder: str):
    documents = []

    for file in os.listdir(folder):
        if not file.endswith(".pdf"):
            continue

        path = os.path.join(folder, file)
        doc_id = str(uuid.uuid4())
        ingested_at = datetime.utcnow()

        text_pages = extract_text(path)
        tables = extract_tables(path)

        unit_index = 0

        for page_num, text in text_pages:
            if not is_valid_text(text):
                continue

            metadata = ChunkMetadata(
                doc_id=doc_id,
                source=file,
                source_file=file,
                file_path=path,
                doc_type="pdf",
                chunk_id=None,
                chunk_index=unit_index,
                page=page_num,
                section="general",
                type="text",
                content_type="text",
                char_length=len(text),
                word_count=len(text.split()),
                extraction_method="pymupdf",
                extraction_confidence=1.0,
                ingested_at=ingested_at,
            )

            documents.append(
                DocumentUnit(
                    text=text.strip(),
                    metadata=metadata,
                )
            )
            unit_index += 1

        for table in tables:
            table_text = table.get("text", "").strip()
            if not is_valid_text(table_text):
                continue

            table_meta = table.get("metadata", {})

            metadata = ChunkMetadata(
                doc_id=doc_id,
                source=file,
                source_file=file,
                file_path=path,
                doc_type="pdf",
                chunk_id=None,
                chunk_index=unit_index,
                page=table_meta.get("page"),
                section="table",
                type="table",
                content_type="table",
                table_index=table_meta.get("table_index"),
                row_count=table_meta.get("row_count"),
                column_count=table_meta.get("column_count"),
                headers=table_meta.get("headers"),
                char_length=len(table_text),
                word_count=len(table_text.split()),
                extraction_method=table_meta.get("extraction_method", "pdfplumber"),
                extraction_confidence=1.0,
                ingested_at=ingested_at,
                extra={
                    "rows": table.get("rows", [])
                }
            )

            documents.append(
                DocumentUnit(
                    text=table_text,
                    metadata=metadata,
                )
            )
            unit_index += 1

    return documents