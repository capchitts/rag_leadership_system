import uuid
from processing.metadata_extractor import detect_section

def chunk_text(text, chunk_size=600, overlap=120):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


def hierarchical_chunk_documents(documents):

    chunks = []

    for doc in documents:

        section = detect_section(doc["text"])

        pieces = chunk_text(doc["text"])

        for idx, p in enumerate(pieces):

            chunks.append({

                "chunk_id": str(uuid.uuid4()),

                "text": p,

                "metadata": {
                    **doc["metadata"],
                    "section": section,
                    "chunk_index": idx
                }
            })

    return chunks