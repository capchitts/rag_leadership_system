from pathlib import Path
import pickle

from config import settings
from embedding.embedder import Embedder
from indexing.bm25_index import BM25Index
from indexing.vector_index import VectorIndex
from ingestion.pdf_loader import load_pdf_documents
from observability.tracer import PipelineTracer
from processing.hierarchical_chunker import hierarchical_chunk_documents
from utils.logger import get_logger

logger = get_logger(__name__)


def run_ingestion_pipeline(data_folder: str):
    tracer = PipelineTracer()

    with tracer.measure("document_loading") as t:
        docs = load_pdf_documents(data_folder)
        t.set_payload(document_count=len(docs))
    logger.info("Documents loaded", extra={"extra_data": {"count": len(docs), "data_folder": data_folder}})

    with tracer.measure("chunking") as t:
        chunks = hierarchical_chunk_documents(docs)
        t.set_payload(chunk_count=len(chunks))
    logger.info("Chunks created", extra={"extra_data": {"count": len(chunks)}})

    embedder = Embedder()

    with tracer.measure("embedding") as t:
        embeddings = embedder.embed_chunks(chunks)
        t.set_payload(embedding_count=len(embeddings))
    logger.info("Embeddings created", extra={"extra_data": {"count": len(embeddings), "model": settings.EMBEDDING_MODEL}})

    dim = len(embeddings[0])

    with tracer.measure("vector_index_build") as t:
        vector_index = VectorIndex(dim)
        vector_index.add_embeddings(embeddings, chunks)
        vector_index.save(settings.FAISS_INDEX_PATH, settings.FAISS_CHUNK_STORE_PATH)
        t.set_payload(index_size=len(vector_index))
    logger.info(
        "Built and saved FAISS index",
        extra={
            "extra_data": {
                "index_path": settings.FAISS_INDEX_PATH,
                "chunk_store_path": settings.FAISS_CHUNK_STORE_PATH,
                "index_size": len(vector_index),
            }
        },
    )

    with tracer.measure("bm25_index_build") as t:
        bm25_index = BM25Index(chunks)
        bm25_index.save(settings.BM25_INDEX_PATH)
        t.set_payload(chunk_count=len(chunks))
    logger.info(
        "Built and saved BM25 index",
        extra={
            "extra_data": {
                "bm25_index_path": settings.BM25_INDEX_PATH,
                "chunk_count": len(chunks),
            }
        },
    )

    logger.info("Ingestion pipeline completed", extra={"extra_data": tracer.to_dict()})
    return tracer