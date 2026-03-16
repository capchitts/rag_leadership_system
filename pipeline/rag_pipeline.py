# pipeline/rag_pipeline.py
from pathlib import Path

from config import settings
from ingestion.pdf_loader import load_pdf_documents
from processing.hierarchical_chunker import hierarchical_chunk_documents
from embedding.embedder import Embedder
from indexing.vector_index import VectorIndex
from indexing.bm25_index import BM25Index
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.query_analyzer import QueryAnalyzer
from retrieval.query_expander import LLMQueryExpander
from llm.llm_client import get_llm_client, load_llm_config_from_settings

from utils.logger import get_logger
from observability.tracer import PipelineTracer

logger = get_logger(__name__)


def build_pipeline(data_folder):
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

    faiss_index_path = settings.FAISS_INDEX_PATH
    faiss_store_path = settings.FAISS_CHUNK_STORE_PATH

    if Path(faiss_index_path).exists() and Path(faiss_store_path).exists():
        with tracer.measure("vector_index_load") as t:
            vector_index = VectorIndex.load(faiss_index_path, faiss_store_path)
            t.set_payload(index_size=len(vector_index))
        logger.info(
            "Loaded persisted FAISS index",
            extra={"extra_data": {"index_path": faiss_index_path, "chunk_store_path": faiss_store_path, "index_size": len(vector_index)}}
        )
    else:
        with tracer.measure("embedding") as t:
            embeddings = embedder.embed_chunks(chunks)
            t.set_payload(embedding_count=len(embeddings))
        logger.info("Embeddings created", extra={"extra_data": {"count": len(embeddings), "model": settings.EMBEDDING_MODEL}})

        dim = len(embeddings[0])

        with tracer.measure("vector_index_build") as t:
            vector_index = VectorIndex(dim)
            vector_index.add_embeddings(embeddings, chunks)
            vector_index.save(faiss_index_path, faiss_store_path)
            t.set_payload(index_size=len(vector_index))
        logger.info(
            "Built and saved FAISS index",
            extra={"extra_data": {"index_path": faiss_index_path, "chunk_store_path": faiss_store_path, "index_size": len(vector_index)}}
        )

    with tracer.measure("bm25_index_build") as t:
        bm25_index = BM25Index(chunks)
        t.set_payload(chunk_count=len(chunks))
    logger.info("BM25 index built", extra={"extra_data": {"chunk_count": len(chunks)}})

    query_analyzer = QueryAnalyzer()

    llm_config = load_llm_config_from_settings(settings)
    llm_client = get_llm_client(llm_config)
    query_expander = LLMQueryExpander(llm_client)

    retriever = HybridRetriever(
        vector_index=vector_index,
        bm25_index=bm25_index,
        embedder=embedder,
        query_analyzer=query_analyzer,
        query_expander=query_expander if settings.ENABLE_QUERY_EXPANSION else None,
        vector_top_k=settings.VECTOR_TOP_K,
        bm25_top_k=settings.BM25_TOP_K,
        final_retriever_k=settings.FINAL_RETRIEVAL_K,
    )

    reranker = Reranker()

    logger.info(
        "Pipeline ready",
        extra={
            "extra_data": {
                "vector_top_k": settings.VECTOR_TOP_K,
                "bm25_top_k": settings.BM25_TOP_K,
                "final_retrieval_k": settings.FINAL_RETRIEVAL_K,
                "rerank_top_k": settings.RERANK_TOP_K,
                "query_expansion_enabled": settings.ENABLE_QUERY_EXPANSION,
            }
        },
    )

    return retriever, reranker, tracer