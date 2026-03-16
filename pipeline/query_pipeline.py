from pathlib import Path

from config import settings
from embedding.embedder import Embedder
from indexing.bm25_index import BM25Index
from indexing.vector_index import VectorIndex
from llm.llm_client import get_llm_client, load_llm_config_from_settings
from observability.tracer import PipelineTracer
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.query_analyzer import QueryAnalyzer
from retrieval.query_expander import LLMQueryExpander
from retrieval.reranker import Reranker
from utils.logger import get_logger

logger = get_logger(__name__)


def build_query_pipeline():
    tracer = PipelineTracer()

    faiss_index_path = settings.FAISS_INDEX_PATH
    faiss_store_path = settings.FAISS_CHUNK_STORE_PATH
    bm25_index_path = settings.BM25_INDEX_PATH

    if not Path(faiss_index_path).exists() or not Path(faiss_store_path).exists():
        raise FileNotFoundError(
            f"FAISS artifacts not found. Run ingestion first. Missing: {faiss_index_path} or {faiss_store_path}"
        )

    if not Path(bm25_index_path).exists():
        raise FileNotFoundError(
            f"BM25 artifact not found. Run ingestion first. Missing: {bm25_index_path}"
        )

    embedder = Embedder()

    with tracer.measure("vector_index_load") as t:
        vector_index = VectorIndex.load(faiss_index_path, faiss_store_path)
        t.set_payload(index_size=len(vector_index))
    logger.info(
        "Loaded FAISS index",
        extra={
            "extra_data": {
                "index_path": faiss_index_path,
                "chunk_store_path": faiss_store_path,
                "index_size": len(vector_index),
            }
        },
    )

    with tracer.measure("bm25_index_load") as t:
        bm25_index = BM25Index.load(bm25_index_path)
        t.set_payload(chunk_count=len(bm25_index.chunks))
    logger.info(
        "Loaded BM25 index",
        extra={
            "extra_data": {
                "bm25_index_path": bm25_index_path,
                "chunk_count": len(bm25_index.chunks),
            }
        },
    )

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
        "Query pipeline ready",
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