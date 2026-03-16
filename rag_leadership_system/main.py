import os
from pathlib import Path

from config import settings
from evaluation.evaluator import RAGEvaluator
from generation.context_packer import pack_context
from generation.insight_generator import generate_insight
from llm.llm_client import get_llm_client, load_llm_config_from_settings
from pipeline.query_pipeline import build_query_pipeline
from utils.logger import get_logger

logger = get_logger(__name__)


def print_banner() -> None:
    print("=" * 80)
    print("RAG Leadership System")
    print("=" * 80)


def ensure_artifacts_exist() -> None:
    required = [
        settings.FAISS_INDEX_PATH,
        settings.FAISS_CHUNK_STORE_PATH,
        settings.BM25_INDEX_PATH,
    ]
    missing = [p for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing persisted artifacts. Run ingestion pipeline first.\nMissing:\n- " + "\n- ".join(missing)
        )


def run_query_flow(query: str, rerank_top_k: int) -> None:
    print("\n[1/4] Loading query pipeline...")
    retriever, reranker, build_tracer = build_query_pipeline()
    logger.info("Query pipeline loaded", extra={"extra_data": build_tracer.to_dict()})

    print("[2/4] Retrieving candidate chunks...")
    retrieval_debug = retriever.retrieve_with_debug(query)
    retrieved_chunks = retrieval_debug["results"]

    if not retrieved_chunks:
        print("No relevant chunks were retrieved.")
        return

    print(f"Retrieved {len(retrieved_chunks)} chunks.")

    print("[3/4] Reranking retrieved chunks...")
    final_chunks = reranker.rerank(query, retrieved_chunks, top_k=rerank_top_k)

    if not final_chunks:
        print("No chunks remained after reranking.")
        return

    print(f"Selected top {len(final_chunks)} chunks after reranking.")

    packed_context = pack_context(final_chunks)
    answer = generate_insight(query, final_chunks)

    print("\n" + "=" * 80)
    print("EXECUTIVE REPORT")
    print("=" * 80)
    print(answer)

    print("\n[4/4] Running lightweight evaluation...")
    try:
        llm_config = load_llm_config_from_settings(settings)
        llm_client = get_llm_client(llm_config)
        rag_evaluator = RAGEvaluator(llm_client)

        evaluation_result = rag_evaluator.evaluate(
            query=query,
            retrieved_chunks=final_chunks,
            answer=answer,
            context=packed_context,
            relevant_chunk_ids=None,
            graded_relevance=None,
            retrieval_k=rerank_top_k,
        )

        print("\n" + "=" * 80)
        print("EVALUATION")
        print("=" * 80)
        print(evaluation_result)
    except Exception as exc:
        logger.exception("Evaluation failed")
        print(f"Evaluation skipped due to error: {exc}")


def main() -> None:
    print_banner()
    ensure_artifacts_exist()

    print("\nEnter your leadership/business question.")
    print("Example: What are the key strategic risks and growth opportunities in this report?")
    query = input("Query: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    run_query_flow(
        query=query,
        rerank_top_k=settings.RERANK_TOP_K,
    )


if __name__ == "__main__":
    main()