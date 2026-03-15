import os
from pathlib import Path

from pipeline.rag_pipeline import build_pipeline
from generation.insight_generator import generate_insight
from evaluation.evaluator import RAGEvaluator
from config import settings
from llm.llm_client import get_llm_client, load_llm_config_from_settings
from config.settings import *

def print_banner() -> None:
    print("=" * 80)
    print("RAG Leadership System")
    print("=" * 80)


def validate_data_path(data_path: str) -> str:
    resolved = str(Path(data_path).resolve())
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Data path does not exist: {resolved}")
    return resolved


def run_query_flow(data_path: str, query: str, rerank_top_k: int) -> None:
    print("\n[1/5] Building retrieval pipeline...")
    retriever, reranker = build_pipeline(data_path)

    print("[2/5] Retrieving candidate chunks...")
    retrieved_chunks = retriever.retrieve(query)

    if not retrieved_chunks:
        print("No relevant chunks were retrieved.")
        return

    print(f"Retrieved {len(retrieved_chunks)} chunks.")

    print("[3/5] Reranking retrieved chunks...")
    final_chunks = reranker.rerank(query, retrieved_chunks, top_k=rerank_top_k)

    if not final_chunks:
        print("No chunks remained after reranking.")
        return

    print(f"Selected top {len(final_chunks)} chunks after reranking.")

    # Build packed context from final chunks for generation/evaluation
    packed_context = "\n\n".join(
            chunk.get("text") or chunk.get("content") or str(chunk)
            if isinstance(chunk, dict)
            else str(chunk)
            for chunk in final_chunks
    )   

    print("[4/5] Generating executive insight...")
    answer = generate_insight(query, final_chunks)

    print("\n" + "=" * 80)
    print("EXECUTIVE REPORT")
    print("=" * 80)
    print(answer)

    print("\n[5/5] Running lightweight evaluation...")
    try:
        llm_config = load_llm_config_from_settings(settings)
        llm_client = get_llm_client(llm_config)
        rag_evaluator = RAGEvaluator(llm_client)

        evaluation_result = rag_evaluator.evaluate(
            query=query,
            retrieved_chunks=retrieved_chunks,
            answer=answer,
            context=packed_context,
            relevant_chunk_ids=None,
        )
            
        print("\n" + "=" * 80)
        print("EVALUATION")
        print("=" * 80)
        print(evaluation_result)
    except Exception as exc:
        print(f"Evaluation skipped due to error: {exc}")


def main() -> None:
    print_banner()

    project_root = Path(__file__).resolve().parent
    default_data_path = project_root / DATA_DIR

    print(f"Default data path: {default_data_path}")
    user_data_path = input("Enter data folder path or press Enter to use default: ").strip()
    data_path = user_data_path if user_data_path else str(default_data_path)
    data_path = validate_data_path(data_path)

    print("\nEnter your leadership/business question.")
    print("Example: What are the key strategic risks and growth opportunities in this report?")
    query = input("Query: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    run_query_flow(data_path=data_path, query=query,top_k=settings.VECTOR_TOP_K,rerank_top_k=settings.RERANK_TOP_K)


if __name__ == "__main__":
    main()