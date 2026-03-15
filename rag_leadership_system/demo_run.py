from __future__ import annotations

from config import settings
from evaluation.evaluator import RAGEvaluator
from pipeline.rag_pipeline import build_pipeline
from generation.context_packer import pack_context
from generation.insight_generator import generate_insight
from prompts.executive_report_prompt import build_executive_report_prompt
from llm.llm_client import get_llm_client, load_llm_config_from_settings



def run_demo() -> None:
    print("\nBuilding RAG pipeline...\n")

    retriever, reranker = build_pipeline("data/reports")

    print("Pipeline ready.")

    while True:
        query = input("\nEnter your query (or type 'exit'): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Query cannot be empty.")
            continue

        print("\nRetrieving documents...\n")

        candidates = retriever.retrieve(query)
        print(f"Retrieved {len(candidates)} candidate chunks.")

        if not candidates:
            print("No chunks retrieved.")
            continue

        top_chunks = reranker.rerank(query, candidates,top_k=settings.RERANK_TOP_K)
        print(f"Reranked to {len(top_chunks)} final chunks.")

        if not top_chunks:
            print("No chunks left after reranking.")
            continue

        packed_context = pack_context(top_chunks)

        system_prompt, user_prompt = build_executive_report_prompt(
            query=query.strip(),
            packed_context=packed_context,
        )

        final_report = generate_insight(query, top_chunks)

        print("\n===== PACKED CONTEXT =====\n")
        print(packed_context)

        print("\n===== RETRIEVED CHUNKS =====\n")
        for idx, chunk in enumerate(top_chunks, start=1):
            metadata = chunk.metadata
            print(f"[Chunk {idx}]")
            print(f"Source : {metadata.source or 'unknown'}")
            print(f"Page   : {metadata.page or 'unknown'}")
            print(f"Section: {metadata.section or 'general'}")
            print(f"Text   : {chunk.text[:800]}")
            print("-" * 80)

        print("\n===== PACKED CONTEXT =====\n")
        print(packed_context)

        print("\n===== SYSTEM PROMPT =====\n")
        print(system_prompt)

        print("\n===== USER PROMPT =====\n")
        print(user_prompt)

        print("\n===== EXECUTIVE REPORT =====\n")
        print(final_report)

        print("\n===== EVALUATION =====\n")
        try:
            llm_config = load_llm_config_from_settings(settings)
            llm_client = get_llm_client(llm_config)
            rag_evaluator = RAGEvaluator(llm_client)

            evaluation_result = rag_evaluator.evaluate(
                query=query,
                retrieved_chunks=top_chunks,
                answer=final_report,
                context=packed_context,
                relevant_chunk_ids=None,
            )
            
            print(evaluation_result)

        except Exception as exc:
            print(f"Evaluation skipped due to error: {exc}")

        print("\nDemo cycle complete.\n")


if __name__ == "__main__":
    run_demo()