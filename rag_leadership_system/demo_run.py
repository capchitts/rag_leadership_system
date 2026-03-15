from __future__ import annotations

import json

from config import settings
from evaluation.evaluator import RAGEvaluator
from pipeline.rag_pipeline import build_pipeline
from generation.context_packer import pack_context
from generation.insight_generator import generate_insight
from prompts.executive_report_prompt import build_executive_report_prompt
from llm.llm_client import get_llm_client, load_llm_config_from_settings

from utils.logger import get_logger
from observability.tracer import PipelineTracer
from storage.mongo_logger import MongoRunLogger

logger = get_logger(__name__)


def run_demo() -> None:
    print("\nBuilding RAG pipeline...\n")

    retriever, reranker, build_tracer = build_pipeline(settings.DATA_DIR)

    print("Pipeline ready.")
    logger.info("Pipeline built", extra={"extra_data": build_tracer.to_dict()})

    mongo_logger = None
    try:
        mongo_logger = MongoRunLogger()
        logger.info("Mongo logger initialized")
    except Exception as exc:
        logger.warning("Mongo logger unavailable", extra={"extra_data": {"error": str(exc)}})

    while True:
        query = input("\nEnter your query (or type 'exit'): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Query cannot be empty.")
            continue

        query_tracer = PipelineTracer()

        logger.info("Query received", extra={"extra_data": {"query": query}})

        print("\nRetrieving documents...\n")

        with query_tracer.measure("retrieval") as t:
            retrieval_debug = retriever.retrieve_with_debug(query)
            candidates = retrieval_debug["results"]
            t.set_payload(candidate_count=len(candidates), query_variants=retrieval_debug["query_variants"])

        print(f"Retrieved {len(candidates)} candidate chunks.")

        if not candidates:
            print("No chunks retrieved.")
            logger.info("No chunks retrieved", extra={"extra_data": {"query": query}})
            continue

        with query_tracer.measure("reranking") as t:
            top_chunks = reranker.rerank(query, candidates, top_k=settings.RERANK_TOP_K)
            t.set_payload(final_count=len(top_chunks))

        print(f"Reranked to {len(top_chunks)} final chunks.")

        if not top_chunks:
            print("No chunks left after reranking.")
            logger.info("No chunks after reranking", extra={"extra_data": {"query": query}})
            continue

        with query_tracer.measure("context_packing") as t:
            packed_context = pack_context(top_chunks)
            t.set_payload(context_chars=len(packed_context))

        system_prompt, user_prompt = build_executive_report_prompt(
            query=query.strip(),
            packed_context=packed_context,
        )

        with query_tracer.measure("generation") as t:
            final_report = generate_insight(query, top_chunks)
            t.set_payload(answer_chars=len(final_report))

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

        evaluation_result = None
        relevant_chunk_ids = None
        graded_relevance = None

        print("\n===== EVALUATION =====\n")
        try:
            llm_config = load_llm_config_from_settings(settings)
            llm_client = get_llm_client(llm_config)
            rag_evaluator = RAGEvaluator(llm_client)

            with query_tracer.measure("evaluation") as t:
                evaluation_result = rag_evaluator.evaluate(
                    query=query,
                    retrieved_chunks=top_chunks,
                    answer=final_report,
                    context=packed_context,
                    relevant_chunk_ids=relevant_chunk_ids,
                    graded_relevance=graded_relevance,
                    retrieval_k=settings.RERANK_TOP_K,
                )
                t.set_payload(
                    faithfulness_score=evaluation_result.faithfulness.faithfulness_score,
                    groundedness_score=evaluation_result.groundedness.groundedness_score,
                )

            print(evaluation_result)

        except Exception as exc:
            print(f"Evaluation skipped due to error: {exc}")
            logger.exception("Evaluation failed")

        run_payload = {
            "query": query,
            "query_variants": retrieval_debug["query_variants"],
            "query_plan": (
                retrieval_debug["query_plan"].model_dump()
                if hasattr(retrieval_debug["query_plan"], "model_dump")
                else retrieval_debug["query_plan"]
            ),
            "retrieved_chunk_ids": [chunk.chunk_id for chunk in candidates],
            "reranked_chunk_ids": [chunk.chunk_id for chunk in top_chunks],
            "retrieved_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.metadata.source,
                    "page": chunk.metadata.page,
                    "section": chunk.metadata.section,
                }
                for chunk in candidates
            ],
            "reranked_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.metadata.source,
                    "page": chunk.metadata.page,
                    "section": chunk.metadata.section,
                }
                for chunk in top_chunks
            ],
            "packed_context_chars": len(packed_context),
            "answer_chars": len(final_report),
            "answer": final_report,
            "prompts": {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "formatted_context_chars": len(packed_context),
            },
            "trace": query_tracer.to_dict(),
            "llm_provider": settings.LLM_PROVIDER,
            "llm_model": settings.LLM_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "ground_truth": {
                "relevant_chunk_ids": list(relevant_chunk_ids) if relevant_chunk_ids else [],
                "graded_relevance": graded_relevance or {},
            },
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "estimated_cost_usd": None,
            },
        }

        if evaluation_result is not None:
            evaluation_dict = (
                evaluation_result.model_dump()
                if hasattr(evaluation_result, "model_dump")
                else evaluation_result
            )

            faithfulness = evaluation_result.faithfulness
            groundedness = evaluation_result.groundedness

            evaluation_dict["quality_summary"] = {
                "faithfulness_score": faithfulness.faithfulness_score,
                "groundedness_score": groundedness.groundedness_score,
                "unsupported_claim_count": len(faithfulness.unsupported_claims),
                "partially_supported_claim_count": len(faithfulness.partially_supported_claims),
                "unsupported_statement_count": len(groundedness.unsupported_statements),
                "hallucination_detected": (
                    evaluation_result.diagnostics.hallucination_detected
                    if evaluation_result.diagnostics is not None
                    else (
                        len(faithfulness.unsupported_claims) > 0
                        or len(groundedness.unsupported_statements) > 0
                    )
                ),
            }

            run_payload["evaluation"] = evaluation_dict

        logger.info("Query run complete", extra={"extra_data": run_payload})

        if mongo_logger is not None:
            try:
                mongo_logger.log_run(run_payload)
            except Exception as exc:
                logger.warning("Mongo logging failed", extra={"extra_data": {"error": str(exc)}})

        print("\nDemo cycle complete.\n")


if __name__ == "__main__":
    run_demo()