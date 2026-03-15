from __future__ import annotations

from typing import Optional, Dict, Set

from evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    mrr_at_k,
    ndcg_at_k,
    has_relevant_in_top_k,
    retrieval_failure_at_k,
)
from evaluation.faithfulness import evaluate_faithfulness
from evaluation.groundedness import evaluate_groundedness
from core.schemas import (
    EvaluationResult,
    EvaluationDiagnosticsResult,
    FaithfulnessResult,
    GroundednessResult,
    RetrievalMetricsResult,
    Chunk,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGEvaluator:
    def __init__(self, llm):
        self.llm = llm

    def evaluate(
        self,
        query: str,
        retrieved_chunks: list[Chunk],
        answer: str,
        context: str,
        relevant_chunk_ids: Optional[Set[str]] = None,
        graded_relevance: Optional[Dict[str, int]] = None,
        retrieval_k: int = 5,
    ) -> EvaluationResult:
        logger.info(
            "RAG evaluation started",
            extra={
                "extra_data": {
                    "query": query,
                    "retrieved_count": len(retrieved_chunks),
                    "retrieval_k": retrieval_k,
                    "has_labels": bool(relevant_chunk_ids),
                    "has_graded_labels": bool(graded_relevance),
                }
            },
        )

        precision_value = precision_at_k(retrieved_chunks, relevant_chunk_ids, retrieval_k)
        recall_value = recall_at_k(retrieved_chunks, relevant_chunk_ids, retrieval_k)
        mrr_value = mrr_at_k(retrieved_chunks, relevant_chunk_ids, retrieval_k)
        ndcg_value = ndcg_at_k(retrieved_chunks, graded_relevance, retrieval_k)

        retrieval_result = RetrievalMetricsResult(
            precision_at_k=precision_value,
            recall_at_k=recall_value,
            mrr=mrr_value,
            ndcg_at_k=ndcg_value,
        )

        faithfulness_result = evaluate_faithfulness(self.llm, answer, context)
        groundedness_result = evaluate_groundedness(self.llm, answer, context)

        if isinstance(faithfulness_result, dict):
            faithfulness_result = FaithfulnessResult.model_validate(faithfulness_result)

        if isinstance(groundedness_result, dict):
            groundedness_result = GroundednessResult.model_validate(groundedness_result)

        diagnostics = EvaluationDiagnosticsResult(
            has_relevant_in_top_k=has_relevant_in_top_k(
                retrieved_chunks,
                relevant_chunk_ids,
                retrieval_k,
            ),
            retrieval_failure=retrieval_failure_at_k(
                retrieved_chunks,
                relevant_chunk_ids,
                retrieval_k,
            ),
            unsupported_claim_count=len(faithfulness_result.unsupported_claims),
            partially_supported_claim_count=len(faithfulness_result.partially_supported_claims),
            unsupported_statement_count=len(groundedness_result.unsupported_statements),
            hallucination_detected=(
                len(faithfulness_result.unsupported_claims) > 0
                or len(groundedness_result.unsupported_statements) > 0
            ),
        )

        logger.info(
            "RAG evaluation completed",
            extra={
                "extra_data": {
                    "query": query,
                    "precision_at_k": precision_value,
                    "recall_at_k": recall_value,
                    "mrr": mrr_value,
                    "ndcg_at_k": ndcg_value,
                    "faithfulness_score": faithfulness_result.faithfulness_score,
                    "groundedness_score": groundedness_result.groundedness_score,
                    "diagnostics": diagnostics.model_dump(),
                }
            },
        )

        return EvaluationResult(
            query=query,
            retrieval=retrieval_result,
            faithfulness=faithfulness_result,
            groundedness=groundedness_result,
            diagnostics=diagnostics,
        )