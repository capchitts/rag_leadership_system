from evaluation.retrieval_metrics import precision_at_k, recall_at_k
from evaluation.faithfulness import evaluate_faithfulness
from evaluation.groundedness import evaluate_groundedness
from core.schemas import (
    EvaluationResult,
    FaithfulnessResult,
    GroundednessResult,
    RetrievalMetricsResult,
    Chunk,
)


class RAGEvaluator:
    def __init__(self, llm):
        self.llm = llm

    def evaluate(
        self,
        query: str,
        retrieved_chunks: list[Chunk],
        answer: str,
        context: str,
        relevant_chunk_ids=None,
    ) -> EvaluationResult:

        if relevant_chunk_ids:
            precision_value = precision_at_k(retrieved_chunks, relevant_chunk_ids, 5)
            recall_value = recall_at_k(retrieved_chunks, relevant_chunk_ids, 5)
        else:
            precision_value = "N/A (no labeled relevant_chunk_ids provided)"
            recall_value = "N/A (no labeled relevant_chunk_ids provided)"

        retrieval_result = RetrievalMetricsResult(
            precision_at_5=precision_value,
            recall_at_5=recall_value,
            mrr=None,
            ndcg_at_5=None,
        )

        faithfulness_result = evaluate_faithfulness(self.llm, answer, context)
        groundedness_result = evaluate_groundedness(self.llm, answer, context)

        if isinstance(faithfulness_result, dict):
            faithfulness_result = FaithfulnessResult.model_validate(faithfulness_result)

        if isinstance(groundedness_result, dict):
            groundedness_result = GroundednessResult.model_validate(groundedness_result)

        return EvaluationResult(
            query=query,
            retrieval=retrieval_result,
            faithfulness=faithfulness_result,
            groundedness=groundedness_result,
        )