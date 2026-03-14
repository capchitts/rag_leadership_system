from evaluation.retrieval_metrics import precision_at_k, recall_at_k
from evaluation.faithfulness import evaluate_faithfulness
from evaluation.groundedness import evaluate_groundedness


class RAGEvaluator:
    def __init__(self, llm):
        self.llm = llm

    def evaluate(
        self,
        query,
        retrieved_chunks,
        answer,
        context,
        relevant_chunk_ids=None,
    ):
        results = {}

        if relevant_chunk_ids is not None:
            results["precision@5"] = precision_at_k(retrieved_chunks, relevant_chunk_ids, 5)
            results["recall@5"] = recall_at_k(retrieved_chunks, relevant_chunk_ids, 5)
        else:
            results["precision@5"] = "N/A (no labeled relevant_chunk_ids provided)"
            results["recall@5"] = "N/A (no labeled relevant_chunk_ids provided)"

        results["faithfulness"] = evaluate_faithfulness(self.llm, answer, context)
        results["groundedness"] = evaluate_groundedness(self.llm, answer, context)

        return results