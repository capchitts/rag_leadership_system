from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, Optional

from pymongo import MongoClient

from config import settings


def _safe_number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(value):
            return None
        return float(value)
    return None


def _get_nested(doc: Dict[str, Any], path: str, default=None):
    current = doc
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _avg(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return round(numerator / denominator, 3)


def _extract_stage_latency(trace: Dict[str, Any], stage_name: str) -> Optional[float]:
    events = trace.get("events", [])
    for event in events:
        if event.get("stage") == stage_name:
            return _safe_number(event.get("duration_ms"))
    return None


def aggregate_rag_runs(
    mongo_uri: str,
    db_name: str,
    collection_name: str,
) -> Dict[str, Any]:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    collection = client[db_name][collection_name]

    docs = list(collection.find({}))
    total_runs = len(docs)

    if total_runs == 0:
        return {
            "total_runs": 0,
            "message": "No run documents found in MongoDB.",
        }

    faithfulness_scores = []
    groundedness_scores = []

    hallucination_true = 0

    labeled_runs = 0
    retrieval_failures = 0

    precision_values = []
    recall_values = []
    mrr_values = []
    ndcg_values = []

    retrieval_latencies = []
    reranking_latencies = []
    context_packing_latencies = []
    generation_latencies = []
    evaluation_latencies = []

    llm_model_counter = Counter()
    embedding_model_counter = Counter()
    query_type_counter = Counter()
    faithfulness_distribution = Counter()
    groundedness_distribution = Counter()

    for doc in docs:
        llm_model = doc.get("llm_model")
        embedding_model = doc.get("embedding_model")
        query_type = _get_nested(doc, "query_plan.query_type")

        if llm_model:
            llm_model_counter[llm_model] += 1
        if embedding_model:
            embedding_model_counter[embedding_model] += 1
        if query_type:
            query_type_counter[query_type] += 1

        faithfulness_score = _safe_number(_get_nested(doc, "evaluation.faithfulness.faithfulness_score"))
        groundedness_score = _safe_number(_get_nested(doc, "evaluation.groundedness.groundedness_score"))

        if faithfulness_score is not None:
            faithfulness_scores.append(faithfulness_score)
            faithfulness_distribution[int(faithfulness_score)] += 1

        if groundedness_score is not None:
            groundedness_scores.append(groundedness_score)
            groundedness_distribution[int(groundedness_score)] += 1

        hallucination_detected = _get_nested(doc, "evaluation.diagnostics.hallucination_detected")
        if hallucination_detected is True:
            hallucination_true += 1

        precision_at_k = _safe_number(_get_nested(doc, "evaluation.retrieval.precision_at_k"))
        recall_at_k = _safe_number(_get_nested(doc, "evaluation.retrieval.recall_at_k"))
        mrr = _safe_number(_get_nested(doc, "evaluation.retrieval.mrr"))
        ndcg_at_k = _safe_number(_get_nested(doc, "evaluation.retrieval.ndcg_at_k"))

        has_any_retrieval_label = any(
            metric is not None for metric in [precision_at_k, recall_at_k, mrr, ndcg_at_k]
        )

        if has_any_retrieval_label:
            labeled_runs += 1

        if precision_at_k is not None:
            precision_values.append(precision_at_k)
        if recall_at_k is not None:
            recall_values.append(recall_at_k)
        if mrr is not None:
            mrr_values.append(mrr)
        if ndcg_at_k is not None:
            ndcg_values.append(ndcg_at_k)

        retrieval_failure = _get_nested(doc, "evaluation.diagnostics.retrieval_failure")
        if retrieval_failure is True:
            retrieval_failures += 1

        trace = doc.get("trace", {})
        retrieval_ms = _extract_stage_latency(trace, "retrieval")
        reranking_ms = _extract_stage_latency(trace, "reranking")
        context_packing_ms = _extract_stage_latency(trace, "context_packing")
        generation_ms = _extract_stage_latency(trace, "generation")
        evaluation_ms = _extract_stage_latency(trace, "evaluation")

        if retrieval_ms is not None:
            retrieval_latencies.append(retrieval_ms)
        if reranking_ms is not None:
            reranking_latencies.append(reranking_ms)
        if context_packing_ms is not None:
            context_packing_latencies.append(context_packing_ms)
        if generation_ms is not None:
            generation_latencies.append(generation_ms)
        if evaluation_ms is not None:
            evaluation_latencies.append(evaluation_ms)

    result = {
        "run_summary": {
            "total_runs": total_runs,
            "labeled_runs": labeled_runs,
            "unlabeled_runs": total_runs - labeled_runs,
        },
        "quality_metrics": {
            "avg_faithfulness_score": _avg(faithfulness_scores),
            "avg_groundedness_score": _avg(groundedness_scores),
            "hallucination_rate": _rate(hallucination_true, total_runs),
        },
        "retrieval_metrics": {
            "avg_precision_at_k": _avg(precision_values),
            "avg_recall_at_k": _avg(recall_values),
            "avg_mrr": _avg(mrr_values),
            "avg_ndcg_at_k": _avg(ndcg_values),
            "retrieval_failure_rate": _rate(retrieval_failures, labeled_runs),
        },
        "latency_metrics_ms": {
            "avg_retrieval_ms": _avg(retrieval_latencies),
            "avg_reranking_ms": _avg(reranking_latencies),
            "avg_context_packing_ms": _avg(context_packing_latencies),
            "avg_generation_ms": _avg(generation_latencies),
            "avg_evaluation_ms": _avg(evaluation_latencies),
        },
        "distributions": {
            "faithfulness_score_distribution": dict(sorted(faithfulness_distribution.items())),
            "groundedness_score_distribution": dict(sorted(groundedness_distribution.items())),
        },
        "breakdowns": {
            "llm_models": dict(llm_model_counter),
            "embedding_models": dict(embedding_model_counter),
            "query_types": dict(query_type_counter),
        },
    }

    return result


def main() -> None:
    mongo_uri = settings.MONGO_URI
    db_name = settings.MONGO_DB_NAME
    collection_name = settings.MONGO_COLLECTION_NAME

    result = aggregate_rag_runs(
        mongo_uri=mongo_uri,
        db_name=db_name,
        collection_name=collection_name,
    )

    print("\n===== RAG OBSERVABILITY SUMMARY =====\n")
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()