from __future__ import annotations

import math
from typing import List, Optional, Set, Dict

from core.schemas import Chunk


NO_LABELS_MSG = "N/A (no labeled relevant_chunk_ids provided)"


def _top_k_chunk_ids(retrieved: List[Chunk], k: int) -> List[str]:
    return [chunk.chunk_id for chunk in retrieved[:k]]


def precision_at_k(
    retrieved: List[Chunk],
    relevant_ids: Optional[Set[str]],
    k: int,
) -> float | str:
    """
    Precision@K = (# relevant retrieved in top K) / K
    """

    if not relevant_ids:
        return NO_LABELS_MSG

    if k <= 0:
        return 0.0

    retrieved_k = _top_k_chunk_ids(retrieved, k)
    relevant_retrieved = sum(1 for chunk_id in retrieved_k if chunk_id in relevant_ids)

    return round(relevant_retrieved / k, 3)


def recall_at_k(
    retrieved: List[Chunk],
    relevant_ids: Optional[Set[str]],
    k: int,
) -> float | str:
    """
    Recall@K = (# relevant retrieved in top K) / (total relevant)
    """

    if not relevant_ids:
        return NO_LABELS_MSG

    if len(relevant_ids) == 0:
        return 0.0

    retrieved_k = _top_k_chunk_ids(retrieved, k)
    relevant_retrieved = sum(1 for chunk_id in retrieved_k if chunk_id in relevant_ids)

    return round(relevant_retrieved / len(relevant_ids), 3)


def mrr_at_k(
    retrieved: List[Chunk],
    relevant_ids: Optional[Set[str]],
    k: int,
) -> float | None:
    """
    Reciprocal rank of first relevant result in top K.
    """

    if not relevant_ids:
        return None

    retrieved_k = _top_k_chunk_ids(retrieved, k)

    for rank, chunk_id in enumerate(retrieved_k, start=1):
        if chunk_id in relevant_ids:
            return round(1.0 / rank, 3)

    return 0.0


def dcg_at_k(
    retrieved: List[Chunk],
    graded_relevance: Optional[Dict[str, int]],
    k: int,
) -> float | None:
    """
    DCG@K using graded relevance.
    """

    if not graded_relevance:
        return None

    retrieved_k = _top_k_chunk_ids(retrieved, k)

    dcg = 0.0
    for rank, chunk_id in enumerate(retrieved_k, start=1):
        rel = graded_relevance.get(chunk_id, 0)
        if rel > 0:
            dcg += rel / math.log2(rank + 1)

    return round(dcg, 3)


def ndcg_at_k(
    retrieved: List[Chunk],
    graded_relevance: Optional[Dict[str, int]],
    k: int,
) -> float | None:
    """
    nDCG@K = DCG@K / IDCG@K
    """

    if not graded_relevance:
        return None

    actual_dcg = dcg_at_k(retrieved, graded_relevance, k)
    if actual_dcg is None:
        return None

    ideal_relevances = sorted(graded_relevance.values(), reverse=True)[:k]

    ideal_dcg = 0.0
    for rank, rel in enumerate(ideal_relevances, start=1):
        if rel > 0:
            ideal_dcg += rel / math.log2(rank + 1)

    if ideal_dcg == 0:
        return 0.0

    return round(actual_dcg / ideal_dcg, 3)


def has_relevant_in_top_k(
    retrieved: List[Chunk],
    relevant_ids: Optional[Set[str]],
    k: int,
) -> bool | None:
    """
    True if at least one relevant chunk appears in top K.
    """

    if not relevant_ids:
        return None

    retrieved_k = _top_k_chunk_ids(retrieved, k)
    return any(chunk_id in relevant_ids for chunk_id in retrieved_k)


def retrieval_failure_at_k(
    retrieved: List[Chunk],
    relevant_ids: Optional[Set[str]],
    k: int,
) -> bool | None:
    """
    Retrieval failure means no relevant chunk appears in top K.
    """

    found = has_relevant_in_top_k(retrieved, relevant_ids, k)
    if found is None:
        return None
    return not found