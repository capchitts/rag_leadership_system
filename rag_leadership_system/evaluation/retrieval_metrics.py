from core.schemas import Chunk
from typing import List, Optional, Set


def precision_at_k(
    retrieved: List[Chunk],
    relevant_ids: Optional[Set[str]],
    k: int,
) -> float | str:
    """
    Precision@K = (# relevant retrieved in top K) / K
    """

    if not relevant_ids:
        return "N/A (no labeled relevant_chunk_ids provided)"

    retrieved_k = retrieved[:k]

    relevant_retrieved = sum(
        1 for chunk in retrieved_k
        if chunk.chunk_id in relevant_ids
    )

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
        return "N/A (no labeled relevant_chunk_ids provided)"

    if len(relevant_ids) == 0:
        return 0.0

    retrieved_k = retrieved[:k]

    relevant_retrieved = sum(
        1 for chunk in retrieved_k
        if chunk.chunk_id in relevant_ids
    )

    return round(relevant_retrieved / len(relevant_ids), 3)