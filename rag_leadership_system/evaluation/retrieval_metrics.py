def precision_at_k(retrieved, relevant_ids, k):

    retrieved = retrieved[:k]

    rel = sum(
        1 for c in retrieved
        if c["chunk_id"] in relevant_ids
    )

    return rel / k


def recall_at_k(retrieved, relevant_ids, k):

    retrieved = retrieved[:k]

    rel = sum(
        1 for c in retrieved
        if c["chunk_id"] in relevant_ids
    )

    return rel / len(relevant_ids)