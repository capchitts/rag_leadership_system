from __future__ import annotations

import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from core.schemas import Chunk


class BM25Index:
    """
    BM25 index over Chunk objects.

    Returns ranked results as:
        List[Tuple[Chunk, float]]
    """

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.corpus = [self._tokenize(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    def search(self, query: str, k: int = 10) -> List[Tuple[Chunk, float]]:
        if not self.bm25 or not query.strip():
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        results: List[Tuple[Chunk, float]] = []
        for idx in ranked_indices:
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(scores[idx])))

        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Lightweight production-safe tokenizer.
        Keeps words and numbers, lowercased.
        """
        return re.findall(r"\b\w+\b", text.lower())