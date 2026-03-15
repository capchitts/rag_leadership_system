# indexing/vector_index.py
from __future__ import annotations

from typing import List, Tuple
import pickle
from pathlib import Path

import faiss
import numpy as np

from core.schemas import Chunk


class VectorIndex:
    """
    FAISS vector index using inner product search.
    With L2-normalized embeddings, inner product approximates cosine similarity.

    Returns ranked results as:
        List[Tuple[Chunk, float]]
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunk_store: List[Chunk] = []

    def add_embeddings(self, embeddings, chunks: List[Chunk]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks must have the same length.")

        vectors = np.asarray(embeddings, dtype="float32")

        if vectors.ndim != 2:
            raise ValueError(f"Embeddings must be 2D. Got shape {vectors.shape}")

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.dimension}, got {vectors.shape[1]}"
            )

        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.chunk_store.extend(chunks)

    def search(self, query_embedding, k: int = 10) -> List[Tuple[Chunk, float]]:
        if self.index.ntotal == 0:
            return []

        query_vector = np.asarray([query_embedding], dtype="float32")

        if query_vector.ndim != 2 or query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch. Expected ({1}, {self.dimension}), got {query_vector.shape}"
            )

        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, k)

        results: List[Tuple[Chunk, float]] = []

        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            if idx >= len(self.chunk_store):
                continue

            results.append((self.chunk_store[idx], float(score)))

        return results

    def save(self, index_path: str, store_path: str) -> None:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(store_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, index_path)

        with open(store_path, "wb") as f:
            pickle.dump(self.chunk_store, f)

    @classmethod
    def load(cls, index_path: str, store_path: str):
        index = faiss.read_index(index_path)

        with open(store_path, "rb") as f:
            chunk_store = pickle.load(f)

        obj = cls(index.d)
        obj.index = index
        obj.chunk_store = chunk_store
        return obj

    def __len__(self) -> int:
        return self.index.ntotal