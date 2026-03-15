from sentence_transformers import CrossEncoder
from config import settings

class Reranker:

    def __init__(self):

        self.model = CrossEncoder(settings.RERANKER_MODEL)

    def rerank(self, query, chunks, top_k=settings.RERANK_TOP_K):

        pairs = []

        for chunk in chunks:
            pairs.append([query, chunk.text])

        scores = self.model.predict(pairs)

        scored_chunks = []

        for chunk, score in zip(chunks, scores):

            scored_chunks.append({
                "chunk": chunk,
                "score": score
            })

        ranked = sorted(
            scored_chunks,
            key=lambda x: x["score"],
            reverse=True
        )

        top_chunks = [item["chunk"] for item in ranked[:top_k]]

        return top_chunks