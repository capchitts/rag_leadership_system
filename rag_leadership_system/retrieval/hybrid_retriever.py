from typing import Dict, List
from config import settings
from core.schemas import Chunk, QueryPlan, RetrievalCandidate, RetrievalScores


class HybridRetriever:
    """
    Hybrid retriever using:
    - query analyzer
    - optional query expansion
    - vector retrieval
    - BM25 retrieval
    - weighted score fusion
    """

    def __init__(
        self,
        vector_index,
        bm25_index,
        embedder,
        query_analyzer,
        query_expander,
        vector_top_k,
        bm25_top_k,
        final_retriever_k
    ):
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.embedder = embedder
        self.query_analyzer = query_analyzer
        self.query_expander = query_expander if settings.ENABLE_QUERY_EXPANSION else None
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.final_retriever_k = final_retriever_k

    def retrieve(self, query: str) -> List[Chunk]:
        result = self.retrieve_with_debug(query=query)
        return result["results"]

    def retrieve_with_debug(self, query: str):
        analysis = self.query_analyzer.analyze(query)

        query_variants = [query]
        if analysis.should_expand and self.query_expander is not None:
            query_variants = self.query_expander.expand(
                query,
                max_expansions=analysis.max_expansions
            )

        aggregated: Dict[str, RetrievalCandidate] = {}

        trace = []

        for q in query_variants:
            vector_results = []
            bm25_results = []

            if analysis.use_vector:
                query_emb = self.embedder.embed_query(q)
                vector_results = self.vector_index.search(query_emb, self.vector_top_k)

            if analysis.use_bm25:
                bm25_results = self.bm25_index.search(q, self.bm25_top_k)

            normalized_vector = self._normalize_results(vector_results, source="vector")
            normalized_bm25 = self._normalize_results(bm25_results, source="bm25")

            self._accumulate_scores(
                aggregated=aggregated,
                normalized_results=normalized_vector,
                weight=analysis.vector_weight,
                score_field="vector_score",
            )

            self._accumulate_scores(
                aggregated=aggregated,
                normalized_results=normalized_bm25,
                weight=analysis.bm25_weight,
                score_field="bm25_score",
            )

            trace.append({
                "query_variant": q,
                "vector_results_count": len(normalized_vector),
                "bm25_results_count": len(normalized_bm25),
            })

        ranked_candidates = sorted(
            aggregated.values(),
            key=lambda x: x.scores.hybrid_score or 0.0,
            reverse=True
        )

        final_chunks = [candidate.chunk for candidate in ranked_candidates[:self.final_retriever_k]]

        return {
            "query_plan": QueryPlan.model_validate(analysis.model_dump() if hasattr(analysis, "model_dump") else analysis.__dict__),
            "query_variants": query_variants,
            "trace": trace,
            "scored_results": ranked_candidates[:self.final_retriever_k],
            "results": final_chunks,
        }

    def _normalize_results(self, results, source: str):
        """
        Supports:
        - List[Chunk]
        - List[tuple[Chunk, score]]
        """
        parsed = []

        for rank, item in enumerate(results):
            if isinstance(item, tuple) and len(item) == 2:
                chunk, score = item
                parsed.append((chunk, float(score)))
            else:
                chunk = item
                parsed.append((chunk, 1.0 / (rank + 1)))

        if not parsed:
            return []

        raw_scores = [score for _, score in parsed]
        min_score = min(raw_scores)
        max_score = max(raw_scores)

        normalized = []
        for chunk, raw_score in parsed:
            if max_score == min_score:
                norm_score = 1.0
            else:
                norm_score = (raw_score - min_score) / (max_score - min_score)

            normalized.append({
                "chunk": chunk,
                "chunk_id": chunk.chunk_id,
                "source": source,
                "raw_score": raw_score,
                "norm_score": norm_score,
            })

        return normalized

    def _accumulate_scores(self, aggregated, normalized_results, weight: float, score_field: str):
        for item in normalized_results:
            chunk_id = item["chunk_id"]
            contribution = item["norm_score"] * weight

            if chunk_id not in aggregated:
                aggregated[chunk_id] = RetrievalCandidate(
                    chunk=item["chunk"],
                    scores=RetrievalScores(
                        bm25_score=0.0,
                        vector_score=0.0,
                        hybrid_score=0.0,
                        rerank_score=None,
                    ),
                    retrieved_by=[],
                )

            candidate = aggregated[chunk_id]

            current_val = getattr(candidate.scores, score_field) or 0.0
            setattr(candidate.scores, score_field, current_val + contribution)

            candidate.scores.hybrid_score = (candidate.scores.hybrid_score or 0.0) + contribution

            if item["source"] not in candidate.retrieved_by:
                candidate.retrieved_by.append(item["source"])