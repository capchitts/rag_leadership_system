from typing import Any, Dict, List
from config import settings
from core.schemas import Chunk, QueryPlan, RetrievalCandidate, RetrievalScores
from utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """
    Hybrid retriever using:
    - query analyzer
    - optional query expansion
    - vector retrieval
    - BM25 retrieval
    - Reciprocal Rank Fusion (RRF)
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
        final_retriever_k,
        rrf_k: int | None = None,
    ):
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.embedder = embedder
        self.query_analyzer = query_analyzer
        self.query_expander = query_expander if settings.ENABLE_QUERY_EXPANSION else None
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.final_retriever_k = final_retriever_k
        self.rrf_k = rrf_k if rrf_k is not None else settings.RRF_K

    def retrieve(self, query: str) -> List[Chunk]:
        result = self.retrieve_with_debug(query=query)
        return result["results"]

    def retrieve_with_debug(self, query: str):

        logger.info("Hybrid retrieval started", extra={"extra_data": {"query": query}})

        analysis = self.query_analyzer.analyze(query)

        logger.info(
            "Query analyzed",
            extra={
                "extra_data": {
                    "query": query,
                    "analysis": analysis.model_dump() if hasattr(analysis, "model_dump") else analysis.__dict__,
                }
            },
        )

        query_variants = [query]

        if analysis.should_expand and self.query_expander is not None:
            query_variants = self.query_expander.expand(
                query,
                max_expansions=analysis.max_expansions
            )

        logger.info(
            "Query expansion complete",
            extra={"extra_data": {"query_variants": query_variants}},
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

            logger.info(
                "Raw retrieval results",
                extra={
                    "extra_data": {
                        "query_variant": q,
                        "vector_results": len(vector_results),
                        "bm25_results": len(bm25_results),
                    }
                },
            )

            ranked_vector = self._ranked_results(vector_results, source="vector")
            ranked_bm25 = self._ranked_results(bm25_results, source="bm25")

            self._accumulate_rrf_scores(
                aggregated=aggregated,
                ranked_results=ranked_vector,
                score_field="vector_score",
            )

            self._accumulate_rrf_scores(
                aggregated=aggregated,
                ranked_results=ranked_bm25,
                score_field="bm25_score",
            )

            trace.append({
                "query_variant": q,
                "vector_results_count": len(ranked_vector),
                "bm25_results_count": len(ranked_bm25),
            })

        ranked_candidates = sorted(
            aggregated.values(),
            key=lambda x: x.scores.hybrid_score or 0.0,
            reverse=True
        )

        final_candidates = ranked_candidates[:self.final_retriever_k]
        final_chunks = [candidate.chunk for candidate in final_candidates]

        logger.info(
            "Hybrid retrieval completed",
            extra={
                "extra_data": {
                    "query": query,
                    "candidate_count": len(ranked_candidates),
                    "final_chunks": [c.chunk.chunk_id for c in final_candidates],
                }
            },
        )

        return {
            "query_plan": QueryPlan.model_validate(
                analysis.model_dump() if hasattr(analysis, "model_dump") else analysis.__dict__
            ),
            "query_variants": query_variants,
            "trace": trace,
            "scored_results": final_candidates,
            "results": final_chunks,
        }

    def _ranked_results(self, results: List[Any], source: str) -> List[dict]:
        ranked = []

        for rank, item in enumerate(results, start=1):
            if isinstance(item, tuple) and len(item) >= 2:
                chunk, raw_score = item[0], item[1]
            elif isinstance(item, dict):
                chunk = item["chunk"]
                raw_score = item.get("score", 0.0)
            else:
                raise ValueError(
                    f"Unsupported result format from {source} retriever: {type(item)}"
                )

            ranked.append({
                "chunk": chunk,
                "rank": rank,
                "raw_score": float(raw_score) if raw_score is not None else 0.0,
                "source": source,
            })

        return ranked

    def _accumulate_rrf_scores(
        self,
        aggregated: Dict[str, RetrievalCandidate],
        ranked_results: List[dict],
        score_field: str,
    ) -> None:
        for item in ranked_results:
            chunk = item["chunk"]
            rank = item["rank"]
            raw_score = item["raw_score"]

            chunk_id = chunk.chunk_id

            if chunk_id not in aggregated:
                aggregated[chunk_id] = RetrievalCandidate(
                    chunk=chunk,
                    scores=RetrievalScores(
                        vector_score=None,
                        bm25_score=None,
                        hybrid_score=0.0,
                    ),
                )

            candidate = aggregated[chunk_id]

            setattr(candidate.scores, score_field, raw_score)

            rrf_score = 1.0 / (self.rrf_k + rank)
            candidate.scores.hybrid_score = (candidate.scores.hybrid_score or 0.0) + rrf_score