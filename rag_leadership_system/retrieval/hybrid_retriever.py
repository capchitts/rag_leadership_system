class HybridRetriever:

    def __init__(self, vector_index, bm25_index, embedder):

        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.embedder = embedder

    def retrieve(self, query, k=10):

        query_emb = self.embedder.embed_query(query)

        v_results = self.vector_index.search(query_emb, k)

        bm_results = self.bm25_index.search(query, k)

        merged = {}

        for c in v_results + bm_results:
            merged[c["chunk_id"]] = c

        return list(merged.values())[:k]