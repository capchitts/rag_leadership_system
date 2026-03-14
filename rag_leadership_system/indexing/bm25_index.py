from rank_bm25 import BM25Okapi


class BM25Index:

    def __init__(self, chunks):

        self.chunks = chunks

        self.corpus = [chunk["text"].split() for chunk in chunks]

        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query, k=10):

        query_tokens = query.split()

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for idx in ranked_indices[:k]:
            results.append(self.chunks[idx])

        return results