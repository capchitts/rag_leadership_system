import faiss
import numpy as np


class VectorIndex:

    def __init__(self, dimension):

        self.index = faiss.IndexFlatIP(dimension)

        self.chunk_store = []

    def add_embeddings(self, embeddings, chunks):

        vectors = np.array(embeddings).astype("float32")

        self.index.add(vectors)

        self.chunk_store.extend(chunks)

    def search(self, query_embedding, k=10):

        query_vector = np.array([query_embedding]).astype("float32")

        scores, indices = self.index.search(query_vector, k)

        results = []

        for idx in indices[0]:

            if idx < len(self.chunk_store):

                results.append(self.chunk_store[idx])

        return results