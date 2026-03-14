from sentence_transformers import SentenceTransformer
from embedding.embedding_utils import build_embedding_text


class Embedder:

    def __init__(self):

        self.model = SentenceTransformer("BAAI/bge-large-en")

    def embed_chunks(self, chunks):

        texts = []

        for chunk in chunks:

            enriched_text = build_embedding_text(chunk)

            texts.append(enriched_text)

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return embeddings
    
    def embed_query(self, query):
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        return query_embedding