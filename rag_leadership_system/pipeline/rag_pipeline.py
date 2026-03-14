from ingestion.pdf_loader import load_pdf_documents
from processing.hierarchical_chunker import hierarchical_chunk_documents
from embedding.embedder import Embedder
from indexing.vector_index import VectorIndex
from indexing.bm25_index import BM25Index
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker

def build_pipeline(data_folder):

    docs = load_pdf_documents(data_folder)

    chunks = hierarchical_chunk_documents(docs)

    embedder = Embedder()

    embeddings = embedder.embed_chunks(chunks)

    dim = len(embeddings[0])

    vector_index = VectorIndex(dim)

    vector_index.add_embeddings(embeddings, chunks)

    bm25_index = BM25Index(chunks)

    retriever = HybridRetriever(
        vector_index,
        bm25_index,
        embedder
    )

    reranker = Reranker()

    return retriever, reranker