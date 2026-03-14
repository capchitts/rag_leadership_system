from config import settings
from ingestion.pdf_loader import load_pdf_documents
from processing.hierarchical_chunker import hierarchical_chunk_documents
from embedding.embedder import Embedder
from indexing.vector_index import VectorIndex
from indexing.bm25_index import BM25Index
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.query_analyzer import QueryAnalyzer
from retrieval.query_expander import LLMQueryExpander
from llm.llm_client import get_llm_client, load_llm_config_from_settings


def build_pipeline(data_folder):
    docs = load_pdf_documents(data_folder)
    chunks = hierarchical_chunk_documents(docs)

    embedder = Embedder()
    embeddings = embedder.embed_chunks(chunks)

    dim = len(embeddings[0])

    vector_index = VectorIndex(dim)
    vector_index.add_embeddings(embeddings, chunks)

    bm25_index = BM25Index(chunks)

    query_analyzer = QueryAnalyzer()
    #initialize llm and inject into it
    llm_config = load_llm_config_from_settings(settings)
    llm_client = get_llm_client(llm_config)
    query_expander = LLMQueryExpander(llm_client)

    retriever = HybridRetriever(
        vector_index=vector_index,
        bm25_index=bm25_index,
        embedder=embedder,
        query_analyzer=query_analyzer,
        query_expander=query_expander,
        vector_top_k=12,
        bm25_top_k=12,
    )

    reranker = Reranker()

    return retriever, reranker