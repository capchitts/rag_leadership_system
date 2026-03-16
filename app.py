from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from config import settings
from evaluation.evaluator import RAGEvaluator
from generation.context_packer import pack_context
from generation.insight_generator import generate_insight
from llm.llm_client import get_llm_client, load_llm_config_from_settings
from pipeline.rag_pipeline import build_pipeline
from utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="RAG Leadership System",
    page_icon="📊",
    layout="wide",
)


def validate_data_path(data_path: str) -> str:
    resolved = str(Path(data_path).resolve())
    if not Path(resolved).exists():
        raise FileNotFoundError(f"Data path does not exist: {resolved}")
    return resolved


@st.cache_resource(show_spinner=False)
def load_pipeline(data_path: str):
    return build_pipeline(data_path)


def compute_uploads_fingerprint(uploaded_files) -> str:
    hasher = hashlib.sha256()

    # sort by filename for stable fingerprint across upload order changes
    sorted_files = sorted(uploaded_files, key=lambda f: f.name)

    for uploaded_file in sorted_files:
        file_bytes = uploaded_file.getvalue()
        hasher.update(uploaded_file.name.encode("utf-8"))
        hasher.update(b"::")
        hasher.update(file_bytes)
        hasher.update(b"||")

    return hasher.hexdigest()


def save_uploaded_files_stable(uploaded_files) -> str:
    """
    Save uploaded files into a deterministic temp directory based on file content hash.
    Same uploaded content => same folder path => Streamlit cache can reuse pipeline.
    """
    fingerprint = compute_uploads_fingerprint(uploaded_files)
    base_dir = Path(tempfile.gettempdir()) / "rag_uploads_cache"
    target_dir = base_dir / fingerprint
    target_dir.mkdir(parents=True, exist_ok=True)

    sorted_files = sorted(uploaded_files, key=lambda f: f.name)

    for uploaded_file in sorted_files:
        file_path = target_dir / uploaded_file.name
        if not file_path.exists():
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

    return str(target_dir.resolve())


def run_rag_query(data_path: str, query: str, rerank_top_k: int):
    retriever, reranker, build_tracer = load_pipeline(data_path)

    retrieval_debug = retriever.retrieve_with_debug(query)
    retrieved_chunks = retrieval_debug["results"]

    if not retrieved_chunks:
        return {
            "error": "No relevant chunks were retrieved.",
            "build_trace": build_tracer.to_dict(),
        }

    final_chunks = reranker.rerank(query, retrieved_chunks, top_k=rerank_top_k)

    if not final_chunks:
        return {
            "error": "No chunks remained after reranking.",
            "build_trace": build_tracer.to_dict(),
        }

    packed_context = pack_context(
        final_chunks,
        max_chars=3500,
        max_chunks_per_section=2,
    )

    answer = generate_insight(query, final_chunks)

    evaluation_result = None
    try:
        llm_config = load_llm_config_from_settings(settings)
        llm_client = get_llm_client(llm_config)
        rag_evaluator = RAGEvaluator(llm_client)

        evaluation_result = rag_evaluator.evaluate(
            query=query,
            retrieved_chunks=final_chunks,
            answer=answer,
            context=packed_context,
            relevant_chunk_ids=None,
            graded_relevance=None,
            retrieval_k=rerank_top_k,
        )
    except Exception as exc:
        logger.exception("Evaluation failed")
        evaluation_result = {"error": str(exc)}

    return {
        "build_trace": build_tracer.to_dict(),
        "query_plan": (
            retrieval_debug["query_plan"].model_dump()
            if hasattr(retrieval_debug["query_plan"], "model_dump")
            else retrieval_debug["query_plan"]
        ),
        "query_variants": retrieval_debug["query_variants"],
        "retrieved_chunks": retrieved_chunks,
        "final_chunks": final_chunks,
        "packed_context": packed_context,
        "answer": answer,
        "evaluation": (
            evaluation_result.model_dump()
            if hasattr(evaluation_result, "model_dump")
            else evaluation_result
        ),
    }


def render_chunk_card(idx, chunk):
    metadata = chunk.metadata
    st.markdown(f"**Chunk {idx}**")
    st.write(f"**Source:** {metadata.source or 'unknown'}")
    st.write(f"**Page:** {metadata.page or 'unknown'}")
    st.write(f"**Section:** {metadata.section or 'general'}")
    st.code(chunk.text[:1200], language="text")


def main():
    st.title("📊 RAG Leadership System")

    project_root = Path(__file__).resolve().parent
    default_data_path = str((project_root / settings.DATA_DIR).resolve())

    with st.sidebar:
        st.header("Configuration")
        mode = st.radio("Document Source", ["Use existing folder", "Upload PDFs"])

        rerank_top_k = st.number_input(
            "Rerank Top K",
            min_value=1,
            max_value=20,
            value=settings.RERANK_TOP_K,
            step=1,
        )

        st.caption(f"Vector Top K: {settings.VECTOR_TOP_K}")
        st.caption(f"BM25 Top K: {settings.BM25_TOP_K}")
        st.caption(f"Final Retrieval K: {settings.FINAL_RETRIEVAL_K}")
        st.caption(f"Embedding Model: {settings.EMBEDDING_MODEL}")
        st.caption(f"LLM Model: {settings.LLM_MODEL}")

    upload_cache_path = None

    if mode == "Use existing folder":
        data_path = st.text_input("Data folder path", value=default_data_path)

    else:
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            upload_cache_path = save_uploaded_files_stable(uploaded_files)
            data_path = upload_cache_path
            st.success(f"Loaded {len(uploaded_files)} file(s).")
            st.caption(f"Stable upload cache: {upload_cache_path}")
        else:
            data_path = None

    query = st.text_area(
        "Enter your leadership/business question",
        placeholder="Example: What are the key strategic risks and growth opportunities in this report?",
        height=120,
    )

    run_button = st.button("Run Analysis", type="primary")

    if run_button:
        if not query.strip():
            st.warning("Query cannot be empty.")
            return

        if not data_path:
            st.warning("Please provide a valid data source.")
            return

        try:
            valid_data_path = validate_data_path(data_path)
        except Exception as exc:
            st.error(str(exc))
            return

        with st.spinner("Running RAG pipeline..."):
            result = run_rag_query(
                data_path=valid_data_path,
                query=query.strip(),
                rerank_top_k=int(rerank_top_k),
            )

        if "error" in result:
            st.error(result["error"])
            return

        st.subheader("Executive Report")
        st.markdown(result["answer"])

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Evaluation", "Query Plan", "Retrieved Chunks", "Packed Context", "Traces"]
        )

        with tab1:
            st.json(result["evaluation"])

        with tab2:
            st.json(
                {
                    "query_plan": result["query_plan"],
                    "query_variants": result["query_variants"],
                }
            )

        with tab3:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Retrieved Chunks")
                for idx, chunk in enumerate(result["retrieved_chunks"], start=1):
                    render_chunk_card(idx, chunk)

            with col2:
                st.markdown("### Final Reranked Chunks")
                for idx, chunk in enumerate(result["final_chunks"], start=1):
                    render_chunk_card(idx, chunk)

        with tab4:
            st.text_area(
                "Packed Context",
                value=result["packed_context"],
                height=500,
            )

        with tab5:
            st.json(result["build_trace"])


if __name__ == "__main__":
    main()