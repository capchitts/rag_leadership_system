# config/settings.py
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _get_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value and value.strip() else default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value and value.strip() else default


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ============================================================
# App / data
# ============================================================

DATA_DIR = _get_str("DATA_DIR", "data/reports")
LOG_LEVEL = _get_str("LOG_LEVEL", "INFO")


# ============================================================
# Chunking
# ============================================================

TEXT_CHUNK_MAX_CHARS = _get_int("TEXT_CHUNK_MAX_CHARS", 800)
TEXT_CHUNK_OVERLAP_PARAGRAPHS = _get_int("TEXT_CHUNK_OVERLAP_PARAGRAPHS", 1)

TABLE_CHUNK_MAX_CHARS = _get_int("TABLE_CHUNK_MAX_CHARS", 1200)
TABLE_CHUNK_OVERLAP_ROWS = _get_int("TABLE_CHUNK_OVERLAP_ROWS", 1)


# ============================================================
# Embeddings
# ============================================================

EMBEDDING_MODEL = _get_str("EMBEDDING_MODEL", "BAAI/bge-large-en")
EMBEDDING_BATCH_SIZE = _get_int("EMBEDDING_BATCH_SIZE", 32)


# ============================================================
# Retrieval
# ============================================================

VECTOR_TOP_K = _get_int("VECTOR_TOP_K", 15)
BM25_TOP_K = _get_int("BM25_TOP_K", 15)
FINAL_RETRIEVAL_K = _get_int("FINAL_RETRIEVAL_K", 10)

RRF_K = _get_int("RRF_K", 60)

ENABLE_QUERY_EXPANSION = _get_bool("ENABLE_QUERY_EXPANSION", True)
MAX_QUERY_EXPANSIONS = _get_int("MAX_QUERY_EXPANSIONS", 3)


# ============================================================
# Reranking
# ============================================================

ENABLE_RERANKING = _get_bool("ENABLE_RERANKING", True)
RERANK_TOP_K = _get_int("RERANK_TOP_K", 5)
RERANKER_MODEL = _get_str("RERANKER_MODEL", "BAAI/bge-reranker-large")


# ============================================================
# Context packing
# ============================================================

CONTEXT_MAX_CHARS_BRIEF = _get_int("CONTEXT_MAX_CHARS_BRIEF", 3500)
CONTEXT_MAX_CHARS_DETAILED = _get_int("CONTEXT_MAX_CHARS_DETAILED", 7000)

MAX_CHUNKS_PER_SECTION_BRIEF = _get_int("MAX_CHUNKS_PER_SECTION_BRIEF", 2)
MAX_CHUNKS_PER_SECTION_DETAILED = _get_int("MAX_CHUNKS_PER_SECTION_DETAILED", 4)


# ============================================================
# LLM provider
# ============================================================

# Active provider: groq | huggingface | google
LLM_PROVIDER = _get_str("LLM_PROVIDER", "groq").lower()

LLM_TEMPERATURE = _get_float("LLM_TEMPERATURE", 0.2)
LLM_MAX_TOKENS = _get_int("LLM_MAX_TOKENS", 1200)
LLM_TIMEOUT = _get_int("LLM_TIMEOUT", 120)

LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_VERSION = os.getenv("LLM_API_VERSION")


# ============================================================
# Provider-specific model names
# ============================================================

GROQ_MODEL = _get_str("GROQ_MODEL", "llama-3.3-70b-versatile")
HUGGINGFACE_MODEL = _get_str("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
GOOGLE_MODEL = _get_str("GOOGLE_MODEL", "gemini-2.5-flash")


def _resolve_active_llm_model() -> str:
    if LLM_PROVIDER == "groq":
        return GROQ_MODEL
    if LLM_PROVIDER == "huggingface":
        return HUGGINGFACE_MODEL
    if LLM_PROVIDER in {"google", "gemini"}:
        return GOOGLE_MODEL
    return GROQ_MODEL


# Model actually used by llm_client.py
LLM_MODEL = _get_str("LLM_MODEL", _resolve_active_llm_model())


# ============================================================
# API keys
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Active provider: groq | huggingface | google
LLM_PROVIDER = "groq"

# Model for selected provider
LLM_MODEL = "llama-3.3-70b-versatile"

LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 1200
LLM_TIMEOUT = 120

LLM_BASE_URL = None
LLM_API_VERSION = None

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")