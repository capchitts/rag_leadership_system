# schemas.py
# Production-grade shared schemas for a RAG system

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator, conint, confloat

# ============================================================
# Base / Shared
# ============================================================

class SchemaBase(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )


# ============================================================
# Metadata Schemas
# ============================================================

ContentType = Literal["text", "table", "table_row", "image", "ocr_text", "unknown"]
DocType = Literal["pdf", "docx", "pptx", "html", "txt", "unknown"]


class ChunkMetadata(SchemaBase):
    # document identity
    doc_id: str
    source: str = Field(..., description="Original source file name, e.g. sample_report.pdf")
    source_file: Optional[str] = Field(
        default=None,
        description="Optional duplicate/fallback source field for compatibility"
    )
    file_path: Optional[str] = None
    doc_type: DocType = "pdf"

    # chunk identity
    chunk_id: Optional[str] = None
    chunk_index: Optional[int] = None

    # location / structure
    page: Optional[int] = None
    page_label: Optional[str] = None
    section: str = "general"
    section_raw: Optional[str] = None
    subsection: Optional[str] = None
    heading: Optional[str] = None

    # content classification
    type: ContentType = "text"
    content_type: Optional[ContentType] = None

    # positional metadata
    char_start: Optional[int] = None
    char_end: Optional[int] = None

    # table metadata
    table_id: Optional[str] = None
    table_index: Optional[int] = None
    row_index: Optional[int] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    headers: Optional[List[str]] = None

    # quality / extraction
    char_length: Optional[int] = None
    word_count: Optional[int] = None
    extraction_method: Optional[str] = None
    extraction_confidence: Optional[confloat(ge=0.0, le=1.0)] = 1.0

    # ingestion / lineage
    ingested_at: Optional[datetime] = None
    embedding_model: Optional[str] = None
    embedding_version: Optional[str] = None

    extra: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def sync_source_fields(self):
        if not self.source_file:
            self.source_file = self.source
        return self

    @model_validator(mode="after")
    def sync_content_type(self):
        if not self.content_type:
            self.content_type = self.type
        return self

    @model_validator(mode="after")
    def validate_char_range(self):
        if self.char_start is not None and self.char_end is not None:
            if self.char_end < self.char_start:
                raise ValueError("char_end must be >= char_start")
        return self


# ============================================================
# Chunk / Document Schemas
# ============================================================

class DocumentUnit(SchemaBase):
    """
    A raw extracted unit before chunking.
    Could be a page of text, a table, etc.
    """
    text: str = Field(..., min_length=1)
    metadata: ChunkMetadata

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str):
        if not v or not v.strip():
            raise ValueError("DocumentUnit.text cannot be empty")
        return v


class Chunk(SchemaBase):
    chunk_id: str
    text: str = Field(..., min_length=1)
    metadata: ChunkMetadata

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str):
        if not v or not v.strip():
            raise ValueError("Chunk.text cannot be empty")
        return v

    @model_validator(mode="after")
    def sync_chunk_id_to_metadata(self):
        if not self.metadata.chunk_id:
            self.metadata.chunk_id = self.chunk_id
        return self


# ============================================================
# Retrieval / Indexing Schemas
# ============================================================

class RetrievalScores(SchemaBase):
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    rerank_score: Optional[float] = None


class RetrievalCandidate(SchemaBase):
    chunk: Chunk
    scores: RetrievalScores = Field(default_factory=RetrievalScores)
    retrieved_by: List[Literal["bm25", "vector", "hybrid", "reranker"]] = Field(default_factory=list)


class RetrievalDebugInfo(SchemaBase):
    query_variant: str
    vector_results_count: int = 0
    bm25_results_count: int = 0
    selected_chunk_ids: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


# ============================================================
# Query Analysis / Routing Schemas
# ============================================================

QueryType = Literal[
    "exact_lookup",
    "table_lookup",
    "semantic_broad",
    "natural_language_detailed",
    "standard"
]

RetrievalMode = Literal["bm25", "vector", "hybrid"]


class QueryPlan(SchemaBase):
    original_query: str
    normalized_query: str
    query_type: QueryType = "standard"

    should_expand: bool = False
    expansion_terms: List[str] = Field(default_factory=list)
    query_variants: List[str] = Field(default_factory=list)

    use_bm25: bool = True
    use_vector: bool = True
    retrieval_mode: RetrievalMode = "hybrid"

    bm25_weight: confloat(ge=0.0, le=1.0) = 0.5
    vector_weight: confloat(ge=0.0, le=1.0) = 0.5

    max_expansions: conint(ge=1, le=10) = 1
    filters: Dict[str, Any] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_weights(self):
        total = round(self.bm25_weight + self.vector_weight, 6)
        if total == 0:
            raise ValueError("bm25_weight + vector_weight cannot both be 0")
        return self


# ============================================================
# Prompt / Generation Schemas
# ============================================================

GenerationMode = Literal["brief", "detailed", "qa", "executive"]


class PromptBundle(SchemaBase):
    system_prompt: str
    user_prompt: str
    packed_context: str
    mode: GenerationMode = "executive"


class Citation(SchemaBase):
    chunk_id: str
    source: str
    page: Optional[int] = None
    section: Optional[str] = None


class GroundedAnswer(SchemaBase):
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"
    insufficiency_reason: Optional[str] = None


# ============================================================
# Evaluation Schemas
# ============================================================

class FaithfulnessResult(SchemaBase):
    faithfulness_score: Optional[conint(ge=1, le=10)] = None
    directly_supported_claims: List[str] = Field(default_factory=list)
    partially_supported_claims: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    summary: str = ""


class GroundednessResult(SchemaBase):
    groundedness_score: Optional[conint(ge=1, le=10)] = None
    explanation: str = ""
    unsupported_statements: List[str] = Field(default_factory=list)


class RetrievalMetricsResult(SchemaBase):
    precision_at_5: Union[str, float] = "N/A"
    recall_at_5: Union[str, float] = "N/A"
    mrr: Optional[float] = None
    ndcg_at_5: Optional[float] = None


class EvaluationResult(SchemaBase):
    query: str
    retrieval: RetrievalMetricsResult
    faithfulness: FaithfulnessResult
    groundedness: GroundednessResult


# ============================================================
# Offline Eval Dataset Schemas
# ============================================================

class EvalExample(SchemaBase):
    query: str
    expected_answer: str
    relevant_chunk_ids: List[str] = Field(default_factory=list)
    expected_sources: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


# ============================================================
# LLM Config Schema
# ============================================================

LLMProvider = Literal["groq", "google", "huggingface"]


class LLMRuntimeConfig(SchemaBase):
    provider: LLMProvider
    model: str
    temperature: float = 0.2
    max_tokens: int = 1200
    timeout: int = 120
    base_url: Optional[str] = None
    api_version: Optional[str] = None


# ============================================================
# Pipeline Trace / Observability Schemas
# ============================================================

class PipelineTrace(SchemaBase):
    query: str
    query_plan: Optional[QueryPlan] = None
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    reranked_chunk_ids: List[str] = Field(default_factory=list)
    final_answer: Optional[str] = None
    evaluation: Optional[EvaluationResult] = None
    latency_ms: Optional[float] = None
    notes: List[str] = Field(default_factory=list)


# ============================================================
# Helper Constructors
# ============================================================

def make_chunk_id(source: str, page: Optional[int], chunk_index: int) -> str:
    safe_source = source.replace(".pdf", "").replace(" ", "_")
    page_part = f"p{page}" if page is not None else "px"
    return f"{safe_source}_{page_part}_c{chunk_index}"





class ExecutiveBullet(BaseModel):
    text: str = Field(..., min_length=1)
    citations: List[str] = Field(default_factory=list)


class ExecutiveReportResult(BaseModel):
    executive_summary: List[ExecutiveBullet] = Field(default_factory=list)
    top_risks: List[ExecutiveBullet] = Field(default_factory=list)
    top_opportunities: List[ExecutiveBullet] = Field(default_factory=list)
    recommended_actions: List[ExecutiveBullet] = Field(default_factory=list)
    supporting_evidence: List[ExecutiveBullet] = Field(default_factory=list)


class QueryExpansionResult(BaseModel):
    expanded_queries: List[str] = Field(default_factory=list, min_length=1, max_length=5)