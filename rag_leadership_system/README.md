# RAG Leadership Insight System

A production-oriented **Retrieval Augmented Generation (RAG)** system that extracts strategic insights, risks, and growth opportunities from enterprise reports.

The system performs:

* Hybrid Retrieval (Vector + BM25)
* Reranking
* Structured Context Packing
* Grounded Insight Generation
* RAG Evaluation
* Observability & Tracing
* Streamlit Frontend

The goal is to convert long enterprise documents into **concise CXO-level insights** grounded in evidence.

---

# System Architecture

```
                ┌────────────────────┐
                │     Documents       │
                │  (PDF Reports)     │
                └─────────┬──────────┘
                          │
                          ▼
               ┌────────────────────┐
               │    Ingestion       │
               │  (Offline Batch)   │
               └─────────┬──────────┘
                          │
      ┌───────────────────┼───────────────────┐
      ▼                   ▼                   ▼

Document Loader      Hierarchical        Embedding
(PDF Parsing)        Chunking             Generation
                                          (MiniLM)

      ▼
Vector Index (FAISS)
      +
BM25 Index
      ▼

Persisted Artifacts
artifacts/
 ├── faiss.index
 ├── chunk_store.pkl
 └── bm25_index.pkl


======================================================


              ┌───────────────────────┐
              │       User Query      │
              └───────────┬───────────┘
                          ▼
                Query Analyzer
                          ▼
               Hybrid Retriever
              (Vector + BM25)
                          ▼
                    Reranker
                          ▼
                 Context Packing
                          ▼
                LLM Insight Engine
                          ▼
               Structured Output
                          ▼
                   Evaluation
                          ▼
                  Observability
                          ▼
                    MongoDB Logs
```

---

# Key Features

## Hybrid Retrieval

The system combines two retrieval strategies:

Vector Retrieval

Semantic similarity search using sentence embeddings.

BM25 Retrieval

Keyword based lexical search.

Hybrid fusion improves recall and robustness.

```
Final score = (vector_weight × vector_score)
            + (bm25_weight × bm25_score)
```

---

## Query Understanding

The system analyzes the user query before retrieval.

Query types supported:

| Query Type                | Example                           |
| ------------------------- | --------------------------------- |
| exact_lookup              | invoice number                    |
| table_lookup              | pricing table                     |
| semantic_broad            | product benefits                  |
| natural_language_detailed | strategic risks and opportunities |

The analyzer automatically adjusts:

* retrieval weights
* query expansion
* search strategy

---

## Hierarchical Chunking

Documents are chunked into structured units:

```
document
  ├─ sections
  │   ├─ paragraphs
  │   └─ tables
```

Each chunk contains metadata:

```
chunk_id
source_file
page
section
```

This enables precise citation grounding.

---

## Reranking

Initial retrieval candidates are reranked using semantic similarity to improve relevance.

Pipeline:

```
Hybrid retrieval →  top K candidates
                    ↓
                 Reranker
                    ↓
               final chunks
```

---

## Context Packing

Chunks are formatted into a structured context before sending to the LLM.

Example:

```
=== TABLE ===

[sample_report_p2_c0 | sample_report.pdf | p.2 | table]

Table row 1...
Table row 2...
```

This ensures:

* deterministic citations
* structured evidence
* smaller prompts

---

## Insight Generation

The LLM produces a structured executive report:

```
Executive Summary
Top Risks
Top Opportunities
Recommended Actions
Supporting Evidence
```

All claims must include citations.

Example:

```
Store network productivity could weaken if fixed costs remain unchanged.
[sample_report_p2_c0 | sample_report.pdf | p.2 | table]
```

---

# RAG Evaluation

The system evaluates responses automatically.

Metrics implemented:

### Retrieval Metrics

| Metric      | Purpose                       |
| ----------- | ----------------------------- |
| Precision@K | relevance of retrieved chunks |
| Recall@K    | coverage of relevant chunks   |
| MRR         | ranking quality               |
| NDCG        | graded relevance              |

---

### Generation Metrics

Faithfulness

Checks whether claims are supported by context.

Groundedness

Detects hallucinations.

Example evaluation:

```
faithfulness_score: 7
groundedness_score: 7
hallucination_detected: true
```

---

# Observability

The system includes tracing and structured logging.

Each query logs:

```
query
query_plan
retrieved_chunks
reranked_chunks
packed_context_chars
answer
evaluation_metrics
latency
```

Logs are stored in MongoDB for analysis.

This enables:

* debugging
* performance monitoring
* failure analysis

---

# Project Structure

```
project/
│
├── ingestion/
│   └── pdf_loader.py
│
├── processing/
│   └── hierarchical_chunker.py
│
├── embedding/
│   └── embedder.py
│
├── indexing/
│   ├── vector_index.py
│   └── bm25_index.py
│
├── retrieval/
│   ├── hybrid_retriever.py
│   ├── query_analyzer.py
│   ├── query_expander.py
│   └── reranker.py
│
├── generation/
│   ├── context_packer.py
│   └── insight_generator.py
│
├── evaluation/
│   ├── evaluator.py
│   ├── faithfulness.py
│   └── groundedness.py
│
├── observability/
│   └── tracer.py
│
├── pipeline/
│   └── rag_pipeline.py
│
├── artifacts/
│
├── data/
│   └── reports/
│
├── ingest.py
├── main.py
├── app.py
└── README.md
```

---

# Running the System

## Step 1 — Install dependencies

```
pip install -r requirements.txt
```

---

## Step 2 — Add documents

Place PDFs inside:

```
data/reports/
```

---

## Step 3 — Run ingestion

```
python ingest.py
```

This performs:

```
PDF loading
→ hierarchical chunking
→ embedding generation
→ FAISS index creation
→ BM25 index creation
→ artifact persistence
```

Artifacts stored in:

```
artifacts/
```

---

## Step 4 — Launch UI

```
streamlit run app.py
```

Ask questions like:

```
What are the strategic risks and growth opportunities?
```

---

# Example Query Flow

```
User query
      ↓
Query Analyzer
      ↓
Hybrid Retrieval
      ↓
Reranking
      ↓
Context Packing
      ↓
LLM Insight Generation
      ↓
Evaluation
      ↓
Observability Logs
```

---

# Design Decisions

## Why Hybrid Retrieval?

Vector search alone struggles with:

* numbers
* identifiers
* tables

BM25 improves recall for structured content.

---

## Why Reranking?

Hybrid retrieval increases recall but reduces precision.

Reranking fixes this by selecting the most relevant chunks.

---

## Why Structured Context?

Prevents hallucination and enforces grounding.

---

# Scaling to 100k Users

Current system is optimized for **single node inference**.

For production scale, the following architecture is recommended.

---

## Scalable Architecture

```
                 ┌──────────────┐
                 │  Load Balancer│
                 └──────┬───────┘
                        ▼
               Query API Service
                        ▼
                 Retrieval Service
           ┌────────────┼────────────┐
           ▼            ▼            ▼
        Vector DB      BM25      Reranker
       (Pinecone)     (Elastic)   Service
                        ▼
                     Context
                        ▼
                    LLM Gateway
                        ▼
                 Insight Generation
                        ▼
                 Evaluation Service
                        ▼
                 Observability DB
```

---

## Improvements for 100k users

### Distributed Vector Search

Replace FAISS with:

* Pinecone
* Weaviate
* Milvus

---

### Distributed Keyword Search

Replace BM25 with:

* Elasticsearch
* OpenSearch

---

### Async Query Pipeline

Convert retrieval and generation to async tasks using:

```
FastAPI
async workers
Redis queue
```

---

### Caching

Cache frequent queries using:

```
Redis
```

---

### Streaming Responses

Use server-sent events for incremental generation.

---

### Observability

Add:

```
Prometheus
Grafana
LangSmith
```

---

# Future Improvements

* multi-document comparison
* better table extraction
* advanced reranking models
* automatic query expansion
* RLHF-based evaluation
* agentic retrieval workflows

---

# Technologies Used

| Component      | Technology            |
| -------------- | --------------------- |
| LLM            | Groq / Llama-3        |
| Embeddings     | sentence-transformers |
| Vector Search  | FAISS                 |
| Keyword Search | BM25                  |
| Evaluation     | custom RAG metrics    |
| Frontend       | Streamlit             |
| Observability  | MongoDB               |

---

# Here is an **updated Engineering Tradeoffs section** you can paste into your README.
This version explicitly mentions **index rebuilding, incremental ingestion, scaling considerations, and architectural choices**, which shows strong **system design thinking** to reviewers.

---

ENGINEERING TRADEOFFS

This system intentionally balances **simplicity, reproducibility, and extensibility** to suit an assignment environment while reflecting how a production RAG system would evolve.

---

FAISS vs Managed Vector Databases

The current system uses FAISS for vector similarity search because:

• It is lightweight and easy to deploy locally
• It requires no external infrastructure
• It provides fast approximate nearest neighbor search

However, FAISS runs on a single machine and does not support distributed indexing.

For large scale production deployments supporting tens of thousands of users and millions of documents, FAISS would typically be replaced by a managed vector database such as:

• Pinecone
• Weaviate
• Milvus

These systems support distributed indexing, replication, and horizontal scaling.

---

Local BM25 vs Elasticsearch

The system implements BM25 locally for keyword retrieval.

Advantages:

• No external dependencies
• Fast indexing for small to medium document collections
• Simple implementation

However, for large document collections, a distributed search engine would be more appropriate.

Production systems typically replace local BM25 with:

• Elasticsearch
• OpenSearch

These engines provide scalable keyword search, filtering, and advanced ranking capabilities.

---

Index Rebuilding vs Incremental Ingestion

The ingestion pipeline currently **rebuilds the entire index whenever ingestion runs**.

This design was chosen because:

• it guarantees deterministic results
• it avoids index corruption issues
• it simplifies the ingestion logic

Pipeline behavior:

load documents → chunk → embed → rebuild FAISS → rebuild BM25

In a production environment with large document collections, the system would support **incremental ingestion** instead.

Incremental ingestion typically works as follows:

scan data directory
compare documents against a manifest
ingest only new or modified files
update indexes incrementally

This approach avoids recomputing embeddings for previously processed documents and significantly reduces ingestion time.

---

Synchronous Pipeline vs Asynchronous Architecture

The current pipeline executes sequentially:

retrieval → reranking → generation → evaluation

This simplifies the implementation and makes debugging easier.

In a high scale environment, this would typically be converted into an **asynchronous architecture** using:

• FastAPI for query services
• Redis or Kafka for task queues
• background worker processes for LLM inference

This allows the system to handle many concurrent requests efficiently.

---

Manual Ingestion vs Event Driven Ingestion

Currently ingestion is executed manually:

python ingest.py

This design ensures that the system remains reproducible and easy to run for demonstration purposes.

In production systems ingestion is usually triggered automatically when new documents are uploaded.

Typical architecture:

document upload API → storage → ingestion queue → background ingestion worker → index update

This enables continuous indexing without manual intervention.

---

Local Observability vs Production Monitoring

The system currently logs pipeline events and evaluation metrics to MongoDB.

This provides visibility into:

• retrieval results
• reranked chunks
• LLM outputs
• evaluation metrics

In production systems observability would typically be expanded with:

• Prometheus for metrics collection
• Grafana dashboards for monitoring
• distributed tracing tools such as OpenTelemetry
• LLM monitoring platforms such as LangSmith

---

LLM API vs Local Models

The system uses Groq-hosted LLM inference.

Advantages:

• high performance inference
• simple integration
• no GPU infrastructure required

For enterprise deployments, organizations may instead run local models using:

• vLLM
• TensorRT-LLM
• self-hosted inference clusters

This provides better control over latency, cost, and data privacy.

---

Summary

The current implementation prioritizes:

• reproducibility
• simplicity
• correctness

while leaving clear upgrade paths for:

• distributed retrieval infrastructure
• incremental ingestion
• asynchronous pipelines
• production-grade observability
• large-scale deployment supporting 100k+ users.

---

# HOW TO RUN

This project separates offline document ingestion from the online query pipeline.

Typical workflow:
    Add Documents → Run Ingestion → Launch Query Interface

Clone the Repository
    git clone <repository-url>
    cd <repository-name>

Create Python Environment
    Recommended Python version: Python 3.10+
    Create virtual environment: python -m venv venv
    Activate environment
        Mac / Linux: source venv/bin/activate
        Windows: venv\Scripts\activate
    Install Dependencies
        pip install -r requirements.txt

    Configure Environment Variables
        Create a .env file in the root directory:
            GROQ_API_KEY=your_groq_key
            MONGODB_URI=mongodb://localhost:27017

    Add Documents
        Place PDF documents inside the configured data directory.
        Default location: data/reports/

        Example:

        data
        └── reports
        ├── retail_report.pdf
        ├── earnings_report.pdf
        └── strategy_overview.pdf

    Run Document Ingestion
        Run the ingestion pipeline: python ingest.py

        This step performs:
            PDF loading
            → hierarchical chunking
            → embedding generation
            → FAISS vector index creation
            → BM25 keyword index creation
            → artifact persistence

            Artifacts will be saved in:

            artifacts
            ├── faiss.index
            ├── chunk_store.pkl
            └── bm25_index.pkl

        This step only needs to be run when new documents are added.

    Launch the Query Interface
    Run the Streamlit application: streamlit run app.py
    Open browser: http://localhost:8501

    You can now:
        • Upload documents
        • Ask strategic questions
        • View generated leadership insights

    Example query:"What strategic risks and growth opportunities are highlighted in this report?"
    Optional: Run via CLI
    You can also query the system from command line.
        python main.py

        Example:
        Enter your leadership/business question:
        "What are the key operational risks in this report?"

    MongoDB Logging
    The system logs query traces and evaluation metrics into MongoDB.
    
    Run MongoDB using Docker: docker run -d -p 27017:27017 --name rag-mongo mongo

    View logs using:
        • MongoDB Compass
        • MongoDB shell

    Each query stores:
        query
        retrieved_chunks
        reranked_chunks
        packed_context_chars
        answer
        evaluation_metrics
        latency

    Example End-to-End Workflow
        Add PDFs to data/reports/
        Run ingestion
            python ingest.py
        Start UI
            streamlit run app.py
    
    Ask questions about the documents

    Troubleshooting

        FAISS index not found
            Run ingestion again: python ingest.py

        MongoDB connection error
            Check container status: docker ps
            Restart container: docker restart rag-mongo

        Streamlit not loading
            Ensure port is open: http://localhost:8501

        Restart Streamlit: streamlit run app.py

# Author
RAG System developed as part of an AI engineering assignment demonstrating production-grade retrieval architecture and grounded generation pipelines.

---

