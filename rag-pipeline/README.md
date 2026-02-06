# Full-Stack Gen AI Application: Legal Document Retrieval System

> An end-to-end Retrieval-Augmented Generation system that transforms how people search and understand county-level legal ordinances across multiple states.

## Collaborators

Built as the capstone project for UC Berkeley MIDS program with Kevin Yi, Roz Huang, Arthur Kang, and Armaan Hiranandani.

---

## 01. The Problem

### Context

County-level legal ordinances in the United States are scattered across thousands of municipal websites, buried in dense PDF documents with inconsistent formatting. A resident trying to understand whether their county requires a dog leash, what the penalty structure looks like, or how noise ordinances compare across jurisdictions has no practical way to search this information semantically. They're stuck downloading individual PDFs and scanning them page by page.

### Challenge

We needed to build a system that could:
- **Ingest** thousands of legal PDFs from municipal websites across multiple states
- **Extract** text from documents with wildly inconsistent formatting — some typeset, some scanned images, some multi-column layouts
- **Search** across these documents using natural language questions, not just keyword matching
- **Answer** questions with LLM-generated responses grounded in the actual legal text
- **Filter** results by state, county, legal classification (penalties, prohibitions, obligations), and readability metrics
- **Deploy** the entire system to production on AWS with GPU-accelerated inference

The core technical challenge: how do you build a retrieval system that handles both conceptual queries ("What are the rules about dogs in parks?") and keyword-specific queries ("leash fine penalty amount") equally well?

The answer turned out to be **hybrid search** — combining dense semantic embeddings with sparse keyword vectors, then reranking with a cross-encoder. This became the architectural backbone of the entire system.

---

## 02. The Approach

### System Architecture

The system is composed of five independently deployable components, each optimized for its specific workload:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AWS Cloud Infrastructure                        │
│                                                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────────┐ │
│  │  S3 Bucket   │    │     ECR      │    │    CloudWatch Logging     │ │
│  │ ┌──────────┐ │    │  Container   │    │  - ECS task logs          │ │
│  │ │input/pdf/│ │    │  Registry    │    │  - API request logs       │ │
│  │ │processed/│ │    └──────┬───────┘    └───────────────────────────┘ │
│  │ └──────────┘ │           │                                          │
│  └──────┬───────┘           │                                          │
│         │                   ▼                                          │
│         │         ┌──────────────────┐                                 │
│         ├────────▶│   ECS Fargate    │──── Pipeline 1: PDF Extraction  │
│         │         │  (CPU, 4GB RAM)  │                                 │
│         │         └────────┬─────────┘                                 │
│         │                  │ Parquet                                    │
│         │                  ▼                                           │
│         │         ┌──────────────────┐                                 │
│         │         │ Pinecone Embed   │──── Pipeline 2: Vectorization   │
│         │         │  (Local/Batch)   │                                 │
│         │         └────────┬─────────┘                                 │
│         │                  │ Vectors                                    │
│         │                  ▼                                           │
│         │         ┌──────────────────┐       ┌────────────────────┐    │
│         │         │  Pinecone DB     │◀─────▶│  EC2 g4dn.xlarge   │   │
│         │         │  (Serverless)    │       │  NVIDIA T4 GPU     │   │
│         │         │  1024-dim dense  │       │  ┌──────────────┐  │   │
│         │         │  + sparse BM25   │       │  │ Flask API    │  │   │
│         │         └──────────────────┘       │  │ LLaMA 3.1 8B│  │   │
│         │                                    │  │ Reranker     │  │   │
│         │                                    │  └──────┬───────┘  │   │
│         │                                    └─────────┼──────────┘   │
│         │                                              │               │
│         │                                              ▼               │
│         │                                    ┌────────────────────┐    │
│         │                                    │ Elastic Beanstalk  │   │
│         │                                    │ ┌──────────────┐   │   │
│         │                                    │ │ Streamlit UI │   │   │
│         │                                    │ │ Port 8501    │   │   │
│         │                                    │ └──────────────┘   │   │
│         │                                    └────────────────────┘   │
│         │                                                              │
└─────────┴──────────────────────────────────────────────────────────────┘
                                    │
                               ┌────▼────┐
                               │  Users  │
                               └─────────┘
```

### Data Flow: Document Ingestion to Answer Generation

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Municode   │     │   PyMuPDF   │     │   Pinecone  │     │   Pinecone  │
│  Web Crawler│────▶│   + OCR     │────▶│  Inference  │────▶│  Vector DB  │
│  (Selenium) │     │  Extraction │     │    API      │     │  (Indexed)  │
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                    │
     PDFs              Parquet             Dense + Sparse           │
  from county        (chunked text,        embeddings with         │
   websites         state/county tags)      metadata               │
                                                                    │
┌───────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Query Pipeline (EC2 GPU)                        │
│                                                                     │
│  User Query ──▶ ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│                 │  Embed   │───▶│ Pinecone │───▶│  Cross-Encoder│  │
│  + Filters     │  Query   │    │ Retrieve │    │  Reranker     │  │
│                 │(dense+   │    │ Top-100  │    │  → Top-5      │  │
│                 │ sparse)  │    │          │    │               │  │
│                 └──────────┘    └──────────┘    └──────┬───────┘  │
│                                                        │          │
│                                                        ▼          │
│                                                 ┌──────────────┐  │
│                                                 │  LLaMA 3.1   │  │
│                                                 │  8B Instruct │  │
│                                                 │  (4-bit NF4) │  │
│                                                 └──────┬───────┘  │
│                                                        │          │
│                              JSON Response ◀───────────┘          │
│                              (answer + chunks + metadata)         │
└─────────────────────────────────────────────────────────────────────┘
```

### Technical Deep Dive

#### Pipeline 0: Municode Web Crawler

Selenium-based scraper running in Google Colab that crawls [Municode Library](https://library.municode.com) to download county-level ordinance PDFs. It filters out city-level municipalities, handles stale elements and timeouts, and stores PDFs to Google Drive for upload to S3.

#### Pipeline 1: Data Engineering (AWS ECS/Fargate)

Containerized text extraction service that reads PDFs from S3 and produces chunked Parquet files. The extraction uses a dual strategy:

- **Primary**: PyMuPDF with layout-aware column detection — handles multi-column legal documents correctly by detecting text block positions and merging columns in reading order
- **Fallback**: Tesseract OCR for scanned or image-based PDFs (capped at 20 pages to control costs)
- **Chunking**: Configurable chunk size (default 1000 chars) with overlap (default 200 chars), preserving sentence boundaries

Output is partitioned Parquet: `s3://bucket/processed/zone=text_chunk/state=california/county=alameda/`

#### Pipeline 2: Pinecone Embedding

Generates hybrid embeddings and indexes them into Pinecone's serverless vector database:

- **Dense embeddings**: `llama-text-embed-v2` (1024 dimensions) — captures semantic meaning
- **Sparse embeddings**: `pinecone-sparse-english-v0` — BM25-style keyword matching
- **Metadata**: State, county, section, readability metrics (Flesch-Kincaid grade, word count, complexity percentage), and legal classifications (penalty, obligation, permission, prohibition)

The combination of dense + sparse vectors in a single index is what enables hybrid search — one of the most impactful architectural decisions in the project.

#### Pipeline 3: RAG Query API (AWS EC2 GPU)

Production Flask REST API exposing two search modes:

| Mode | Retrieval | Speed | Best For |
|------|-----------|-------|----------|
| **Baseline** | Dense embedding only, top-5 | 2-5 seconds | Quick lookups, specific questions |
| **Hybrid** | Dense + sparse, top-100 → rerank to top-5 | 5-10 seconds | Complex queries, cross-county comparison |

The LLM (Meta LLaMA 3.1 8B Instruct) runs with 4-bit NF4 quantization via BitsAndBytes, reducing GPU memory from 16GB to ~6GB. This is critical — it allows the model to run on a T4 GPU (g4dn.xlarge at ~$0.53/hour) instead of requiring an A100 ($3+/hour).

The cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) re-scores the top-100 candidates from hybrid retrieval, producing a final top-5 that's significantly more relevant than dense retrieval alone.

#### Component 4: Streamlit Frontend (AWS Elastic Beanstalk)

Interactive web UI supporting multi-state search (CA, FL, GA, TX), advanced filtering by legal classification and readability metrics, interactive results display with chunk-level scores, and CSV export for offline analysis.

#### Component 5: Evaluation Framework

LLM-as-a-Judge evaluation system using NVIDIA Nemotron via NIMs API. Measures Top-5 Recall, Mean Reciprocal Rank (MRR), chunk coverage, metadata accuracy, and negative test handling (correctly identifying when no law exists).

### Tech Stack

| Layer | Technology | Why This Choice |
|-------|-----------|-----------------|
| **Data Collection** | Selenium, Google Colab | Free compute, built-in Chrome/Drive integration |
| **PDF Processing** | PyMuPDF, Tesseract OCR | Layout-aware extraction + OCR fallback for scanned docs |
| **Data Format** | Parquet, Pandas, PyArrow | Columnar format enables partition pruning by state/county |
| **Storage** | AWS S3 | Scalable data lake, integrates with ECS/EC2 |
| **Batch Compute** | AWS ECS Fargate | Serverless containers, no instance management |
| **Container Registry** | AWS ECR | Native Docker registry for ECS deployments |
| **Vector Database** | Pinecone (Serverless) | Native hybrid search, metadata filtering, zero-ops |
| **Dense Embeddings** | llama-text-embed-v2 | 1024-dim, optimized for retrieval tasks |
| **Sparse Embeddings** | pinecone-sparse-english-v0 | BM25-style keyword matching in same index |
| **LLM** | Meta LLaMA 3.1 8B Instruct | Open-source, strong instruction following, quantizable |
| **Quantization** | BitsAndBytes NF4 | 70% memory reduction, negligible quality loss |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 | Fast cross-encoder, 100→5 reranking in <1s |
| **API Framework** | Flask | Lightweight, sufficient for LLM-dominated latency |
| **GPU Compute** | AWS EC2 g4dn.xlarge (T4) | Cost-effective GPU for 8B model inference |
| **Frontend** | Streamlit | Rapid prototyping, built-in data visualization |
| **Frontend Hosting** | AWS Elastic Beanstalk | Simple deployment, Nginx proxy, auto-restart |
| **Monitoring** | AWS CloudWatch | Unified logging for ECS tasks and EC2 instances |

---

## 03. Results

### System Performance

| Metric | Baseline Mode | Hybrid Mode |
|--------|--------------|-------------|
| Query latency | 2-5 seconds | 5-10 seconds |
| Retrieval candidates | Top-5 (dense) | Top-100 → rerank to Top-5 |
| GPU memory usage | ~6GB (4-bit) | ~6GB (4-bit) |
| Memory reduction | 70% vs full precision | 70% vs full precision |

### Deployment

- **Production-deployed** across 5 AWS services (S3, ECS, ECR, EC2, Elastic Beanstalk) + Pinecone
- **Multi-state coverage**: CA, FL, GA, TX county ordinances indexed and searchable
- **Two search modes**: Users can choose speed (baseline) or accuracy (hybrid) per query
- **Advanced filtering**: Location, legal classification (penalty/obligation/permission/prohibition), readability metrics (Flesch-Kincaid grade, reading ease, word count, complexity)

### Evaluation

Built an LLM-as-a-Judge evaluation framework using NVIDIA Nemotron to systematically measure retrieval quality across dimensions: recall, MRR, chunk coverage, metadata accuracy, and negative test handling (correctly identifying when no applicable law exists).

---

## Technical Decisions & Trade-offs

These are the architectural decisions that shaped the system, and the reasoning behind each one. These map directly to common interview questions.

### Why hybrid search instead of dense-only retrieval?

Dense embeddings capture semantic meaning but miss keyword-specific queries. When a user searches for "leash fine penalty," the word "penalty" is a critical signal that pure semantic search might underweight. Sparse (BM25-style) embeddings handle keyword matching well but miss conceptual relationships. Combining both in a single Pinecone index gives the best of both worlds — semantic understanding for conceptual queries and keyword precision for specific lookups.

### Why 4-bit quantization instead of full precision?

LLaMA 3.1 8B at full precision needs ~16GB of GPU memory, which requires an A100 ($3+/hour). With NF4 quantization via BitsAndBytes, memory drops to ~6GB with negligible quality loss. This enables deployment on a T4 GPU (g4dn.xlarge at ~$0.53/hour) — a 6x cost reduction. For a system that needs to be always-on, this is the difference between $379/month and $2,200+/month.

### Why Pinecone instead of self-hosted (Milvus, Weaviate)?

Pinecone offers native hybrid search (dense + sparse in one query), serverless scaling, and built-in metadata filtering — all without managing infrastructure. Self-hosted alternatives would require separate DevOps for the vector database, and most don't natively support sparse vectors in the same index. For a team focused on the ML pipeline rather than infrastructure, the managed service was the right trade-off.

### Why Flask instead of FastAPI?

Query latency is dominated by LLM generation (2-10 seconds), not the web framework. Flask's async overhead of ~1ms is irrelevant when the model takes 5 seconds to respond. FastAPI's async capabilities would add complexity without meaningful performance improvement. Flask's simplicity made development and debugging faster.

### Why ECS Fargate for data engineering instead of Lambda?

PDF text extraction can take 15+ minutes per file. Lambda has a 15-minute timeout and 10GB memory limit. ECS Fargate allows longer-running tasks with configurable memory (up to 30GB), and the Docker container can include heavy dependencies like Tesseract OCR and PyMuPDF without worrying about Lambda layer size limits.

### Why cross-encoder reranking instead of just increasing top-k?

Simply retrieving more results (higher top-k) doesn't improve relevance — it just adds noise. A cross-encoder processes the query and each candidate jointly, producing much more accurate relevance scores than the initial bi-encoder retrieval. Retrieving 100 candidates and reranking to 5 consistently outperforms retrieving 5 directly. The reranker adds less than 1 second of latency.

### Why S3 partitioning by state/county?

Partitioned Parquet files (`state=california/county=alameda/`) enable partition pruning — when processing a specific county, only that county's data is read. This also maps naturally to Pinecone's metadata filtering, ensuring consistent data organization from storage through retrieval.

---

## Lessons Learned

**PDF extraction is the hardest part.** We expected the ML pipeline to be the bottleneck, but extracting clean text from legal PDFs with inconsistent layouts, multi-column formatting, and scanned pages consumed more engineering time than any other component. Layout-aware extraction with column detection was essential — naive extraction produced garbled text that poisoned downstream embeddings.

**Hybrid search is dramatically better than dense-only.** We initially built with dense embeddings alone. Adding sparse vectors improved retrieval quality noticeably on keyword-heavy legal queries. The cross-encoder reranker on top of that was another significant improvement. Each layer added latency but meaningfully improved answer quality.

**Quantization is effectively free performance.** We expected 4-bit quantization to degrade answer quality. In practice, the quality difference was imperceptible for this use case, while the cost savings were enormous. This should be the default approach for any production LLM deployment on moderate-sized models.

**Metadata-rich chunks enable powerful filtering.** Attaching legal classifications (penalty, prohibition, obligation, permission) and readability metrics (Flesch-Kincaid, word count) to each chunk enables filtering that pure semantic search can't replicate. A user asking "show me all penalty clauses in Alameda County" needs metadata filtering, not better embeddings.

**Separate concerns across services.** Making each pipeline independently deployable was worth the upfront effort. We could iterate on the query API without touching data engineering, and deploy frontend changes without restarting the GPU instance. In production, this separation also allows independent scaling.

---

## Project Structure

```
rag-pipeline/
├── municode-web-crawler/       # Step 0: Web Scraping → PDFs
│   ├── municode_crawler.ipynb  # Selenium scraper notebook
│   └── README.md
│
├── data-engineering/           # Pipeline 1: PDF → Parquet (ECS Fargate)
│   ├── main.py                 # Text extraction + chunking
│   ├── Dockerfile              # ECS container definition
│   ├── requirements.txt
│   └── README.md
│
├── pinecone-embedding/         # Pipeline 2: Parquet → Pinecone (Local/Batch)
│   ├── src/rag_ingest/
│   │   ├── ingest.py           # Main orchestration
│   │   ├── s3_loader.py        # S3 data loading
│   │   ├── embed_dense.py      # Dense embedding generation
│   │   ├── embed_sparse.py     # Sparse embedding generation
│   │   └── upsert.py           # Vector construction & upload
│   ├── pyproject.toml
│   └── README.md
│
├── rag-query/                  # Pipeline 3: Query API (EC2 GPU)
│   ├── api.py                  # Flask REST API entry point
│   ├── pipeline.py             # RAG orchestration
│   ├── models.py               # LLM + reranker initialization
│   ├── retrieval.py            # Pinecone retrieval logic
│   ├── llm_generation.py       # Prompt engineering + generation
│   ├── filters.py              # Filter processing
│   ├── config.py               # Configuration
│   ├── Dockerfile              # CUDA 12.1 + Python 3.10
│   ├── docker-compose.yml      # GPU device mapping
│   └── README.md
│
├── streamlit-app/              # Component 4: Frontend (Elastic Beanstalk)
│   ├── app.py                  # Streamlit application
│   ├── Procfile                # Elastic Beanstalk config
│   ├── requirements.txt
│   └── README.md
│
├── evaluation/                 # Component 5: LLM-as-Judge Evaluation
│   ├── ...                     # Evaluation scripts
│   └── README.md
│
└── README.md                   # This file
```

---

## Component Documentation

Each component has its own detailed README:

- [Municode Web Crawler](municode-web-crawler/README.md) — Selenium scraping pipeline
- [Data Engineering](data-engineering/README.md) — PDF extraction on ECS Fargate
- [Pinecone Embedding](pinecone-embedding/README.md) — Hybrid embedding pipeline
- [RAG Query API](rag-query/README.md) — Flask API with GPU inference
- [Streamlit App](streamlit-app/README.md) — Interactive frontend
- [Evaluation](evaluation/README.md) — LLM-as-Judge quality assessment
