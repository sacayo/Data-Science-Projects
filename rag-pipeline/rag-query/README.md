# RAG Query API

> Production Flask REST API that retrieves relevant legal document chunks from Pinecone and generates LLM-powered answers using LLaMA 3.1 8B with 4-bit quantization — deployed on EC2 with GPU inference.

This is **Pipeline 3** in the RAG system. It exposes two search modes (baseline and hybrid), supports advanced filtering by location, legal classification, and readability metrics, and returns both LLM-generated answers and raw retrieval results.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Flask API (Port 8000)                          │
│                                                                     │
│  POST /query                                                        │
│  ┌─────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  Embed  │───▶│   Pinecone   │───▶│  Cross-Enc.  │               │
│  │  Query  │    │   Retrieve   │    │  Reranker    │               │
│  │         │    │   Top-100    │    │  → Top-5     │               │
│  └─────────┘    └──────────────┘    └──────┬───────┘               │
│                                             │                       │
│                 Baseline Mode:              │  Hybrid Mode:         │
│                 Dense only → Top-5          │  Dense+Sparse →       │
│                 (2-5s latency)              │  Rerank → Top-5       │
│                                             │  (5-10s latency)      │
│                                             ▼                       │
│                                      ┌──────────────┐               │
│                                      │  LLaMA 3.1   │               │
│                                      │  8B Instruct  │               │
│                                      │  (4-bit NF4)  │               │
│                                      └──────┬───────┘               │
│                                             │                       │
│                                     JSON Response                   │
│                                  (answer + chunks + meta)           │
└─────────────────────────────────────────────────────────────────────┘
```

### Search Modes

| Mode | Retrieval | Latency | Best For |
|------|-----------|---------|----------|
| **Baseline** | Dense embedding → top-5 | 2-5s | Quick lookups, specific questions |
| **Hybrid** | Dense + sparse → top-100 → rerank to top-5 | 5-10s | Complex queries, cross-county comparison |

### Filter Support

- **Location**: State and county filtering
- **Legal classification**: Penalty, obligation, permission, prohibition
- **Readability metrics**: Flesch-Kincaid grade, reading ease, word count, complexity percentage

---

## Quick Start

### Docker (Recommended)

```bash
cd rag-pipeline/rag-query

# Configure environment
cp .env.example .env
# Add PINECONE_API_KEY and HF_TOKEN to .env

# Build and run
./build.sh
./run.sh
# or: docker compose up -d

# Verify
curl http://localhost:8000/health
```

### Direct Python

```bash
pip install -r requirements.txt
cp .env.example .env
# Add PINECONE_API_KEY and HF_TOKEN to .env
export $(cat .env | xargs)
python api.py
```

---

## API Usage

### Query Endpoint

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Are dogs allowed in public parks?",
    "filters": {
      "locations": [
        {"state": "ca", "county": ["alameda-county", "butte-county"]}
      ],
      "penalty": "Y",
      "fk_grade": {"min": 5.0, "max": 50.0}
    },
    "mode": "hybrid"
  }'
```

### JSON Input (CLI)

```bash
python main.py --mode hybrid --json query.json
python main.py --mode baseline --query "leash laws in Georgia"
python main.py --example  # Run with built-in example
```

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `PINECONE_API_KEY` | Yes | Pinecone API key |
| `HF_TOKEN` | Yes | HuggingFace token (for LLaMA model access) |

### Model Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| LLM | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Generation model |
| Quantization | 4-bit NF4 | BitsAndBytes quantization config |
| Reranker | `ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| Top-K (baseline) | 5 | Direct retrieval count |
| Top-K (hybrid) | 100 | Candidates before reranking |

### GPU Requirements

- **Recommended**: `g4dn.xlarge` (NVIDIA T4, 16GB VRAM)
- **Minimum VRAM**: 12GB (with 4-bit quantization)
- **Cost**: ~$0.53/hour on-demand

---

## Output

### API Response

```json
{
  "response": "LLM-generated answer based on retrieved context...",
  "results": [
    {
      "id": "ca_alameda_ordinance_42",
      "score": 0.89,
      "rerank_score": 0.95,
      "state": "ca",
      "county": "alameda-county",
      "section": "5.08.010",
      "chunk_text": "It shall be unlawful..."
    }
  ]
}
```

### CSV Export

Results are also saved to `outputs/` as CSV files with full metadata.

---

## Project Structure

```
rag-query/
├── api.py                 # Flask REST API entry point
├── pipeline.py            # RAG pipeline orchestration
├── models.py              # LLM + reranker loading (4-bit quantization)
├── retrieval.py           # Pinecone retrieval (baseline + hybrid)
├── llm_generation.py      # Prompt engineering + generation
├── filters.py             # Filter processing utilities
├── config.py              # Configuration constants
├── utils.py               # Shared utilities
├── main.py                # CLI entry point
├── Dockerfile             # CUDA 12.1 + Python 3.10
├── docker-compose.yml     # GPU device mapping
├── build.sh / run.sh      # Docker helper scripts
├── .env.example           # Environment variable template
├── example_query.json     # Example query format
└── Documentation/
    ├── EC2_SETUP.md       # EC2 deployment guide
    ├── QUICKSTART.md      # 5-minute quick start
    └── DEPLOYMENT_CHECKLIST.md
```

## EC2 Deployment

See [EC2_SETUP.md](Documentation/EC2_SETUP.md) for full deployment instructions including Docker, NVIDIA Container Toolkit, and security group configuration.

## Downstream

The API is consumed by the [Streamlit Frontend](../streamlit-app/README.md) and evaluated by the [Evaluation Framework](../evaluation/README.md).
