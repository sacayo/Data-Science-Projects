# Evaluation Framework

> LLM-as-a-Judge evaluation system that measures retrieval quality by comparing retrieved legal documents against ground truth using NVIDIA Nemotron — assessing recall, ranking, coverage, and metadata accuracy.

This is **Component 5** in the RAG system. It queries the [RAG Query API](../rag-query/README.md) with a curated test dataset, uses an LLM judge to assess retrieval quality, and produces comprehensive metrics including negative test handling (correctly identifying when no law exists).

---

## How It Works

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Gold Dataset    │     │  RAG Query API   │     │  LLM Judge       │     │  Metrics         │
│                  │     │                  │     │                  │     │                  │
│  Questions +     │────▶│  Retrieve chunks │────▶│  NVIDIA Nemotron │────▶│  Recall, MRR,    │
│  Golden answers  │     │  from Pinecone   │     │  (via NIMs API)  │     │  Coverage,       │
│  + Sections      │     │                  │     │  Compare vs gold │     │  Metadata Acc.   │
│  + Neg. tests    │     │                  │     │                  │     │  Composite Score │
└──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
```

### Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| **Retrieval** | Top-5 Recall | % of queries where correct law appears in top 5 |
| **Retrieval** | MRR | Mean Reciprocal Rank — average of 1/rank for first correct result |
| **Retrieval** | Chunk Coverage | Completeness of retrieved chunk vs golden truth (0-100%) |
| **Metadata** | Penalty/Fine Accuracy | Correct identification of monetary penalties |
| **Metadata** | Prohibition Accuracy | Correct identification of prohibited actions |
| **Metadata** | Obligation Accuracy | Correct identification of mandatory requirements |
| **Metadata** | Permission Accuracy | Correct identification of granted authorizations |
| **Negative** | True Negatives | System correctly says "no law exists" |
| **Negative** | False Positives | System incorrectly claims a law exists |

### Composite Score

Weighted combination: **30% Recall + 30% MRR + 20% Coverage + 20% Metadata Accuracy**

---

## Quick Start

```bash
cd rag-pipeline/evaluation
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run evaluation (requires RAG Query API to be running)
python legal_retrieval_evaluator.py \
    -i eval_dataset_final.csv \
    -o evaluation_results.csv \
    -s evaluation_summary.json \
    -m hybrid
```

---

## Configuration

### CLI Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | `eval_dataset_final.csv` | Gold dataset (CSV or Excel) |
| `--output` | `-o` | `evaluation_results.csv` | Per-query results output |
| `--summary` | `-s` | `evaluation_summary.json` | Summary metrics output |
| `--limit` | `-l` | None | Limit number of queries (for testing) |
| `--delay` | — | `1.0` | Delay between API calls (seconds) |
| `--mode` | `-m` | `hybrid` | `hybrid` or `baseline` |

### API Endpoints (in `legal_retrieval_evaluator.py`)

```python
RETRIEVAL_ENDPOINT = "http://your-api-endpoint:8000/query"
NIMS_API_KEY = "your-nvidia-nims-api-key"
NIMS_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL_NAME = "nvidia/llama-3.1-nemotron-nano-8b-v1"
```

---

## Input Dataset Format

| Column | Description |
|--------|-------------|
| `State` | State code (e.g., "CA", "GA") |
| `County` | County name (e.g., "Alameda") |
| `Question` | Legal query to evaluate |
| `Answer` | Golden truth law text (or `NO_LAW_EXISTS` for negative tests) |
| `Section` | Section reference (e.g., "5.08.010") or `N/A` for negative tests |
| `Difficulty Column` | Easy, Medium, or Hard |

---

## Output

### Per-Query Results (`evaluation_results.csv`)

Each row includes query metadata, golden truth, retrieved results, match status, metadata flag comparisons, and LLM reasoning.

### Summary (`evaluation_summary.json`)

Aggregated metrics broken down by difficulty level, with the composite score.

```bash
# Quick test with 10 queries
python legal_retrieval_evaluator.py --limit 10 --delay 0.5

# Compare modes
python legal_retrieval_evaluator.py -m baseline -o results_baseline.csv -s summary_baseline.json
python legal_retrieval_evaluator.py -m hybrid -o results_hybrid.csv -s summary_hybrid.json
```

---

## Dependencies

- Python 3.10+
- pandas, requests, tqdm, openpyxl
- Requires the [RAG Query API](../rag-query/README.md) to be running
- Requires NVIDIA NIMs API key for the Nemotron judge model
