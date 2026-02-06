# Pinecone Embedding Pipeline

> Generates hybrid embeddings (dense + sparse) from pre-chunked legal documents and indexes them into Pinecone — enabling the hybrid search that drives the entire retrieval system.

This is **Pipeline 2** in the RAG system. It reads Parquet files from S3, generates both dense semantic embeddings and sparse keyword embeddings via Pinecone Inference, and upserts the vectors with rich metadata into a Pinecone serverless index.

---

## How It Works

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   S3 Parquet     │     │   Dense Embed    │     │   Vector         │     │   Pinecone       │
│                  │     │                  │     │   Construction   │     │   Serverless     │
│  Chunked text    │────▶│  llama-text-     │────▶│                  │────▶│                  │
│  + metadata      │     │  embed-v2        │     │  ID + dense +    │     │  1024-dim dense  │
│                  │     │  (1024-dim)      │     │  sparse + meta   │     │  + sparse BM25   │
│                  │     │                  │     │                  │     │  + metadata      │
│                  │     │   Sparse Embed   │     │                  │     │                  │
│                  │     │  pinecone-sparse │     │                  │     │                  │
│                  │     │  -english-v0     │     │                  │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
```

The dual embedding approach is the key architectural decision: dense vectors capture semantic meaning ("rules about animals in public spaces"), while sparse vectors capture keyword matches ("leash fine penalty"). Both live in the same Pinecone index, enabling hybrid search with a single query.

---

## Quick Start

```bash
cd rag-pipeline/pinecone-embedding
uv sync

# Create .env with your Pinecone API key
echo "PINECONE_API_KEY=pc_sk_..." > .env

# Ingest a single Parquet file
uv run python src/rag_ingest/ingest.py \
    --index-name "rag-prod-index" \
    --bucket "rag-data-lake" \
    --single-key "processed/ca_alameda_chunks.parquet" \
    --metadata-cols county state url

# Ingest an entire S3 prefix
uv run python src/rag_ingest/ingest.py \
    --index-name "rag-prod-index" \
    --bucket "rag-data-lake" \
    --prefix "processed/zone=text_chunk/"
```

---

## Configuration

### CLI Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--index-name` | Yes | Pinecone index name (auto-created if missing) |
| `--bucket` | Yes | S3 bucket with source Parquet files |
| `--prefix` | No | S3 prefix to ingest all `.parquet` files from |
| `--single-key` | No | Specific S3 key for single file ingestion |
| `--metadata-cols` | No | Columns to attach as metadata (default: all) |

### Environment Variables

| Variable | Source | Description |
|----------|--------|-------------|
| `PINECONE_API_KEY` | `.env` file | Pinecone API key |
| `AWS_ACCESS_KEY_ID` | `~/.aws/credentials` | AWS credentials (auto-detected by boto3) |
| `AWS_SECRET_ACCESS_KEY` | `~/.aws/credentials` | AWS credentials (auto-detected by boto3) |

---

## Project Structure

```
pinecone-embedding/
├── src/rag_ingest/
│   ├── ingest.py          # Main orchestration
│   ├── s3_loader.py       # S3 → Pandas DataFrame
│   ├── pinecone_setup.py  # Index creation/connection
│   ├── embed_dense.py     # Dense embedding via Pinecone Inference
│   ├── embed_sparse.py    # Sparse embedding via Pinecone Inference
│   └── upsert.py          # Vector construction & batch upload
├── tests/                 # Unit tests
├── pyproject.toml         # Dependencies (managed by uv)
└── README.md
```

## Testing

```bash
uv run python -m unittest discover tests
```

## Next Step

After indexing, deploy the [RAG Query API](../rag-query/README.md) to search against the Pinecone index.
