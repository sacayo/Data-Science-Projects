# RAG QA System Evaluation

> A systematic evaluation framework for Retrieval-Augmented Generation systems — testing 12 configurations across embedding models, LLMs, chunking strategies, and audience types to find the optimal RAG pipeline.


Built as a project for UC Berkeley MIDS DATASCI 290 (Generative AI).

---

## 01. The Problem

### Context

RAG systems have a dirty secret: they can look impressive in demos while quietly producing unreliable results. A chatbot that answers confidently but retrieves irrelevant context — or hallucinates details not present in the source documents — is worse than no system at all, because users trust it.

This project addresses a fundamental question in production RAG: **how do you know your retrieval pipeline is actually working?** And more importantly, how do you systematically compare different configurations to find the one that best serves your users?

### Challenge

We were tasked with building and evaluating a RAG system for a tech company with two distinct user groups:
- **300 engineers** who need detailed, technical answers about LLMs, RAG architectures, RLHF, and model training
- **40 marketing staff** who need clear, concise explanations of the same topics for customer-facing materials

The system needed to:
- Ingest documents from multiple sources (ArXiv papers, Wikipedia, technical blogs)
- Answer questions using retrieved context + LLM generation
- Produce audience-appropriate responses (technical depth for engineers, clarity for marketing)
- Be systematically evaluated against a gold-standard dataset of 75 validated question-answer pairs
- Identify which combination of embedding model, LLM, chunking strategy, and retrieval parameters produces the best results

The core insight: there's no single "best" RAG configuration. The optimal setup depends on your audience, your corpus, and which quality dimensions you prioritize.

---

## 02. The Approach

### Evaluation Framework Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    RAG Evaluation Pipeline                           │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                     Document Corpus                            │  │
│  │  ┌──────────┐  ┌──────────────┐  ┌────────────────────────┐  │  │
│  │  │  ArXiv   │  │  Wikipedia   │  │  Technical Blogs       │  │  │
│  │  │ 23 papers│  │  3 articles  │  │  5 blog posts          │  │  │
│  │  │ (RAG,    │  │ (GenAI, IR,  │  │  (Lilian Weng:         │  │  │
│  │  │  RLHF,   │  │  LLMs)       │  │   QA, Prompts,         │  │  │
│  │  │  LoRA...)│  │              │  │   Attention, Agents)   │  │  │
│  │  └──────────┘  └──────────────┘  └────────────────────────┘  │  │
│  └────────────────────────┬───────────────────────────────────────┘  │
│                           │                                          │
│                           ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │              Configurable Pipeline Components                   │ │
│  │                                                                 │ │
│  │  Chunking ──▶ Embedding ──▶ Vector Store ──▶ Retriever ──▶ LLM │ │
│  │                                                                 │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────┐  ┌────────┐│ │
│  │  │Recursive │  │multi-qa  │  │          │  │k=5 │  │Mistral ││ │
│  │  │(128,256) │  │all-mpnet │  │  Qdrant  │  │k=10│  │7B      ││ │
│  │  │Semantic  │  │MiniLM    │  │(in-mem)  │  │    │  │Cohere  ││ │
│  │  │(Unstruct)│  │distilrob │  │          │  │    │  │        ││ │
│  │  │          │  │GIST      │  │          │  │    │  │        ││ │
│  │  └──────────┘  └──────────┘  └──────────┘  └────┘  └────────┘│ │
│  └────────────────────────┬────────────────────────────────────────┘ │
│                           │                                          │
│                           ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                   Evaluation Layer                              │ │
│  │                                                                 │ │
│  │  Gold Dataset ──▶ RAGAS Metrics ──▶ Weighted Score ──▶ Ranking │ │
│  │  (75 questions,    (Semantic Sim,    (0.4 SS + 0.4 BS          │ │
│  │   dual answers)     BERTScore,         + 0.2 ROUGE)            │ │
│  │                     ROUGE-L)                                    │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### A/B Testing Matrix

We tested 12 distinct configurations across 5 tunable dimensions:

```
                    ┌─────────────────────────────────────────────┐
                    │           Configuration Space               │
                    │                                             │
                    │  Embedding Models (5)                       │
                    │  ├── multi-qa-mpnet-base-dot-v1  ◀── best  │
                    │  ├── all-mpnet-base-v2                      │
                    │  ├── all-MiniLM-L6-v2                       │
                    │  ├── all-distilroberta-v1                   │
                    │  └── avsolatorio/GIST-Embedding-v0          │
                    │                                             │
                    │  LLMs (2)                                   │
                    │  ├── Mistral 7B Instruct (open-source)      │
                    │  └── Cohere (proprietary API)               │
                    │                                             │
                    │  Chunking Strategies (2)                    │
                    │  ├── RecursiveCharacter (fixed-size, 128)   │
                    │  └── Unstructured.io (semantic, by-title)   │
                    │                                             │
                    │  Retrieval Depth (2)                        │
                    │  ├── k=5 (concise context)                  │
                    │  └── k=10 (comprehensive context)           │
                    │                                             │
                    │  Audience (2)                               │
                    │  ├── Research (technical prompt)             │
                    │  └── Marketing (business prompt)            │
                    └─────────────────────────────────────────────┘
```

### Evaluation Metrics

We used a three-metric weighted scoring system to evaluate each configuration:

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| **Semantic Similarity** | 40% | Embedding-space alignment between generated and gold answers |
| **BERTScore F1** | 40% | Contextual token-level similarity using BERT embeddings |
| **ROUGE-L** | 20% | Longest common subsequence overlap (surface-level match) |

**Combined Score** = `0.4 × SemanticSimilarity + 0.4 × BERTScore + 0.2 × ROUGE-L`

We weighted ROUGE-L lower because surface-level word overlap is a weaker signal for answer quality than semantic alignment. Two answers can convey the same meaning with different wording — semantic similarity and BERTScore capture this, ROUGE doesn't.

### Chunking Strategy Comparison

**Fixed-Size (RecursiveCharacterTextSplitter)**:
- Splits at `\n\n`, `\n`, space, or character boundaries
- Consistent 128-token chunks, no overlap
- Fast and predictable
- Risk: splits coherent concepts across chunk boundaries

**Semantic (Unstructured.io, by-title strategy)**:
- Groups content by section headers (Introduction, Methods, Results, etc.)
- Variable chunk sizes based on document structure
- Preserves semantic coherence within chunks
- Best suited for structured documents like academic papers
- **Result**: 8-12% improvement in answer coherence over fixed-size chunking

### RAG Chain Architecture

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template    # Audience-specific (research vs marketing)
    | llm                # Mistral 7B or Cohere
    | StrOutputParser()
)
```

**Advanced Pipeline Components Tested**:
- **Query Rewriting**: Safety checks and malicious intent detection before retrieval
- **Embedding Filters**: Filter retrieved chunks by relevance threshold before passing to LLM
- **LLM Re-ranking**: Use a second LLM call to re-order retrieved chunks by relevance
- **Compression Retriever**: Reduce chunk content to only the most relevant portions

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | LangChain | Chain composition, retriever setup |
| **Embeddings** | HuggingFace sentence-transformers | 5 embedding models tested |
| **Open-Source LLM** | Mistral 7B Instruct v0.2 | 4-bit quantized, GPU inference |
| **Proprietary LLM** | Cohere (ChatCohere API) | Polished, safety-filtered outputs |
| **Vector Store** | Qdrant (in-memory) | Similarity search with configurable retrieval |
| **Semantic Chunking** | Unstructured.io | Title-based document segmentation |
| **Evaluation** | RAGAS, BERTScore | Automated metrics against gold dataset |
| **Compute** | Google Colab Pro (T4 GPU) | Model inference and evaluation |
| **Quantization** | BitsAndBytes 4-bit | Mistral 7B on T4 GPU |

---

## 03. Results

### Configuration Rankings

| Rank | Configuration | Weighted Score |
|------|--------------|----------------|
| 1 | Unstructured + Mistral (Marketing) | **0.800** |
| 2 | Unstructured + Cohere (Marketing) | 0.775 |
| 3 | all-mpnet + Mistral (Marketing) | 0.700 |
| 4 | Baseline + Cohere (Marketing) | 0.700 |
| 5 | Baseline + Mistral (Marketing) | 0.650 |
| 6 | all-mpnet + Cohere (Marketing) | 0.630 |

### Key Findings

**Semantic chunking was the single biggest lever.** Switching from fixed-size (128 tokens) to semantic chunking (Unstructured.io, by-title) improved the weighted score by 8-12% across all configurations. For academic papers with clear section structure, semantic chunking preserves the context that fixed-size chunking destroys.

**Embedding model choice matters, but less than chunking.** multi-qa-mpnet-base-dot-v1 consistently outperformed alternatives for QA tasks, but the gap between the best and worst embedding model (~0.07) was smaller than the gap between chunking strategies (~0.10).

**LLM choice is audience-dependent.** Cohere produced more polished, safer outputs that suited marketing use cases. Mistral 7B gave more detailed, technical responses better suited for engineering teams. Neither was universally better — the right choice depends on who's reading.

**Retrieval depth (k) trades conciseness for completeness.** k=5 produced focused, concise answers (better for marketing). k=10 retrieved more context, enabling more comprehensive answers (better for research). More context also means more noise, so the optimal k depends on document quality and question complexity.

### Best Configurations by Audience

| Audience | Best Config | Semantic Sim | BERTScore | ROUGE-L |
|----------|-------------|--------------|-----------|---------|
| **Marketing** | Cohere + multi-qa-mpnet + k=5 + semantic chunking | 0.84 | 0.89 | 0.52 |
| **Research** | Mistral 7B + multi-qa-mpnet + k=10 + semantic chunking | 0.82 | 0.87 | 0.54 |

---

## Technical Decisions & Trade-offs

### Why a weighted evaluation metric instead of a single metric?

No single metric captures RAG quality comprehensively. Semantic similarity measures meaning alignment but can miss factual details. ROUGE captures word overlap but misses paraphrases. BERTScore provides contextual similarity but can be noisy. The weighted combination (0.4/0.4/0.2) balances these perspectives, with ROUGE weighted lower because surface-level overlap is the weakest signal for answer quality.

### Why Qdrant in-memory instead of a persistent vector store?

This is an evaluation project, not a production system. In-memory Qdrant lets us quickly rebuild the index with different chunking parameters without managing a persistent database. Each experiment starts fresh, ensuring no cross-contamination between configurations.

### Why dual audience prompts instead of one generic prompt?

Engineers and marketing staff have fundamentally different needs. A prompt that produces good engineering answers (detailed, technical, includes caveats) produces poor marketing answers (too verbose, too technical). Separate prompt templates with audience-specific instructions (e.g., "use technical terminology" vs "explain in business terms") improved scores by 8-12% compared to a generic prompt.

### Why both open-source and proprietary LLMs?

Testing Mistral 7B (local, controllable, no API costs) against Cohere (cloud, polished, safety-filtered) reveals practical trade-offs for deployment. Open-source gives full control but requires GPU infrastructure. Proprietary APIs are simpler to deploy but add per-call costs and API dependency.

---

## Lessons Learned

**Chunking strategy has outsized impact.** We expected the LLM choice to dominate results. Instead, how you chunk documents matters more than which model generates the answer. Fixed-size chunking at 128 tokens frequently splits a concept across two chunks, meaning the retriever finds half the answer. Semantic chunking keeps concepts together.

**Evaluation is harder than building.** Building a RAG chain takes hours. Building a reliable evaluation framework that measures what you actually care about takes weeks. Defining the gold dataset, choosing metrics, weighting them appropriately, and running experiments across 12 configurations was the majority of the project's effort.

**Temperature tuning had surprisingly little effect.** We expected significant quality differences across temperature settings (0.1, 0.7, 1.0). In practice, with good prompts, the temperature had minimal impact on answer quality for factual questions. This suggests that prompt engineering dominates temperature for retrieval-grounded generation.

**Out-of-context handling is a feature, not a bug.** When the system correctly says "I don't have information about this topic" instead of hallucinating, that's a success. Our best configurations correctly identified out-of-scope questions (e.g., literary questions asked to a GenAI knowledge base) with >90% accuracy.

---

## Project Structure

```
RAG-QA-System-Evaluation/
├── Code/
│   ├── 2024_Fall_267_assignment_V.ipynb    # Full assignment notebook (~630 cells)
│   └── RAG_System_Evaluation_Portfolio.ipynb # Portfolio version (cleaner)
├── assets/
│   └── model-evalution-graph.png            # Configuration comparison chart
├── Final-report/
│   └── RAGAS_Evaluation_report.pdf          # Comprehensive evaluation report
└── README.md                                # This file
```
