# Tesla Manual QA

A production-grade RAG question-answering system built on the Tesla Model 3 owner's manual. It combines semantic chunking, hybrid retrieval, cross-encoder reranking, and LLM generation to answer questions about vehicle operation, maintenance, and safety — grounded strictly in the manual's content.

## Highlights

- **Multi-stage RAG pipeline**: semantic chunking → hybrid retrieval (BM25 + TF-IDF + FAISS + Milvus) → cross-encoder reranking → grounded generation
- **Multiple retrieval backends**: BM25, TF-IDF, FAISS (with BGE/Qwen embeddings), Milvus (with BGEM3 hybrid embedding)
- **Multiple rerankers**: BGE-M3, Qwen3 (native and vLLM-served)
- **Streaming inference server**: FastAPI + SSE with semantic chunking service
- **SFT data generation**: build instruction-tuning data from manual QA pairs for LoRA fine-tuning
- **Evaluation harness**: RAGAS-based scoring (context recall / context precision)

## Architecture

```
                 ┌─────────────────────────────────────────────┐
                 │               Tesla Manual PDF              │
                 └──────────────────────┬──────────────────────┘
                                        ▼
                 ┌─────────────────────────────────────────────┐
                 │              Parser & Chunker               │
                 │   pdf_parse.py ── semantic_chunk.py ──────  │
                 └──────────────────────┬──────────────────────┘
                                        ▼
                 ┌─────────────────────────────────────────────┐
                 │         Indexing & Retrieval Layer          │
                 │  BM25 │ TF-IDF │ FAISS │ Milvus             │
                 └──────────────────────┬──────────────────────┘
                                        ▼
                 ┌─────────────────────────────────────────────┐
                 │               Reranking Layer               │
                 │        BGE-M3 │ Qwen3 │ Qwen3-vLLM          │
                 └──────────────────────┬──────────────────────┘
                                        ▼
                 ┌─────────────────────────────────────────────┐
                 │            Generation & Server              │
                 │   LLM chat client │ FastAPI + SSE           │
                 └─────────────────────────────────────────────┘
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Copy `config.ini` and set your LLM API credentials:

```bash
export DOUBAO_API_KEY="<your-api-key>"
export DOUBAO_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
export DOUBAO_MODEL_NAME="<your-model-name>"
```

### 3. Start the semantic chunking server

```bash
nohup python src/server/semantic_chunk.py > log/semantic_chunk.log 2>&1 &
```

### 4. Build the index

```bash
python build_index.py
```

### 5. Run inference

```bash
python infer.py --query "How do I maximize driving range?"
```

## Data

- `data/qa_pairs/` — question-answer pairs generated from the manual
- `data/rerank_data/` — query-document pairs for reranker training
- `data/processed_docs/` — preprocessed chunked documents
- `data/saved_index/` — built retrieval indexes

A small sample is included under `examples/`. Full manual PDF and generated datasets are loaded at runtime via paths in `src/constant.py`.

## Project Layout

```
├── src/
│   ├── client/          # LLM / MongoDB / chunking service clients
│   ├── fields/          # manual metadata extraction
│   ├── gen_qa/          # QA pair generation
│   ├── parser/          # PDF parsing and image handling
│   ├── reranker/        # BGE-M3 / Qwen3 rerankers
│   ├── retriever/       # BM25 / TF-IDF / FAISS / Milvus retrievers
│   └── server/          # semantic chunking inference server
├── build_index.py       # build retrieval indexes
├── generate_sft_data.py # generate SFT data from QA pairs
├── infer.py             # run end-to-end inference
├── final_score.py       # RAGAS evaluation
└── examples/            # sample QA data
```

## Evaluation

```bash
python final_score.py
```

Uses [RAGAS](https://docs.ragas.io/) metrics: LLM context recall and context precision with reference.

## License

[MIT](LICENSE)

## Notes

- Indexes and raw manuals are loaded at runtime and are not committed to the repository.
- The pipeline is designed to be backend-agnostic: swap the retriever and reranker via configuration for different latency/quality trade-offs.

