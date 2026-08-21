# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release: RAG question-answering system over the Tesla Model 3 owner's manual
- Semantic chunking service, hybrid retrieval (BM25 / TF-IDF / FAISS / Milvus), cross-encoder reranking (BGE-M3 / Qwen3)
- Streaming inference server with FastAPI + SSE
- SFT data generation and RAGAS-based evaluation harness
