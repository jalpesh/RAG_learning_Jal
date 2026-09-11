# Status

Date: 2026-09-11

The measured local RAG lab is functional and repository-ready. Core offline
checks pass. Setup drift was corrected by making the default embedding runtime
a default dependency and replacing the deprecated PyMuPDF import.

The README now documents system installation, required Ollama pulls, optional
Ollama embeddings, llama.cpp GGUF naming, backend startup, benchmark commands,
authless local API startup, a one-command corpus re-ingest, and offline
verification. API-key authentication remains an optional deployment control.

Ollama provisioning is deterministic through `scripts/setup_ollama.sh`.
Preflight now fails when the server or required Qwen3 models are missing, and
tests protect the per-request `keep_alive=-1s` pinned configuration.

Production readiness: lab-grade. The prioritized gap analysis is in
`docs/03-production-readiness.md`; durable state, security controls, bounded
concurrency, observability, and automated quality gates remain open.
