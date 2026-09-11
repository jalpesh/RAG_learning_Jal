# Decisions

## 2026-09-11 — Keep the repository local-first

Retain the exact in-memory index and local llama.cpp model servers for the lab.
Do not add a vector database until measured scale or concurrency requires it.
The next reliability step is SQLite-backed manifest/version state because it
adds restart safety without an unnecessary service or hosting cost.

## 2026-09-11 — Make the default runtime installable by default

`Config.embed_backend` defaults to sentence-transformers and the service uses
that path, so `sentence-transformers` is a core dependency rather than an
optional extra. `uv sync --locked` is the canonical setup command.
