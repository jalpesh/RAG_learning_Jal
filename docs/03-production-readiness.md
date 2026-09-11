# Production readiness review

Reviewed: 2026-09-11

## Executive assessment

This is a strong single-machine RAG lab, not yet a production RAG service.
Its best qualities are unusually good for a learning repository: measured
latency spans, a fixed question set, hybrid retrieval, reranking experiments,
source citations, explicit refusal cases, two model lanes, and confidence-led
escalation. The design is evidence-led rather than framework-led.

The largest gap is operational durability. The vector index, jobs, semantic
cache, and ingest state are process-local; a restart loses them, concurrent
workers disagree, and a corpus change requires rebuilding everything. That is
acceptable for an offline lab and is the first boundary to replace for a real
deployment.

## What already matches real-world practice

| Area | Current implementation | Assessment |
|---|---|---|
| Retrieval | Normalized dense vectors; optional BM25 + reciprocal-rank fusion; optional reranker | Sound baseline; hybrid/multi-stage retrieval is a standard production pattern. |
| Evaluation | Fixed factual, multihop, aggregation, and unanswerable questions; latency spans and saved results | Strong foundation. Preserve it as a release gate. |
| Grounding | Page-level sources; low-confidence checks; deeper-model escalation | Useful defense-in-depth, but heuristic rather than a complete faithfulness measure. |
| Performance | Resident embedding model, streaming generation, measured TTFT, model-size comparison | Appropriate for local inference and backed by actual measurements. |
| Privacy | Corpus and model files ignored by Git; chunks omitted from API responses | Correct default for the sensitive local corpus. |

## Gaps, ordered by delivery risk

### P0 — required before exposing real data

1. **Durable, tenant-aware storage.** Persist documents, chunk lineage,
   embeddings, index version, and ingest status. Use SQLite for a single-node
   deployment; move to Postgres plus pgvector or a vector database only when
   concurrency/scale measurements justify it. Every cache entry must include
   the corpus/index version.
2. **Security boundary.** Fail startup when binding beyond loopback without an
   API key; authenticate `/health` if it reveals operational state; add TLS at
   the reverse proxy, request-size limits, audit logs, and document-level ACL
   filters applied during retrieval—not after generation.
3. **Prompt-injection and data-exfiltration tests.** Treat retrieved documents
   as untrusted text. Add adversarial documents/questions and enforce that
   document instructions cannot override the system policy or reveal chunks a
   caller is not authorized to retrieve.

### P1 — required for a reliable service

1. **Incremental, idempotent ingest.** Hash source content, upsert changed
   chunks, delete stale chunks, retain source/page/offset lineage, and publish a
   new index atomically. Current `/ingest` rebuilds the full in-memory index.
2. **Durable bounded work queues.** Replace the in-memory `jobs` dictionary and
   background tasks with a bounded queue, timeouts, cancellation, retry policy,
   and persisted state. Add cache TTL/LRU limits and single-flight suppression
   for duplicate questions.
3. **Concurrency controls.** Add per-model semaphores/backpressure and load
   tests. Multiple app workers currently create independent indexes/caches and
   can overload the same llama servers.
4. **Modern application lifecycle.** Move model/index initialization from the
   deprecated FastAPI startup event to an application lifespan handler, and
   close HTTP/model resources on shutdown.
5. **Observability.** Export structured traces, metrics, and logs with request,
   corpus-version, retrieval, model, token, cache, queue, and error attributes.
   Define SLOs for availability, TTFT, answer completion, and groundedness.

### P2 — improves answer quality and maintainability

1. **Separate retrieval and answer evaluation.** Track recall@k/MRR or nDCG
   against known relevant chunks, plus answer correctness, citation precision,
   citation completeness, refusal precision/recall, and latency/cost. The
   current manual answer review is valuable but not yet a CI release gate.
2. **Structure-aware parsing/chunking.** Preserve headings, tables, lists, page
   spans, and document identity; add OCR for scanned pages. Fixed token windows
   are a baseline but lose semantic structure.
3. **Query strategy.** Add query rewriting/decomposition only for measured
   failure modes. Calibrate `top_k`, fusion, filtering, reranking, and routing
   per task type using held-out data.
4. **API contracts.** Validate nonblank/max-length questions, return typed job
   schemas, include stable document/chunk citation IDs, expose readiness
   separately from liveness, and version the API.

## Recommended next milestone

Build a repeatable “single-node production” slice before adding a vector DB:

1. SQLite manifest and index versioning with atomic incremental ingest.
2. Lifespan-managed startup/shutdown plus bounded cache and job execution.
3. Retrieval ground-truth labels and automated quality metrics in CI.
4. Authorization filters and prompt-injection regression cases.
5. Load test at the intended concurrency and publish p50/p95/p99 TTFT,
   throughput, error rate, and grounded-answer rate.

This keeps the deployment inexpensive and local while addressing correctness
and reliability first. Introduce pgvector/Qdrant, Redis, or a separate worker
only after measurements show the single-node boundary is the bottleneck.

## Reference points

- FastAPI recommends lifespan handlers over the older startup/shutdown event
  decorators: <https://fastapi.tiangolo.com/advanced/events/>
- Qdrant documents hybrid and multi-stage retrieval as first-class query
  patterns: <https://qdrant.tech/documentation/search/hybrid-queries/>
- OpenAI's knowledge-retrieval starter separates configuration, ingest,
  retrieval evaluation, curated datasets, and reports:
  <https://github.com/openai/openai-knowledge-retrieval>
- OpenAI's evaluation guidance recommends failure taxonomy, human-reviewed
  examples, calibrated graders, held-out tests, and continuous production
  feedback: <https://github.com/openai/openai-cookbook/blob/main/examples/evaluation/Building_resilient_prompts_using_an_evaluation_flywheel.md>
