# Local RAG vs Cloud RAG — Latency Engineering Lab

**Target machine:** Apple M3 Pro, 18 GB unified memory
**Goal:** understand *exactly* where the seconds go, then close the gap
**Constraint:** fully offline. Cloud is a measured baseline only, never a dependency.

---

## 1. The question, restated properly

"Online RAG answers in 3 seconds, my local RAG takes 3 minutes" is not one problem.
It is **eight** problems stacked in a pipeline, and they fail for different reasons.

RAG has two pipelines, not one. Conflating them is the first mistake.

**Ingest pipeline** (runs once per document):
```
parse -> chunk -> embed(N chunks) -> index
```

**Query pipeline** (runs once per question):
```
embed(1 query) -> ANN search -> rerank -> prompt assembly -> LLM generate
```

Cloud feels fast because it optimises these two *separately* and hides ingest
behind an upload progress bar. Local naive implementations run both cold,
sequentially, single-threaded, on CPU, per request.

---

## 2. Where the time actually goes

Reference workload: one 50-page PDF (~450 chunks of 512 tokens), one question,
~400-token answer.

### Cloud (typical hosted RAG)

| Stage | Time | Why it's fast |
|---|---|---|
| Parse 50 pages | 0.3-1.0 s | Native parser, pages fanned across workers |
| Chunk | ~0 ms | Pure string ops |
| Embed 450 chunks | 0.5-2.0 s | **One batched call** to a warm GPU fleet, batch 256+, fp16. Aggregate throughput 100k+ tok/s |
| Index 450 vectors | ~10 ms | In-memory, hosted |
| **Ingest total** | **~1-3 s** | |
| Embed query | 30-80 ms | Network RTT dominates, not compute |
| ANN search | 5-20 ms | Trivial at this scale |
| Rerank 50 candidates | ~50 ms | Cross-encoder on GPU, batched |
| LLM time-to-first-token | 200-500 ms | Model permanently resident, prefix-cached |
| Stream 400 tokens | 3-8 s | 50-150 tok/s, but *streaming* so it feels instant |
| **Query total (to first word)** | **~0.4 s** | |

### Local, naive (LangChain/Chroma/Ollama defaults)

| Stage | Time | Why it's slow |
|---|---|---|
| Python cold start + `import torch` | 3-8 s | Paid on **every** CLI invocation |
| First-run model download | 30-180 s | 1.3 GB embedder + 4.5 GB LLM from HF/Ollama |
| Parse 50 pages | 1-60 s | `unstructured` in `hi_res` mode loads a detectron layout model. Cold: +30-60 s |
| Embed 450 chunks | **60-180 s** | **The #1 killer.** sentence-transformers on CPU, `batch_size=32`, fp32, no MPS, no `torch.no_grad` tuning |
| Chroma insert | 2-15 s | Per-batch sqlite/duckdb commits, HNSW graph built incrementally |
| **Ingest total** | **90-240 s** | |
| Embed query | 100-400 ms | Same CPU path, batch of 1 |
| ANN search | 5-50 ms | Fine |
| Rerank 50 | 5-20 s | Second cross-encoder cold-loaded, CPU, unbatched |
| LLM cold load | 5-20 s | Ollama default `keep_alive=5m`. Model evicted between queries -> **you pay this every single time** |
| LLM TTFT after load | 0.3-1.5 s | Prompt prefill of ~4k context tokens |
| Stream 400 tokens | 12-25 s | 7B q4_K_M on M3 Pro ~= 20-35 tok/s |
| **Query total (to first word)** | **10-40 s** | |

**The headline:** on a *cold, naive* local stack roughly **70% of the gap is not
compute at all** — it is cold starts, model loading, process restarts and
absent batching. Only the token generation rate is genuinely a hardware limit.

---

## 3. The fixes, ranked by return on effort

Ordered by (seconds saved) / (hours of work). Do them in this order.

### Tier 1 — free wins, ~80% of the gap

1. **Never restart the process.** Run the RAG as a resident FastAPI/uvicorn
   service holding the embedder in memory. A CLI that re-imports torch per
   query can never be fast. *Saves 5-10 s per invocation.*
2. **Pin the LLM in memory.** `OLLAMA_KEEP_ALIVE=-1`, or run `llama-server`
   from llama.cpp as a permanent daemon. *Saves 5-20 s per query.*
3. **Batch the embeddings.** Encode all 450 chunks in one call with
   `batch_size=64..256`, not a loop over chunks. *Saves 30-120 s per ingest.*
4. **Move embeddings to Metal.** `device="mps"` in sentence-transformers, or a
   GGUF embedder served by `llama-server --embedding`. M3 Pro GPU is 5-15x the
   CPU path here. *Saves 40-150 s per ingest.*

### Tier 2 — right-sizing the architecture

5. **Use a small, strong embedder.** `bge-small-en-v1.5` (33M params, 384-dim)
   or `nomic-embed-text-v1.5`. A 1B-param embedder buys you ~2 points of recall
   and costs 20x the latency. Wrong trade at this scale.
6. **Skip the ANN index below ~100k vectors.** Exact cosine over a numpy fp16
   matrix is sub-millisecond at 450 vectors and *has no build cost*. HNSW is a
   tax you pay on ingest to save time you don't need at query. Introduce
   FAISS/HNSW only when the corpus crosses six figures.
7. **Make the reranker earn its latency.** Cap candidates at 20, use
   `bge-reranker-base`, run it on MPS, keep it resident. If it costs more than
   ~150 ms, replace it with hybrid retrieval: BM25 + dense, fused with
   Reciprocal Rank Fusion. Near-free, and often beats a slow cross-encoder.
8. **Parse in parallel.** Page-level fan-out across a process pool. And do not
   use `hi_res` OCR unless the document actually needs it — try `fast` first
   and fall back only on low text yield.

### Tier 3 — feel and steady state

9. **Stream tokens.** Perceived latency is time-to-first-token, not total.
   A 12-second answer that starts in 300 ms feels faster than a 4-second answer
   that arrives as a block.
10. **Prefix-cache the system prompt.** llama.cpp caches the KV for a stable
    prefix. Put the system prompt and any fixed instructions first, the
    retrieved chunks after.
11. **Incremental ingest.** Hash each chunk; never re-embed unchanged text.
    Re-uploading an edited document should cost only the diff.
12. **Cap generation.** `max_tokens` and a prompt that instructs brevity are
    the cheapest latency lever you own.

---

## 4. Realistic target budget (M3 Pro 18 GB, warm)

| Stage | Target |
|---|---|
| Ingest 50-page PDF (parse+chunk+embed+index) | **3-8 s** |
| Query embed | 10-20 ms |
| Retrieve (exact, 450 vectors) | < 5 ms |
| Rerank top-20 | 60-120 ms |
| LLM time-to-first-token | 250-600 ms |
| Generation rate (8B q4_K_M) | 25-35 tok/s |
| **Perceived query latency** | **< 1 s to first word** |

That is within ~2x of a hosted service. The residual gap is pure GPU FLOPs and
you cannot buy it back with software — only with a smaller model or a bigger
machine.

---

## 5. Architecture

Three long-lived processes. Everything else is a client.

```
                    +---------------------------+
  documents ------> |  ingest worker            |
                    |  parse | chunk | hash-dedup|
                    +------------+--------------+
                                 | batched encode
                                 v
+------------------+    +--------+---------+    +------------------+
| llama-server     |<---| rag-service      |--->| llama-server     |
| --embedding      |    | FastAPI, resident |    | --chat (8B q4)   |
| bge-small GGUF   |    | numpy flat index  |    | Metal, keep-alive |
| Metal            |    | BM25 + RRF        |    | prefix cache     |
+------------------+    | reranker (MPS)    |    +------------------+
                        +--------+---------+
                                 |
                                 v
                        SSE token stream -> client
```

Storage: parquet for chunks + metadata, a single `.npy` fp16 matrix for
vectors, sqlite for the chunk-hash ledger. No vector database until the corpus
demands one. This is deliberate — a DB hides the very latency you are trying to
measure.

---

## 6. The measurement harness (this is the actual deliverable)

Understanding beats guessing. Every stage is wrapped in a span timer that emits
structured JSON:

```json
{"run":"local-v3","stage":"embed_chunks","n":450,"ms":2140,"cold":false,
 "device":"mps","batch":128,"tokens":230400}
```

`bench/` runs a fixed matrix over the same corpus and the same 25 questions:

- **Configs:** naive-cpu, batched-cpu, batched-mps, mps+resident, +hybrid,
  +prefix-cache
- **Metrics:** ingest wall time, tok/s embed throughput, p50/p95 query TTFT,
  generation tok/s, peak RSS, recall@5 against a hand-labelled answer key
- **Output:** a latency waterfall per config, plus a quality-vs-speed frontier

The frontier is the point. Every speed-up above costs something — recall,
answer quality, memory. The harness makes that trade visible instead of
vibes-based.

---

## 7. Stack decision

| Layer | Choice | Why |
|---|---|---|
| Runtime | Python 3.12 + `uv` | Fast resolves, reproducible lockfile |
| Serving | FastAPI + uvicorn, single worker | Resident models; workers would duplicate memory |
| Parse | pymupdf (fast) -> unstructured (fallback) | pymupdf is ~50x faster and handles most PDFs |
| Chunk | Token-aware recursive, 400 tok / 80 overlap | Overlap tuned in the harness, not guessed |
| Embed | bge-small-en-v1.5, GGUF via llama-server, Metal | 384-dim, tiny, strong; same runtime as the LLM |
| Index | numpy fp16 flat + rank_bm25, RRF fusion | Zero build cost, exact, trivially debuggable |
| Rerank | bge-reranker-base on MPS, top-20 | Optional; harness decides if it earns its keep |
| Generate | llama.cpp `llama-server`, Qwen3-8B or Llama-3.1-8B q4_K_M | Metal, prefix cache, OpenAI-compatible API |
| Obs | structlog JSON spans -> parquet -> waterfall plots | No vendor, no network |

**Memory budget on 18 GB:** LLM 8B q4_K_M ~5.0 GB + KV cache at 8k ctx ~1.0 GB
+ embedder ~0.1 GB + reranker ~0.3 GB + Python/numpy ~1.5 GB = **~8 GB
resident**. Comfortable. Do not attempt a 14B at 8k context on this box while
also holding a reranker — you will hit swap and every number becomes noise.

---

## 8. Build order

1. Harness + span timers + the fixed 25-question eval set
2. Naive baseline, deliberately unoptimised — this is your control
3. Batched CPU embed
4. Metal embed via llama-server
5. Resident service (kill the cold start)
6. Hybrid BM25 + RRF, optional reranker
7. llama-server with prefix cache + streaming
8. Incremental ingest with chunk hashing
9. Waterfall report + quality/speed frontier

Measure after every single step. A change you did not measure did not happen.

---

## 9. Open questions to settle with data, not opinion

- Does the reranker beat BM25+RRF by enough to justify its milliseconds?
- Is 400/80 the right chunk geometry for your corpus, or is it cargo cult?
- At what corpus size does flat search actually lose to HNSW on this machine?
- Is 8B worth its tok/s over a 4B for your question types?
