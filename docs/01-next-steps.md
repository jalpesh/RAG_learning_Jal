# Next steps — switch to Claude Code now

The harness and naive baseline are on disk. Everything from here needs Metal,
Ollama and multi-GB models. **Switch now.**

```bash
cd /Volumes/JalExt/projects/Learning/local-rag-lab
claude
```

---

## Step 0 — preflight (10 min)

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[st]"
ollama pull nomic-embed-text
ollama pull llama3.1:8b
./scripts/preflight.sh
```

`preflight.sh` must report Ollama up, `mps available: True`, and both models
present. Do not proceed on a red line — you will spend an evening debugging a
benchmark that was never valid.

## Step 1 — build the corpus and the eval set (60–90 min, unskippable)

Drop 3–5 real PDFs into `corpus/`. Use documents you actually know, because you
have to be able to grade the answers. Then replace the four placeholders in
`harness/questions.json` with **25 real questions**, spread across:

- **factual** — answerable from one chunk
- **multihop** — needs two sections combined
- **aggregation** — requires summarising across many chunks
- **unanswerable** — plausible, absent from the corpus, correct answer is a refusal

This is the least fun step and the one that decides whether the whole project
is worth anything. A speed benchmark with no quality axis will happily tell you
that returning garbage in 40 ms is the winning configuration. The unanswerable
questions matter most — they are the only thing that catches a retrieval
config that has quietly started hallucinating.

## Step 2 — the control run

```bash
uv run python -m bench.run --config naive-cpu
uv run python -m bench.report
```

Expect something ugly: 60–180 s ingest, 10–40 s to first token. **Do not fix
anything yet.** This number is the entire point of the exercise — it is what
every later result is measured against.

## Step 3 — walk the matrix, one variable at a time

```bash
uv run python -m bench.run --config batched-cpu     # batch 32 -> 128
uv run python -m bench.run --config batched-mps     # + Metal
export OLLAMA_KEEP_ALIVE=-1 && ollama serve         # in another terminal
uv run python -m bench.run --config mps-pinned      # + pinned LLM
uv run python -m bench.run --config hybrid-pinned   # + BM25/RRF
uv run python -m bench.report
```

Report after **every** run. One variable per run — if you change two and the
number improves, you have learned nothing.

## Step 4 — read the waterfall and write it up

`docs/02-findings.md`. For each config: what moved, by how much, and *why* —
mechanism, not correlation. Bring that file back here and I'll turn it into the
CTO-facing version.

## Step 5 — only then, build the resident service

FastAPI wrapper around `Pipeline`, SSE streaming, incremental ingest via the
chunk-hash ledger. Not before. A service built on unmeasured assumptions is
just a faster way to be wrong.

---

## What I still owe you (bring findings back to Cowork)

- ADR on flat-vs-ANN with your actual crossover point
- The quality/speed frontier plot
- A leadership-grade write-up: "what on-prem RAG costs in latency, and what
  it buys in data residency" — directly relevant to a Swiss bank

---

## Two traps

1. **Tuning before measuring.** You will want to jump to `batched-mps` because
   it is obviously right. It is obviously right. Run the control anyway, or you
   will never be able to say which of five changes bought the speed — which is
   the exact position you are in today, one level up.
2. **Optimising ingest when query is what hurts.** Ingest runs once per
   document. Query runs a hundred times a day. If the waterfall says generation
   dominates, no amount of embedding cleverness will save you — the answer is a
   smaller model, fewer output tokens, or streaming.
