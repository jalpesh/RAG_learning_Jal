# local-rag-lab

Understand why a local RAG takes minutes when a hosted one takes seconds —
then close the gap on an M3 Pro / 18 GB, fully offline.

Read `docs/00-design.md` first. It is the plan.

## Where to build this

**In Claude Code, on the Mac.** Not in a Cowork sandbox.
This project needs Metal, Ollama/llama.cpp, multi-GB model files and a
long-lived local server. The Cowork session's shell is a 3 GB headless Linux VM
with no GPU and no model runtime — it can read and write these files, but it
can never run or benchmark them.

## Setup (run on the Mac)

```bash
cd /Volumes/JalExt/projects/Learning/local-rag-lab
uv sync --locked
source .venv/bin/activate
brew install llama.cpp        # provides llama-server with Metal
```

## Layout

```
docs/      design, ADRs, benchmark write-ups
corpus/    test documents (gitignored)
harness/   span timers, eval set, runners
bench/     config matrix + results
rag/       ingest | retrieve | generate
scripts/   one-shot utilities
```

## Rule

Measure after every change. A change you did not measure did not happen.

## Verification

```bash
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q rag service bench harness scripts
```

The lightweight suite does not start model servers or download GGUF files.
Run the benchmark harness on Apple Silicon to validate end-to-end retrieval,
generation quality, and latency.
