# local-rag-lab

Understand why a local RAG takes minutes when a hosted one takes seconds —
then close the gap on an M3 Pro / 18 GB, fully offline.

Read `docs/00-design.md` for the benchmark design and
`docs/03-production-readiness.md` for the production gap assessment.

## Where to build this

Run the complete benchmark on an Apple Silicon Mac. It depends on Metal,
Ollama and/or llama.cpp, multi-GB model files, and long-lived local servers.
The lightweight unit tests can run on macOS or Linux without downloading a
generation model.

## One-shot setup on a new Mac

After cloning the repository, run:

```bash
./scripts/bootstrap_new_machine.sh
```

The script is idempotent. It installs missing Homebrew packages, starts Ollama
only if needed, pulls missing Ollama models first, synchronizes Python,
resumes missing official GGUF downloads, starts only absent llama.cpp lanes,
and starts the API with automatic ingestion. Every phase prints its status.
If `corpus/` is empty, it stops after preparing the models; copy PDF/DOCX files
into `corpus/` and run the same command again.

Preview what it would do without installing, downloading, or starting anything:

```bash
./scripts/bootstrap_new_machine.sh --dry-run
```

> **Important:** seeing `qwen3:8b` in `ollama list` does not mean the 8B GGUF
> exists. Ollama stores models in its own managed format; `llama-server` needs
> the separate `models/qwen3-8b-official.gguf`. The bootstrap downloads and
> verifies both formats before starting anything that depends on them.

## 1. Install system prerequisites

```bash
brew install uv ollama llama.cpp

cd /Volumes/JalExt/projects/Learning/local-rag-lab
uv sync --locked
source .venv/bin/activate
```

`uv sync` installs the default local embedder,
`BAAI/bge-small-en-v1.5`. Sentence Transformers downloads its weights on the
first run. The download is cached outside this repository.

## 2. Download the Ollama models

The Ollama benchmark matrix in `rag/config.py` uses these generation models:

| Model | Purpose | Approximate class |
|---|---|---|
| `qwen3:4b` | Faster generation benchmark | 4B |
| `qwen3:8b` | Higher-quality control/deep model | 8B |

Start Ollama in a dedicated terminal:

```bash
ollama serve
```

Then install every required Ollama model with one command from another
terminal:

```bash
./scripts/setup_ollama.sh
```

If the Ollama desktop application is already running, do not start a second
server; run the setup script directly. It skips models already present and
pulls only missing ones.

Verify the complete machine setup:

```bash
./scripts/preflight.sh
```

Pinned configurations send `keep_alive="-1s"` in every Ollama API request, so
they behave the same after cloning to another machine. No
`OLLAMA_KEEP_ALIVE` shell variable is required.

Optional: the code can use Ollama for embeddings too. The default and measured
configuration uses Sentence Transformers, so this model is **not required**:

```bash
./scripts/setup_ollama.sh --with-embed
```

When creating that optional configuration, set both
`embed_backend="ollama"` and `embed_model="nomic-embed-text"`. Do not reuse
the default Hugging Face model name with the Ollama backend.

## 3. Add documents

Place `.pdf` or `.docx` files in `corpus/`. Corpus contents are ignored by Git
because they may contain private data; only `corpus/.gitkeep` is committed.

```bash
cp /path/to/document.pdf corpus/
```

## 4. Run Ollama-backed benchmarks

Run one configuration first, inspect its answers, and then run the matrix:

```bash
uv run python -m bench.run --config naive-cpu --regression
uv run python -m bench.run --config ollama-4b --regression
uv run python -m bench.run --all
uv run python -m bench.report
```

Available names are defined in `rag/config.py`. Runs using a `llamacpp-*`
configuration also require the matching llama.cpp server described below.

## 5. Run the resident two-lane service

The FastAPI service uses llama.cpp rather than Ollama: 4B is the fast lane and
8B is the deep lane. Ollama's downloaded model blobs are not the same as the
named files expected in `models/`. Download compatible Qwen3 GGUF files and
save them as:

```text
models/qwen3-4b-official.gguf
models/qwen3-8b-official.gguf
```

The `models/` directory is ignored by Git. Start each model server in its own
terminal only after confirming both files exist. The recommended command is
the one-shot bootstrap above because it downloads, verifies, and starts them
in the correct order. If starting manually, fail early with a useful message:

```bash
test -s models/qwen3-4b-official.gguf || {
  echo "Missing 4B GGUF; run ./scripts/bootstrap_new_machine.sh"
  exit 1
}
llama-server \
  -m models/qwen3-4b-official.gguf \
  --host 127.0.0.1 --port 8082 -ngl 99 -c 4096
```

```bash
test -s models/qwen3-8b-official.gguf || {
  echo "Missing 8B GGUF; run ./scripts/bootstrap_new_machine.sh"
  exit 1
}
llama-server \
  -m models/qwen3-8b-official.gguf \
  --host 127.0.0.1 --port 8081 -ngl 99 -c 4096
```

The GGUF downloads are roughly 2.5 GB (4B) and 5.0 GB (8B). Interrupted
downloads remain as `.part` files and resume on the next bootstrap run. The
script checks the exact expected byte size before a model can be started.

Then start the API in a third terminal:

```bash
uv run uvicorn service.app:app --host 127.0.0.1 --port 8000
```

This automatically ingests every supported file currently in `corpus/` during
startup. After adding, replacing, or removing corpus files while the API is
already running, re-ingest everything with this one command:

```bash
curl --fail --request POST http://127.0.0.1:8000/ingest
```

Check readiness and query it:

```bash
curl --fail http://127.0.0.1:8000/health

curl --fail http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the document say?"}'

uv run python scripts/chat.py
```

The default local setup is authless and bound to `127.0.0.1`, so it is only
reachable from the same machine. Authentication is optional: set
`RAG_API_KEY` on both the API and terminal client if you intentionally expose
the service beyond localhost. Never bind an unauthenticated instance to a LAN
interface.

## 6. Verify the code without model servers

```bash
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q rag service bench harness scripts tests
uv run python -m rag.confidence
uv run python -m rag.route
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

The lightweight suite does not start model servers or download GGUF files.
Run the benchmark harness on Apple Silicon to validate end-to-end retrieval,
generation quality, and latency.

## Model references

- [Qwen3 in the Ollama library](https://ollama.com/library/qwen3)
- [Nomic Embed Text in the Ollama library](https://ollama.com/library/nomic-embed-text)
