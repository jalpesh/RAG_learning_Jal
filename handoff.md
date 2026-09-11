# Handoff

Current state: repository setup and production-readiness audit complete.
The root README is the canonical installation and model-setup guide.
Run `./scripts/setup_ollama.sh` on each new machine after starting Ollama.
For the complete stack, use `./scripts/bootstrap_new_machine.sh`; reruns skip
healthy services and completed downloads.

Verification commands:

```bash
uv sync --locked
uv run python -m compileall -q rag service bench harness scripts tests
uv run python -m unittest discover -s tests -v
uv run python -m rag.confidence
uv run python -m rag.route
```

Hardware-dependent generation was not re-benchmarked during repository setup.
Existing benchmark evidence remains in the ignored `bench/results/` directory.
The next implementation target is the single-node production milestone in
`docs/03-production-readiness.md`.
