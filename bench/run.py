"""Run one config end to end over the corpus and the fixed question set.

    uv run python -m bench.run --config naive-cpu
    uv run python -m bench.run --all
"""
from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path

from harness.spans import Run
from rag.config import MATRIX, GEN_MATRIX, Config
from rag.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
QUESTIONS = ROOT / "harness" / "questions.json"


def run_config(cfg: Config, questions_path: Path, warmups: int = 1) -> None:
    questions = json.loads(questions_path.read_text())["questions"]
    meta = {"machine": platform.machine(), "python": platform.python_version(),
            "questions_file": questions_path.name, **cfg.as_dict()}

    with Run(config=cfg.name, meta=meta) as run:
        pipe = Pipeline(cfg, run)
        pipe.ingest(CORPUS)

        # Warm-up queries are recorded but tagged, so cold and steady-state
        # numbers never get averaged together into a meaningless middle.
        for i, q in enumerate(questions[:warmups]):
            run.emit("phase", 0.0, phase="warmup", i=i)
            pipe.query(q["q"])

        for i, q in enumerate(questions):
            run.emit("phase", 0.0, phase="measure", i=i)
            res = pipe.query(q["q"])
            run.emit("answer", 0.0, i=i, ttft_ms=res["ttft_ms"],
                     tok_per_s=res["tok_per_s"], sources=res["sources"],
                     expect=q.get("expect", ""), answer=res["answer"][:500])

    print(f"[done] {cfg.name} -> {run.path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="config name from rag.config.MATRIX")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--regression", action="store_true",
                     help="use harness/regression.json (fast subset) instead of the full 25-question set")
    ap.add_argument("--questions", help="use harness/<this file> instead of questions.json (e.g. questions_v2.json)")
    a = ap.parse_args()

    if a.questions:
        questions_path = ROOT / "harness" / a.questions
    elif a.regression:
        questions_path = ROOT / "harness" / "regression.json"
    else:
        questions_path = QUESTIONS

    all_configs = MATRIX + GEN_MATRIX
    todo = all_configs if a.all else [c for c in all_configs if c.name == a.config]
    if not todo:
        raise SystemExit(f"no such config. available: {[c.name for c in all_configs]}")
    for cfg in todo:
        run_config(cfg, questions_path)


if __name__ == "__main__":
    main()
