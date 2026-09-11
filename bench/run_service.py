"""Run a question set against the live service (not Pipeline directly) -
the only way to actually exercise cache/router/escalation together, since
those live in service/app.py, not rag/pipeline.py.

    uv run python -m bench.run_service --questions questions_v2.json
    uv run python -m bench.run_service --questions questions_v2.json --repeat-first 3
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def ask(client: httpx.Client, question: str) -> dict:
    t0 = time.perf_counter()
    r = client.post("/query", json={"question": question}, timeout=120)
    r.raise_for_status()
    out = r.json()
    out["_wall_s"] = round(time.perf_counter() - t0, 2)
    if out.get("lane") == "deep":
        # poll the async job to completion so the eval sees a real answer
        job_id = out["job_id"]
        while True:
            time.sleep(1)
            j = client.get(f"/jobs/{job_id}", timeout=30).json()
            if j["status"] != "running":
                out = {**out, **j, "_wall_s": round(time.perf_counter() - t0, 2)}
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="questions_v2.json")
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    ap.add_argument("--repeat-first", type=int, default=0,
                     help="re-ask the first N questions again at the end, to demonstrate cache hits")
    a = ap.parse_args()

    questions = json.loads((ROOT / "harness" / a.questions).read_text())["questions"]
    results = []
    with httpx.Client(base_url=a.host) as client:
        for i, q in enumerate(questions):
            res = ask(client, q["q"])
            results.append({"i": i, "type": q["type"], "q": q["q"], "expect": q.get("expect", ""), **res})
            print(f"[{i:2}] {q['type']:12} lane={res.get('lane'):5} "
                  f"low_conf={res.get('low_confidence', '-')!s:5} "
                  f"wall={res['_wall_s']:5.1f}s  {q['q'][:55]}")

        if a.repeat_first:
            print(f"\n--- repeating first {a.repeat_first} questions (expect cache hits) ---")
            for q in questions[: a.repeat_first]:
                res = ask(client, q["q"])
                print(f"     lane={res.get('lane'):5} cache_sim={res.get('cache_sim')}  {q['q'][:55]}")

    out_path = ROOT / "bench" / "results" / f"service_{a.questions.replace('.json','')}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
