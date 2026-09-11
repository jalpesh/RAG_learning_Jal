"""Turn JSONL spans into a latency waterfall and a cross-config comparison.

    uv run python -m bench.report
"""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"

INGEST = ["embed_model_load", "parse", "chunk", "dedup", "embed_chunks", "index_build"]
QUERY = ["embed_query", "retrieve", "prompt_build", "generate"]


def load() -> pd.DataFrame:
    rows = []
    for f in sorted(RESULTS.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("no results yet - run bench.run first")
    return pd.DataFrame(rows)


def main() -> None:
    df = load()
    if "error" not in df.columns:
        df["error"] = None
    failed = df[df.error.notna()]
    if not failed.empty:
        print("\n=== ERRORS (excluded from waterfalls below - fix these, don't average them in) ===")
        print(failed.groupby(["config", "stage"])["error"].agg(["count", "first"]).to_string())

    df = df[df.error.isna()]

    ing = (df[df.stage.isin(INGEST)]
           .groupby(["config", "stage"])["ms"].sum().unstack(fill_value=0))
    ing = ing.reindex(columns=[c for c in INGEST if c in ing.columns])
    ing["TOTAL"] = ing.sum(axis=1)

    print("\n=== INGEST waterfall (ms) ===")
    print(ing.round(0).to_string())

    q = df[df.stage.isin(QUERY)]
    piv = q.groupby(["config", "stage"])["ms"].median().unstack(fill_value=0)
    piv = piv.reindex(columns=[c for c in QUERY if c in piv.columns])
    piv["TOTAL"] = piv.sum(axis=1)

    print("\n=== QUERY waterfall, median ms ===")
    print(piv.round(1).to_string())

    ans = df[df.stage == "answer"]
    if not ans.empty:
        perceived = ans.groupby("config").agg(
            ttft_p50=("ttft_ms", "median"),
            ttft_p95=("ttft_ms", lambda s: s.quantile(0.95)),
            tok_per_s=("tok_per_s", "median"),
            n=("ttft_ms", "size"),
        )
        print("\n=== PERCEIVED latency (what the user feels) ===")
        print(perceived.round(1).to_string())

    thr = df[df.stage == "embed_chunks"]
    if not thr.empty:
        t = df[df.stage == "embed_throughput"].set_index("config")["tokens"]
        e = thr.set_index("config")["ms"]
        both = pd.concat([t, e], axis=1).dropna()
        both["tok_per_s"] = both["tokens"] / (both["ms"] / 1000)
        print("\n=== EMBED throughput ===")
        print(both.round(0).to_string())

    gen = df[df.stage == "generate"]
    if "server_prompt_ms" in gen.columns and gen["server_prompt_ms"].notna().any():
        srv = gen[gen.server_prompt_ms.notna()].groupby("config").agg(
            prefill_ms=("server_prompt_ms", "median"),
            prefill_tok_per_s=("server_prompt_tok_per_s", "median"),
            decode_ms=("server_predict_ms", "median"),
            decode_tok_per_s=("server_predict_tok_per_s", "median"),
        )
        print("\n=== GENERATE, server-side split (llama-server native timings, median) ===")
        print(srv.round(1).to_string())

    print("\nBaseline is naive-cpu. Read every other row as a delta against it.")


if __name__ == "__main__":
    main()
