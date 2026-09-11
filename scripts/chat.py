#!/usr/bin/env python3
"""Interactive terminal client for the service - type a question, see which
lane answered it, watch a deep-lane job get polled to completion.

    uv run python scripts/chat.py
    uv run python scripts/chat.py --host http://127.0.0.1:8000

Requires the service already running (uv run uvicorn service.app:app --port 8000)
and both llama-server processes it depends on.
"""
from __future__ import annotations
import argparse
import sys
import time

import httpx

LANE_COLOR = {"cache": "\033[36m", "fast": "\033[32m", "deep": "\033[33m"}
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"


def ask(client: httpx.Client, question: str) -> None:
    t0 = time.perf_counter()
    try:
        r = client.post("/query", json={"question": question}, timeout=120)
        r.raise_for_status()
    except httpx.HTTPStatusError as exc:
        print(f"  {exc.response.status_code}: {exc.response.text}")
        return
    except httpx.ConnectError:
        print("  can't reach the service - is it running? (uv run uvicorn service.app:app --port 8000)")
        return
    out = r.json()

    if out.get("lane") == "deep":
        print(f"  {LANE_COLOR['deep']}[deep]{RESET} thinking", end="", flush=True)
        job_id = out["job_id"]
        while True:
            time.sleep(1)
            print(".", end="", flush=True)
            j = client.get(f"/jobs/{job_id}", timeout=30).json()
            if j["status"] != "running":
                print()
                out = j
                break

    lane = out.get("lane", "?")
    color = LANE_COLOR.get(lane, "")
    wall = round(time.perf_counter() - t0, 1)

    if out.get("status") == "error":
        print(f"  {color}[{lane}]{RESET} error: {out.get('error')}")
        return

    tag = f"{color}[{lane}]{RESET}"
    if out.get("cache_sim") is not None:
        tag += f" {DIM}(cache sim={out['cache_sim']}){RESET}"
    if out.get("low_confidence"):
        tag += f" {DIM}(flagged: {out.get('low_confidence_reason', out.get('trigger', '?'))}){RESET}"
    print(f"  {tag} {DIM}{wall}s{RESET}")
    print(f"\n{out.get('answer', '(no answer field)')}\n")
    if out.get("sources"):
        print(f"{DIM}sources: {', '.join(out['sources'])}{RESET}")
    if lane == "fast" and out.get("verify_job"):
        print(f"{DIM}low-confidence, verifying in background: /jobs/{out['verify_job']}{RESET}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    a = ap.parse_args()

    with httpx.Client(base_url=a.host) as client:
        try:
            h = client.get("/health", timeout=5).json()
            print(f"{BOLD}local-rag-lab{RESET} - connected to {a.host}, lanes={h.get('lanes')}, "
                  f"cache_size={h.get('cache_size')}")
        except httpx.ConnectError:
            print(f"can't reach {a.host} - start the service first:")
            print("  uv run uvicorn service.app:app --port 8000")
            sys.exit(1)

        print(f"{DIM}type a question, or 'exit' to quit{RESET}\n")
        while True:
            try:
                q = input(f"{BOLD}> {RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                break
            ask(client, q)


if __name__ == "__main__":
    main()
