"""Span timing. Every stage of the pipeline is wrapped in one of these.

Design rule: a stage that is not measured does not exist. Spans are written as
JSONL so a run is append-only and crash-safe, and so bench/report.py can build
a waterfall without any of the pipeline code knowing about reporting.
"""
from __future__ import annotations

import json
import os
import platform
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

RESULTS_DIR = Path(__file__).resolve().parents[1] / "bench" / "results"


def _rss_mb() -> float:
    """Resident set size in MB. Cheap enough to sample per span."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports kilobytes - by platform, not by magnitude.
        divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
        return peak / divisor
    except Exception:
        return -1.0


@dataclass
class Run:
    """One benchmark run. Owns the JSONL sink and the shared run metadata."""

    config: str
    notes: str = ""
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    meta: dict[str, Any] = field(default_factory=dict)
    _fh: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = RESULTS_DIR / f"{stamp}_{self.config}_{self.run_id}.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")
        self.emit("run_start", 0.0, pid=os.getpid(), **self.meta)

    def emit(self, stage: str, ms: float, **fields: Any) -> None:
        rec = {
            "ts": time.time(),
            "run_id": self.run_id,
            "config": self.config,
            "stage": stage,
            "ms": round(ms, 3),
            "rss_mb": round(_rss_mb(), 1),
            **fields,
        }
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()

    @contextmanager
    def span(self, stage: str, **fields: Any) -> Iterator[dict[str, Any]]:
        """Time a block. Mutate the yielded dict to attach post-hoc facts
        (row counts, token counts) that are only known once the block ran."""
        extra: dict[str, Any] = {}
        t0 = time.perf_counter()
        error = None
        try:
            yield extra
        except Exception as exc:  # measure failures too; they are data
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            ms = (time.perf_counter() - t0) * 1000
            payload = {**fields, **extra}
            if error:
                payload["error"] = error
            self.emit(stage, ms, **payload)

    def close(self) -> None:
        self.emit("run_end", 0.0)
        if self._fh:
            self._fh.close()

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
