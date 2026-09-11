from __future__ import annotations
from rag.config import Config
from rag.generate.prompt import build_prompt

__all__ = ["build_prompt", "stream_answer"]


def stream_answer(cfg: Config, messages: list[dict]) -> dict:
    """Dispatch to the backend named in cfg.gen_backend. Pipeline calls this
    and only this - which HTTP API it hits is a config value, not a code
    branch anywhere else."""
    if cfg.gen_backend == "ollama":
        from rag.generate.ollama import stream_answer as impl
    elif cfg.gen_backend == "llamacpp":
        from rag.generate.llamacpp import stream_answer as impl
    else:
        raise ValueError(f"unknown gen_backend: {cfg.gen_backend!r}")
    return impl(cfg, messages)
