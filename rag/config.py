from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Config:
    """One point in the benchmark matrix.

    Everything that could plausibly affect latency or quality lives here, so a
    result row is fully explained by its config. No hidden defaults.
    """
    name: str

    # --- embedding ---
    embed_backend: str = "st"          # "st" (sentence-transformers) | "ollama"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_device: str = "cpu"          # "cpu" | "mps"   (st backend only)
    embed_batch: int = 32              # the single biggest ingest lever

    # --- chunking ---
    chunk_tokens: int = 400
    chunk_overlap: int = 80

    # --- retrieval ---
    top_k: int = 5
    candidates: int = 20               # pre-rerank / pre-fusion pool
    hybrid: bool = False               # BM25 + dense, fused with RRF
    metadata_filter: bool = False      # restrict search to one source when the question names it unambiguously
    reranker: bool = False             # cross-encoder second pass over the candidate pool
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_device: str = "mps"       # cpu | mps

    # --- generation ---
    gen_backend: str = "ollama"        # "ollama" | "llamacpp" (llama-server, OpenAI-compatible endpoint)
    gen_model: str = "qwen3:8b"
    gen_keep_alive: str = "5m"         # "-1s" pins the model in memory (Ollama duration syntax needs a unit; ignored by llamacpp - the server process IS the pin)
    gen_think: bool = False            # qwen3 reasons by default; thinking tokens stream as content="" and eat max_tokens
    max_tokens: int = 400
    ollama_host: str = "http://127.0.0.1:11434"
    llama_host: str = "http://127.0.0.1:8081"

    def as_dict(self) -> dict:
        return asdict(self)


# The control. Deliberately unoptimised - this is what you are trying to beat.
NAIVE = Config(name="naive-cpu")

MATRIX = [
    NAIVE,
    Config(name="batched-cpu", embed_batch=128),
    Config(name="batched-mps", embed_backend="st", embed_device="mps", embed_batch=128),
    Config(name="mps-pinned", embed_device="mps", embed_batch=128, gen_keep_alive="-1s"),
    Config(name="hybrid-pinned", embed_device="mps", embed_batch=128,
           gen_keep_alive="-1s", hybrid=True),
]

# Engine x model-size matrix. Embed side held fixed at the best-known
# settings from MATRIX above (mps, batch 128) so only gen_backend/gen_model
# move - "ollama-8b" is intentionally omitted, it's mps-pinned above (same
# config), reused as that cell instead of re-run.
GEN_MATRIX = [
    Config(name="ollama-4b", embed_device="mps", embed_batch=128,
           gen_keep_alive="-1s", gen_model="qwen3:4b"),
    Config(name="llamacpp-8b", embed_device="mps", embed_batch=128,
           gen_backend="llamacpp", gen_model="qwen3-8b",
           llama_host="http://127.0.0.1:8081"),
    Config(name="llamacpp-4b", embed_device="mps", embed_batch=128,
           gen_backend="llamacpp", gen_model="qwen3-4b",
           llama_host="http://127.0.0.1:8082"),
    Config(name="llamacpp-1.7b", embed_device="mps", embed_batch=128,
           gen_backend="llamacpp", gen_model="qwen3-1.7b",
           llama_host="http://127.0.0.1:8083"),
    Config(name="llamacpp-0.6b", embed_device="mps", embed_batch=128,
           gen_backend="llamacpp", gen_model="qwen3-0.6b",
           llama_host="http://127.0.0.1:8084"),
    Config(name="llamacpp-4b-filtered", embed_device="mps", embed_batch=128,
           gen_backend="llamacpp", gen_model="qwen3-4b",
           llama_host="http://127.0.0.1:8082", metadata_filter=True),
    Config(name="llamacpp-4b-reranked", embed_device="mps", embed_batch=128,
           gen_backend="llamacpp", gen_model="qwen3-4b",
           llama_host="http://127.0.0.1:8082", reranker=True),
    Config(name="llamacpp-4b-best", embed_device="mps", embed_batch=128,
           gen_backend="llamacpp", gen_model="qwen3-4b",
           llama_host="http://127.0.0.1:8082", metadata_filter=True, reranker=True),
    Config(name="llamacpp-8b-filtered", embed_device="mps", embed_batch=128,
           gen_backend="llamacpp", gen_model="qwen3-8b",
           llama_host="http://127.0.0.1:8081", metadata_filter=True),
    Config(name="llamacpp-1.7b-filtered", embed_device="mps", embed_batch=128,
           gen_backend="llamacpp", gen_model="qwen3-1.7b",
           llama_host="http://127.0.0.1:8083", metadata_filter=True),
    Config(name="llamacpp-0.6b-filtered", embed_device="mps", embed_batch=128,
           gen_backend="llamacpp", gen_model="qwen3-0.6b",
           llama_host="http://127.0.0.1:8084", metadata_filter=True),
]
