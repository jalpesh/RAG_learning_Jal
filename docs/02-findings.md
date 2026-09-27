# Findings — first pass, M3 Pro / 18 GB

Corpus: 5 real PDFs (Aadhaar card, a camp registration form, two resumes, a
7-day trip itinerary), 16 pages, 23 chunks, 5386 tokens. 25 fixed questions
(7 factual, 6 multihop, 6 aggregation, 6 unanswerable). Embedder
`bge-small-en-v1.5`, generator `qwen3:8b` via Ollama (swapped in for
`llama3.1:8b` — already resident on this machine, same size class).

## Two bugs found before any number could be trusted

The harness did exactly what it was built for: it surfaced its own lies
before I could mistake them for results.

1. **`keep_alive="-1"` is not valid Ollama syntax.** Go's duration parser
   needs a unit (`"-1s"`), so both pinned configs' every `generate` call
   returned `400 Bad Request`. The span timer caught it and logged the
   failure (per design — "a crash that takes 90s is data") but
   `bench/report.py` didn't know to treat an error row differently from a
   real one, so the waterfall blended a 9 ms crash into the "generate"
   median and made `mps-pinned`/`hybrid-pinned` look impossibly fast.
   Fixed the config value and added an `=== ERRORS ===` table to
   `bench/report.py` that pulls failed spans out of every average instead
   of silently averaging them in.

2. **`qwen3:8b` reasons by default.** Ollama streams its chain-of-thought as
   a separate `thinking` field with `content: ""`; the harness only counted
   `content` as "the answer has started," so every TTFT number included the
   full internal reasoning phase (12–24 **seconds**), and ~20% of answers
   came back completely empty because `max_tokens=400` was consumed by
   thinking before any real answer token appeared. This is precisely the
   failure mode the eval set exists to catch — a fast, confident, silently
   wrong (here: silently *empty*) answer — except it hit generation, not
   retrieval. Fixed with `"think": false` in the `/api/chat` payload,
   gated behind a new `Config.gen_think` field. Re-ran the full matrix
   after the fix; 0 empty answers across all 5 configs afterward.

Both are one-line fixes. Neither would have been visible from the top-line
numbers alone — only from reading actual answer text and actual error
fields, which is the entire argument for keeping this harness instead of
trusting `bench.run`'s exit code.

## Ingest waterfall (ms)

| config | embed_model_load | embed_chunks | TOTAL |
|---|---|---|---|
| naive-cpu | 9589 | 869 | 10524 |
| batched-cpu | 9688 | 970 | 10733 |
| batched-mps | 22338 | 2229 | 24780 |
| mps-pinned | 9584 | 288 | 9933 |
| hybrid-pinned | 9830 | 295 | 10189 |

**Model load dominates ingest at this corpus size, and no config variable
touches it** — it's `import torch` + `sentence-transformers` weight load,
~9.6–9.8s everywhere except `batched-mps`.

**`batched-mps` is the one real anomaly, and it's a one-time tax, not a
config regression.** It was the first config in the run to ever touch the
MPS backend in that process — PyTorch has to JIT-compile Metal kernels on
first dispatch. That shows up twice: model load jumps to 22.3s (backend
init folded into the load span) and embed throughput *drops* to 2416 tok/s
— slower than plain CPU (5555–6200 tok/s). `mps-pinned` and `hybrid-pinned`
ran immediately after in the same warm environment and hit 18266–18689
tok/s, 3–8x faster than CPU. **Conclusion: MPS is a large win for embedding,
but only after the first-dispatch compilation cost is paid once per
process.** In the eventual resident service (Step 5) this cost is paid
exactly once at startup and amortized over every ingest after — a CLI that
re-imports torch per call would pay it every time, which is the whole
argument for `Pipeline` being a resident object.

At 23 chunks, batch size (32→128) made no measurable difference — there's
only one batch either way. This variable needs a bigger corpus to mean
anything; it wasn't tested honestly here.

## Query waterfall, median ms (post-fix)

| config | embed_query | retrieve | generate | TOTAL |
|---|---|---|---|---|
| naive-cpu | 23.8 | 0.1 | 8087.5 | 8111.5 |
| batched-cpu | 24.6 | 0.1 | 8063.8 | 8088.5 |
| batched-mps | 85.4 | 0.1 | 8488.1 | 8573.6 |
| mps-pinned | 57.2 | 0.0 | 8100.9 | 8158.2 |
| hybrid-pinned | 62.2 | 0.2 | 8357.4 | 8419.9 |

**`generate` is 99%+ of query time in every single config, and it is
statistically identical across all five** (8064–8488ms median — the spread
is noise, not signal). `embed_query` and `retrieve` are 0.1–0.7% of the
total. This is Trap 2 from `01-next-steps.md`, exactly as predicted:
*"If the waterfall says generation dominates, no amount of embedding
cleverness will save you."* None of batching, MPS, pinning, or hybrid
retrieval could have moved perceived latency here even in principle — the
bottleneck is 100% in `qwen3:8b`'s token generation, which none of those
five variables touch.

## Perceived latency

| config | TTFT p50 | TTFT p95 | tok/s |
|---|---|---|---|
| naive-cpu | 5737.5 | 7564.9 | 24.0 |
| batched-cpu | 5752.5 | 7559.5 | 24.1 |
| batched-mps | 5829.7 | 8104.6 | 24.1 |
| mps-pinned | 5732.4 | 7571.8 | 24.2 |
| hybrid-pinned | 5909.3 | 7974.5 | 24.2 |

~5.8s to first token, ~24 tok/s steady state, flat across the matrix. Against
the design doc's own target budget (TTFT 250–600ms, 25–35 tok/s) this is
roughly **10x over budget on TTFT** and slightly under on throughput — and
the gap has nothing to do with retrieval. It's prompt prefill + decode on an
8B q4 model over ~1200 context tokens on this hardware/runtime.

## Quality check (the part that was almost skipped)

All 6 unanswerable questions were correctly refused post-fix, e.g. *"The
context does not provide the name of Jalpesh's father"* — no hallucination
observed. Spot-checked factual and multihop answers (DOB, cross-document
phone number match, resume date consistency) against the corpus and they
were correct. The quality axis holds across the whole matrix; the fast
configs are not winning by cutting corners on correctness.

## What this run actually proves, and what it doesn't

Proves: at a 5-document / 23-chunk corpus, none of {batch size, embed
device, LLM pinning, hybrid retrieval} affect what the user feels, because
generation swamps everything else by 2–3 orders of magnitude. Retrieval-side
tuning here would be optimizing 0.5% of the wall clock.

Doesn't prove: whether MPS batching matters at realistic scale. 23 chunks is
below where any of these levers should matter — the design doc's own
crossover discussion (flat vs ANN at "six figures" of vectors) is the same
shape of question as batch-size-matters-at-what-N. Re-running this matrix
against a corpus in the thousands-of-chunks range (multiple long PDFs,
not five short ones) is the next thing that would produce a real answer
instead of noise.

## Addendum — the actually-resident scenario (warm process, new ingest, then query)

The matrix above answers "which config wins" but every `bench.run` invocation
is its own process, so it always re-pays `embed_model_load` once. That's not
what Step 5's resident service looks like. Measured the real scenario
directly: one long-lived `Pipeline`, warm-up ingest, then a *second* ingest
call (simulating a new PDF arriving) and a query, both timed in-process.

**Found a third bug doing this**: `Embedder.load()` unconditionally
reconstructed `SentenceTransformer` on every call — so even within one
resident process, a second `ingest()` silently re-paid the full ~6s model
load. `Pipeline` was resident in name only. Fixed with a no-op guard in
`rag/ingest/embed.py`. Before/after, same warm process:

| | ingest (2nd call, same process) | query TTFT |
|---|---|---|
| before fix (bug) | 5949 ms (re-loaded the model) | 13384 ms |
| after fix | **200 ms** | 171–8293 ms (see below) |

**Warm ingest of a new document, once the bug is fixed: ~200ms, consistently**
(3 repeats, no variance) — parse + chunk + embed 23 chunks + index, no model
reload. This is the number Step 5 should actually deliver per document.

**Warm query TTFT is highly bimodal**, and the mechanism turned out to be
worth chasing down: sent a synthetic ~1200-token context (matching real
retrieved-chunk size) directly to Ollama, isolated from retrieval entirely,
and got ~4.9s TTFT — same order as the slow real queries. A near-empty
context returns in ~170ms. **The dominant cost of TTFT at realistic RAG
context sizes is LLM prompt prefill, not decode, not retrieval, and not
`qwen3:8b`'s reasoning mode** (confirmed `think:false` is honored — no
`thinking` field appears in the stream). This machine/runtime's prefill
throughput for ~1200 tokens on `qwen3:8b` q4 via Ollama's Metal backend is
the real bottleneck behind the design doc's 250–600ms TTFT target being
missed by ~10x, not anything in the retrieval-side matrix above.

**Implication for Step 5**: prefix-caching the system prompt (already on the
open-questions list) won't help much here, because the part that's slow is
the *variable* per-query context (the retrieved chunks), which can't be
cached across different questions. The real lever is context size itself —
fewer/shorter retrieved chunks (lower `top_k` or `candidates`), or a
faster-prefill runtime path (llama.cpp `llama-server` instead of Ollama, per
the original architecture sketch) — not anything already in `Config`.

## Addendum 2 — engine (Ollama vs llama-server) x model size (8B vs 4B)

Added `rag/generate/llamacpp.py` (OpenAI-compatible `/v1/chat/completions`
against `llama-server`) alongside the existing Ollama path, dispatched via
new `Config.gen_backend`. Pulled `qwen3:4b` (Ollama registry) as the smaller
model. Ran a clean 2x2: `{ollama, llamacpp} x {8b, 4b}`, embed side held
fixed at the best-known settings (mps, batch 128) so only engine and model
size move.

**Three more thinking-suppression bugs, one per model/source combination**,
found the same way as the earlier ones - by checking `n_chunks_streamed`/
answer length, not by trusting a clean exit code:

1. `llamacpp-4b` against the Ollama-derived GGUF (same blob-symlink trick
   used successfully for the 8B model): `chat_template_kwargs.enable_thinking:
   false` was silently ignored - 18/25 answers came back empty, and
   `server_predict_ms` showed the full `max_tokens` budget being spent on
   `reasoning_content` regardless of the flag.
2. Tried the server-level `--reasoning off` flag instead - still ignored.
3. Tried `--reasoning-budget 0` (forces immediate end-of-thinking) - also
   ignored, verified by streaming the raw response and watching it reason
   for 100+ tokens anyway.
4. Tried the Qwen3-era `/no_think` prompt suffix - the model reasoned
   *about* the literal string "/no_think" instead of obeying it, meaning
   this checkpoint was never trained to treat it as a control token.

Root cause: Ollama's own model packaging/template for `qwen3:4b` apparently
carries logic (or an architecture variant) that only Ollama's runtime knows
how to switch off - none of it survives into the raw GGUF that llama-server
reads. **Fixed by downloading the official `Qwen/Qwen3-4B-GGUF`
(`Qwen3-4B-Q4_K_M.gguf`, ~2.5GB, Hugging Face) instead of reusing the
Ollama blob** - its template correctly honors `enable_thinking`. The 8B
Ollama blob never had this problem (`chat_template_kwargs` worked on the
first try), so the issue is specific to this particular 4B packaging, not
llama-server or Qwen3 generally.

Separately, spot-checking `ollama-4b`'s "unanswerable" answers found a
milder version of the same class of bug: `think:false` over Ollama's API
does silence the separate `thinking` stream field for `qwen3:4b`, but the
model still writes visible "Hmm, let me check the context..." reasoning as
regular `content`, terminated by a stray literal `</think>` tag Ollama
doesn't strip. The refusal at the end is still correct, but the answer is
long, unpolished, and about 3x the length it needs to be. `llamacpp-4b` and
`llamacpp-8b` (official/blob GGUF respectively) never show this - clean,
direct answers. Net: for `qwen3:4b` specifically, Ollama's non-thinking
mode is not actually clean; llama-server's is.

### Results (all 25/25 answered, no errors, refusals correct on both engines)

| Config | TTFT p50 | TTFT p95 | tok/s | Ingest total |
|---|---|---|---|---|
| ollama-8b (= `mps-pinned` above) | 5732 ms | 7572 ms | 24.2 | 9933 ms |
| ollama-4b | 3318 ms | 4449 ms | 40.8 | 10158 ms |
| llamacpp-8b | 4087 ms | 7159 ms | 24.8 | 10119 ms |
| llamacpp-4b | **2358 ms** | **4028 ms** | **43.6** | 10479 ms |

**Model size is the dominant lever, bigger than engine choice.** 4B is
~1.6-1.7x faster to first token than 8B on *either* engine, and ~1.7x
higher decode throughput - roughly what you'd expect from half the
parameters at the same quantization. This is a much bigger and more
reliable win than anything in the original embed-side matrix.

**Engine matters too, independently of size**, and llama-server's edge is
almost entirely in prefill, not decode. Native server-side split
(`llamacpp` rows only, `median ms`):

| Config | prefill (ms) | prefill tok/s | decode (ms) | decode tok/s |
|---|---|---|---|---|
| llamacpp-8b | 4027 | 268 | 1414 | 24.8 |
| llamacpp-4b | 2303 | 477 | 1215 | 43.7 |

Decode tok/s is essentially identical to Ollama at the same size (24.8 vs
24.2 for 8B; 43.6-43.7 vs 40.8 for 4B - within noise). The real gap is
prefill: llama-server's own reported prefill throughput (268-477 tok/s) is
the mechanism behind its lower TTFT - this matches the earlier finding that
TTFT is dominated by prompt processing of the ~1200-token retrieved
context, not decode, and confirms llama.cpp's direct Metal path processes
that prefill faster than going through Ollama's layer on top of the same
llama.cpp core.

**Best combination found so far: `llamacpp-4b` - 2.4s TTFT, 43.6 tok/s.**
Down from the original `naive-cpu` control's ~20s+ TTFT-including-thinking-
time, and about 2.4x better TTFT than the best 8B config. Still ~5-10x a
typical hosted API's TTFT (per the cloud comparison discussed earlier), but
this is the actual ceiling this hardware/model-family combination has shown
so far, not a guess.

## Addendum 3 — going smaller: qwen3:1.7b and qwen3:0.6b, and where quality actually breaks

Downloaded official `Qwen/Qwen3-1.7B-GGUF` and `Qwen/Qwen3-0.6B-GGUF`
(Q8_0 — these sizes don't ship a Q4_K_M on the official repo, so this pair
is actually *higher* precision than the 4B/8B tests, not lower). Added
`llamacpp-1.7b` / `llamacpp-0.6b` configs. Both ran clean: 25/25 answered,
no errors, no empty responses.

**Speed keeps improving monotonically with smaller size** — no surprises
here, this is the expected FLOPs relationship:

| Config | warm TTFT p50 | warm tok/s |
|---|---|---|
| llamacpp-8b | 4087 ms | 24.8 |
| llamacpp-4b | 2358 ms | 43.6 |
| llamacpp-1.7b | **935 ms** | **60.1** |
| llamacpp-0.6b | **425 ms** | **123.7** |

`llamacpp-0.6b` actually clears the design doc's original TTFT target
(250-600ms) on the low end — first result in this whole project to do so.

**Quality does not break uniformly — it breaks per task type, and refusal
correctness turned out to be the most robust capability of all**, not the
most fragile. Checked all 25 answers by hand across all four sizes,
including the same four cross-document/counting questions at every size:

| Task | 8b | 4b | 1.7b | 0.6b |
|---|---|---|---|---|
| Refuse all 6 unanswerable questions correctly | ✓ | ✓ | ✓ | ✓ |
| Count 9 Mumbai pick-up points | ✓ (9) | ✓ (9) | ✗ (8) | ✗ (7) |
| Reliance Jio dates: do the 2 resumes agree? (yes) | ✓ | ✓ | ✓ | ✗ (said no) |
| Jalpesh_Rajani.pdf years of experience (18) | ✓ | ✓* | ✓ | ✗ (said 19) |
| HCL resume years of experience (19, stated in prose not a bullet) | ✗ | ✗ | ✗ | ✗ |

(*4B got this right in the direct factual question but wrong - "19" - when
asked the multihop version combining it with graduation year, suggesting
retrieval pulled in the other resume's chunk and it cross-contaminated.)

Three separate, genuinely different findings here, not one "small models
are worse" story:

1. **Refusal/hallucination-avoidance held perfectly at every size down to
   0.6B.** This is the capability the entire eval set was built to protect
   (`01-next-steps.md`: "the unanswerable questions matter most"), and it's
   the one that degraded *least*. Worth remembering next time a shrink is
   tempting - it's not automatically the risky part.
2. **Counting/enumeration is the first thing to go**, breaking between 4B
   and 1.7B, and getting worse at 0.6B (off by one, then off by two). This
   tracks with general small-model literature - precise multi-item
   enumeration needs more working "state" than a 0.6-1.7B model reliably
   holds across a ~1200-token context.
3. **Cross-document consistency-checking holds until the very bottom** -
   0.6B is the only size that got the Reliance Jio date comparison wrong,
   and it didn't hedge, it confidently asserted the wrong thing ("do not
   agree"). That's a more concerning failure mode than a refusal, because
   it's confidently incorrect rather than honestly uncertain.
4. **The HCL years-of-experience question broke at every single size**,
   8B included. Not a model-size story at all - likely a document-
   formatting issue (the number is inside a prose sentence -
   "...19 years of experience driving technical excellence..." - versus
   the other resume's clean bullet "18 years of progressive experience").
   Worth fixing the eval question or the retrieval/chunking around it
   before drawing size conclusions from it.

**Bottom line on model size**: `qwen3:1.7b` looks like a genuinely usable
floor for this kind of document-QA RAG - fast (935ms TTFT), and only broke
on precise counting, not on the higher-stakes refusal/hallucination axis.
`qwen3:0.6b` is fast enough to matter (425ms TTFT, sub-cloud-target) but
starts making confidently wrong claims on cross-document comparisons,
which is a real quality regression, not just a speed/quality tradeoff
curve - it crossed from "slower but careful" to "fast but occasionally
confidently wrong."

## Addendum 4 — elaborate ingest/query table, cold vs warm, all sizes

Definitions, made precise because "warm" and "cold" mean different things
at different layers:

- **Cold ingest** = a fresh process's first `ingest()` call, including
  `embed_model_load` (paying for `import torch` + weight load once).
- **Warm ingest** = a second `ingest()` call in the same already-loaded
  process (simulating a new document arriving at a resident service).
  Measured once (~200ms, Addendum 1) - this number is *identical* across
  every config below because ingest only touches the embedder
  (`bge-small-en-v1.5`), which is the same for all of them; the generation
  model/backend has no effect on ingest time.
- **Cold query** = the harness's own warm-up query (`bench/run.py` always
  fires one untimed warm-up before the 25 measured questions) - the first
  request that model has ever answered in this process/server.
  For the `ollama-*` rows this can include Ollama loading the model into
  GPU memory for the first time if it wasn't already resident: that's why
  `ollama-4b`'s cold number below is an outlier (15s) while `ollama-8b`'s
  isn't (it happened to already be warm in Ollama from earlier testing
  this session). For the `llamacpp-*` rows, the server process had already
  loaded the model file at startup before any request arrived, so "cold"
  here mostly reflects an empty KV/prefill cache, not a model load - a
  materially different, smaller kind of "cold."
- **Warm query** = median TTFT across the 25 measured questions, steady
  state, same process/server that answered the warm-up query.

| Config | Cold ingest | Warm ingest | Cold query (TTFT) | Warm query (TTFT p50) | Warm tok/s |
|---|---|---|---|---|---|
| ollama-8b | 9933 ms | ~200 ms | 5586 ms | 5732 ms | 24.2 |
| ollama-4b | 10158 ms | ~200 ms | **15001 ms** † | 3318 ms | 40.8 |
| llamacpp-8b | 10119 ms | ~200 ms | 5404 ms | 4087 ms | 24.8 |
| llamacpp-4b | 10479 ms | ~200 ms | 3017 ms | 2358 ms | 43.6 |
| llamacpp-1.7b | 11116 ms | ~200 ms | 1184 ms | 935 ms | 60.1 |
| llamacpp-0.6b | 9682 ms | ~200 ms | 650 ms | 425 ms | 123.7 |

† Ollama's cold-query outlier for `qwen3:4b` (15s vs its own 3.3s warm
median) is very likely a genuine GPU-memory model-swap cost - at the point
that run started, Ollama had `qwen3:8b` pinned resident from earlier
testing, so loading `qwen3:4b` for the first time meant evicting one model
and loading another, not just a cold cache. This is itself a real,
reportable cost of Ollama's shared-server model, and it's a cost
`llama-server` structurally cannot hit (a llama-server process only ever
serves the one model it was started with, so there's no other model to
evict) - one more point in favor of the dedicated-process-per-model
pattern for a production resident service.

**Read of the whole table**: ingest is a rounding error either way (~10s
once, ~200ms per document after) and doesn't depend on which generation
setup you pick at all. Query is where every real decision lives, and the
cold/warm gap on the `llamacpp` rows is modest (roughly 1.3-1.6x) because
the model is already resident in memory the moment the server starts -
compare that to Ollama's up-to-4.5x cold/warm gap when a model swap is
involved. For a resident service design, this is the strongest argument
yet for one `llama-server` process per model over Ollama's shared,
swappable model store.

## Addendum 5 — diagnosing the HCL years-of-experience failure (broke at every model size)

Root-caused rather than guessed at. Checked `sources` on every failing run:
the correct chunk (`Resume-Jalpesh-Rajani-HCLSoftware.pdf p.1`, containing
the literal sentence *"...19 years of experience driving technical
excellence..."*) **was retrieved every single time, at every model size.**
Not a retrieval miss. Then checked the chunk itself for a boundary split -
the sentence is fully intact inside one 400-token chunk, not cut across two.
Not a chunking bug either.

**Real cause, confirmed by reading the model's own stated reasoning**: the
prompt puts 5 chunks from *two different resumes* into one ~1200-token
context. Rather than quoting the explicit "19 years" sentence, the model
consistently chooses to *compute* an answer by summing job-duration bullets
- and the wrong resume's chunks (3 of 5 retrieved) dominate the context, so
it sums *that* resume's durations instead. This is a reasoning-strategy
bias (prefers arithmetic over quotation), not a search or formatting bug.

Tested two candidate fixes live, both inconclusive or negative - worth
recording so the next pass doesn't re-try them expecting a clean win:

- **`hybrid=True` (BM25+RRF, already built, just never enabled) made it
  worse**, not better: it dropped the correct HCL chunk from the top-5
  entirely (RRF fusion returned a duplicate chunk instead) and the answer
  got further from correct (14 vs the true 19). Recorded as a real result,
  not a guess - turning on hybrid retrieval is not automatically an
  improvement, and this project's own harness is what caught that.
- **A system-prompt instruction to "quote explicit values, don't
  calculate"** did not change the model's behavior - it still computed an
  answer from job-duration arithmetic rather than quoting the sentence,
  and happened to land on the correct total this one time. Not a reliable
  fix; didn't re-test for consistency given time, but the underlying
  behavior (compute over quote) visibly didn't change, so treat this as
  unresolved, not fixed.

**What would actually attack the root cause** (untested, ranked by
directness):
1. **Metadata filtering** - when a question names a specific document
   ("the HCL Software resume"), filter retrieval to that source before
   ranking, so the competing resume's chunks never enter context at all.
   Most direct fix for this exact failure mode.
2. **Contextual retrieval** (Anthropic's published technique - prepend a
   short LLM-generated blurb like "This chunk is from Jalpesh Rajani's HCL
   Software resume, summary section" to each chunk before embedding) -
   improves the embedding's ability to disambiguate near-duplicate
   documents at the retrieval stage instead of leaving disambiguation to
   the generator.
3. **Self-consistency** (ask 2-3 times, take the majority/median answer) -
   cheap to do now that `qwen3:1.7b`/`0.6b` answer in under a second; would
   not fix the bias but might average it out for numeric questions.
4. A cross-encoder **reranker** would likely help generally but isn't
   guaranteed to fix this specific case, since the correct chunk was
   already in the top-5 - reranking order, not recall, isn't obviously the
   bottleneck here.

## Addendum 6 — metadata filtering, implemented and measured

Built the fix proposed in Addendum 5: `rag/retrieve/filter.py` tokenizes
each corpus filename, keeps only tokens that belong to exactly one source
(so `"jalpesh"`/`"rajani"`, shared by both resumes' filenames, never
falsely narrow anything), and matches a question against those tokens plus
an exact-filename substring check. `FlatIndex.dense()`/`.sparse()` gained
an `allowed: set[int] | None` parameter to restrict the candidate pool
before ranking; `Pipeline.query()` applies it when `Config.metadata_filter`
is on and the question unambiguously names one or more sources - multi-
source matches (e.g. "the Aadhaar card and the Camp Ethnosphere form")
filter to the *union*, not to nothing, so genuinely cross-document
questions still get both documents while everything else is excluded.
Self-check in the module itself (`python -m rag.retrieve.filter`) covers
the shared-token case, the plural-vs-singular case, and the multi-source
case before any of this touches retrieval.

Added `llamacpp-4b-filtered` config and re-ran the full 25-question eval.
25/25 answered, no errors, no empty responses, **all 6 refusals still
correct** (no new hallucination introduced).

**Confirmed fix, on the exact bug that motivated this**: the HCL
years-of-experience question, wrong at every model size in Addendum 3,
now answers correctly with metadata filtering on:

| | Unfiltered | Filtered |
|---|---|---|
| "How many years does the HCL resume claim?" | *"...claims 20 years... based on Bachelor of Engineering in 2005..."* (wrong, computed from the wrong resume's dates) | *"...claims 19 years of experience [Resume-Jalpesh-Rajani-HCLSoftware.pdf p.1]"* (correct, quotes the source) |
| Retrieved sources | 3 of 5 chunks from the *other* resume | 5 of 5 chunks from the correct resume |

**Bonus, unplanned fix**: the multihop question combining graduation year
with years-of-experience (Addendum 3 didn't call this one out separately,
but it has the same contamination) also flipped from wrong (19) to right
(18) under filtering - the same root cause, same fix, no extra work.

**Checked all 25 for regressions, found one real limitation, not swept
under the rug**: question 10 ("whose phone number appears on both the Camp
Ethnosphere form *and the resumes*") matched only `Camp_Ethnosphere...pdf`
- "resumes" is plural and generic, so it doesn't hit any single resume's
distinctive token, and the filter has no way to know a second document
*type* is needed when it's referenced that generically rather than by
name. The final answer was still correct (9820377527 - the Camp form
lists it directly), but the model asserted it "appears on both" without
the resume text actually being in its filtered context to verify that.
Got the right number for the wrong (unverifiable, from its own context)
reason. Every other multi-document question that named its sources
explicitly (e.g. "the Aadhaar card and the Camp Ethnosphere form") matched
both and was unaffected. **Lexical filename matching is a real, working
fix for the failure mode it targets, and it has a known, narrow blind
spot: questions that reference a document category generically instead of
by name or keyword.**

**Speed**: filtering isn't just free, it's slightly net-positive - narrowing
the candidate pool before ranking means the on-topic answers also carry
less irrelevant context into the prompt, so prefill is a bit shorter on
the questions where it fires:

| | Warm TTFT p50 | tok/s |
|---|---|---|
| llamacpp-4b (unfiltered) | 2358 ms | 43.6 |
| llamacpp-4b-filtered | 2105 ms | 44.6 |

## Addendum 7 — the reranker: built, tested, net-negative as configured (like hybrid retrieval in Addendum 6)

Built `rag/retrieve/rerank.py` - `BAAI/bge-reranker-base` as a second pass
over the `candidates`-sized pool (before, `retrieve` truncated straight to
`top_k`; now it keeps the full candidate pool and an optional rerank step
picks `top_k` from that). Also added `harness/regression.json`, a 13-
question fast subset (the questions that actually caught a bug or a
size-dependent break this session, plus every refusal check) and a
`bench.run --regression` flag, so iterating on retrieval changes doesn't
require a full 25-question run every time - exactly the workflow this
addendum needed.

**First result looked like a clean win**: reranker alone fixed the HCL
years-of-experience bug (same root cause as Addendum 5/6 - correctly
re-scored the right resume's chunk above the wrong one) *without* metadata
filtering's help. Promising.

**Running the full 13-question regression set found two real regressions**,
not caught by the single-question spot check:

1. *"How many Mumbai pick-up points..."* - correct (9, listed) on every
   prior config - came back **"the context does not mention any pick-up
   points"** with the reranker on. The correct chunk (all 9 points, intact,
   verified by direct inspection) *was* in the final context. Reproduced
   and diffed dense-only vs. reranked top-5: dense-only pulled the pickup
   chunk plus 4 itinerary pages (topically coherent, reinforces "this is
   about a trip"); the reranker swapped 3 of those 4 itinerary pages for
   resume content, cutting document diversity from 2 sources to 3
   unrelated ones and apparently confusing the model about whether "the
   trip" the question means is even represented in its context.
2. *"What is Jalpesh's father's name?"* - correctly refused everywhere
   else - **came back "Jalpesh's father's name is Hetansh Rajani"** with
   the reranker on. Hetansh is Jalpesh's *son* (the camp participant) -
   the model inverted the relationship. Diffed retrieval again: the
   reranker dropped `Aadhaar.pdf` in favor of a *second, different* chunk
   from the same HCL-resume page it had already included (page 1 splits
   into two real chunks at the 400-token boundary - confirmed by direct
   inspection, not a duplicate-selection bug), again reducing the number
   of distinct documents represented from 4 to 3, with the Camp
   Ethnosphere parent/child listing now dominating the context by
   proportion.

**Same mechanism both times**: `bge-reranker-base` scores each (query,
chunk) pair independently, with no awareness of which *documents* are
already represented in the picks. Nothing stops it from spending 2 of 5
context slots on two chunks from the same page while dropping a source
that, even if not obviously "relevant" by pure query-chunk similarity,
kept the context from over-concentrating on one document's framing. This
is a known, named failure mode in retrieval (why production rerankers
often pair with per-document diversity caps or MMR) - not a fluke.

**Verdict, scored honestly on the 13-question regression set**: +2 fixed
(both instances of the years-of-experience bug), -2 broken, one of them a
genuine hallucination on a previously-safe refusal - which is a worse
failure than either of the bugs it fixed (a silently wrong number vs. an
actively fabricated relationship). **Net negative, not net positive, as
currently configured.** Same call as hybrid retrieval in Addendum 6: built,
measured, kept **off by default** (`Config.reranker` defaults to `False`;
`llamacpp-4b-reranked` and `llamacpp-4b-best` stay in `GEN_MATRIX` as
documented, deliberately-not-default configs, not recommendations).

**What would actually fix it, untested**: cap reranked picks to N chunks
per source document (simple, a few lines), or blend the reranker score
with the original dense-retrieval rank instead of replacing it outright
(so a reranker mistake can demote a chunk but not evict a whole document
from context). Worth trying before this ever becomes the default - not
worth shipping on a "it fixed the one bug I was looking for" spot check
alone, which is exactly the trap the regression set just caught.

**On testing across other model sizes**: given this result, propagating
the reranker to `llamacpp-8b`/`1.7b`/`0.6b` right now would just spread a
known-negative change further. Metadata filtering (Addendum 6) is the one
validated as a clean, no-observed-regression win on this corpus - that's
the one worth checking against the other sizes next, using the regression
set for speed and the full 25 only once a size looks promising.

## Addendum 8 — metadata filtering across model sizes (regression set)

Added `llamacpp-{8b,1.7b,0.6b}-filtered`, ran the regression set on all
three, diffed against each size's original unfiltered full-25 answer on
the same 5 questions (matched by question text, not index - the two
question files don't share indices).

| Size | HCL years (target bug) | Jalpesh years | grad+years multihop | Jio dates agree | Mumbai count |
|---|---|---|---|---|---|
| 8b | fixed | already correct | already correct | already correct | already correct |
| 1.7b | fixed | already correct | already correct | already correct | **unchanged (still off by one)** |
| 0.6b | fixed | **changed, still wrong (19→2)** | fixed | **unchanged (still wrong)** | **unchanged (still wrong, 7→8)** |

**Clean win at 8b, 4b (Addendum 6), and 1.7b** - the retrieval-contamination
bug fixes with zero regressions, every time, at every size where the
underlying model is otherwise reliable. This is the generalization the
reranker (Addendum 7) didn't get to show.

**At 0.6b the picture is genuinely mixed, not a clean win** - filtering
fixed 2 of 5 checks and correctly left the other 2 untouched (counting and
cross-document comparison are generation-capacity limits, not retrieval
problems - filtering was never going to fix those, and didn't, exactly as
predicted from Addendum 3). But the Jalpesh-years factual question, right
next to two questions it fixed, went from one wrong answer (19) to a
different, worse one (2) - not a filtering-caused regression in the
reranker sense (no evidence retrieval got worse), more likely just 0.6b's
general answer-to-answer instability. **Takeaway: retrieval fixes transfer
across model sizes for retrieval-caused failures, but they don't buy back
reliability the base model doesn't have - and 0.6b doesn't reliably have
it for numeric extraction, filtered context or not.**

Practical read: ship metadata filtering as default for anything 1.7b and
up. Below that, treat every answer as needing a second check regardless of
what retrieval does.

## Addendum 9 — the two-lane service, built and tested

Built `service/app.py` (FastAPI, resident on startup, matching the
project's own "Pipeline as resident object" thesis from day one) and
`rag/route.py` (the fast/deep classifier). Both lanes share one embedder
and one index - only `Config.gen_model`/`gen_think`/`llama_host` differ,
so retrieval is never duplicated in memory.

**Router**: two free signals, no LLM call. Lexical (`across`, `combine`,
`compare`, `agree`, `summarise`, `how many`, ...) and `idx.matched_sources()`
- already built for metadata filtering - naming 2+ sources is itself a
complexity signal. Validated against the full 25-question eval by type:
all 6 unanswerable and 5/7 factual stayed fast (2 "how many years"
questions over-classify deep - harmless, just slower than necessary); 5/6
multihop and 6/6 aggregation routed deep. Self-check in
`python -m rag.route`.

**End-to-end test, real requests against the running service**:
- Fast lane: correct answer, streamed via the normal `Pipeline.query()`
  path, 3.4-5.0s wall time depending on answer length (TTFT 1.5-2.4s
  consistent with earlier benchmarks).
- Deep lane: `POST /query` returns a job id immediately (non-blocking,
  confirmed by timing the POST itself), `GET /jobs/{id}` polling found the
  correct answer after ~22s - TTFT 19.4s, matching the reasoning-tax
  measured earlier (Addendum on concurrency: 418 reasoning tokens ≈ 10s at
  4B; 8B's higher decode cost pushes it further).
- **Concurrency across the two separate llama-server processes** (not
  within one, which `bench/concurrency.py` already covers): fired fast
  queries while a deep job was mid-generation. First attempt read 8s
  (concerning); three controlled follow-ups, isolating for prefix-cache
  effects and firing at different points in the deep job's lifecycle, all
  came back under 1.2s - as fast or faster than standalone. Treat the 8s
  reading as a cold-start artifact, not evidence of real GPU contention
  between the two processes, but this deserves a proper sweep (many
  concurrent fast + several deep, sustained) before trusting it at real
  scale - `bench/concurrency.py` tests one process's internal slots, not
  two independent processes sharing one GPU, which is a different question
  this service actually creates.

**Known gap, not yet built**: `/ingest` re-embeds the whole corpus, same
non-incremental limitation flagged since the original design doc - fine
for this demo, a real problem for concurrent uploaders. Auth is a single
static bearer token via `RAG_API_KEY` - adequate for a LAN pilot, not for
production (no per-user identity, no rate limiting, no revocation).

## Addendum 10 — real corpus: 31 files, PDF + DOCX, 16x bigger

User added 31 real documents (legal order, tax report, bank statement,
whitepapers, game design doc, ~10 draft itineraries) - PDF and DOCX mixed.
Added DOCX parsing (`python-docx`, whole-file-as-one-page convention,
matching the existing `.txt`/`.md` handling - DOCX has no fixed page
concept to extract). 266 pages -> 383 chunks -> 376 after dedup, 93,913
tokens - 17x the corpus every prior finding in this document was measured
against.

**Ingest scales cleanly.** 3,519ms to embed 93,913 tokens (26,690 tok/s) -
now a real ~26% share of ingest instead of the ~3% it was before, finally
exercising batch/device the way the "Open items" list always wanted.
Index build stayed 0.2ms at 376 vectors - flat cosine still free, as
designed.

**Two real things this scale exposed, one already fixed:**

1. **Bug**: `rag/retrieve/filter.py`'s `_tokenize()` only stripped `.pdf`,
   leaving a dead `.docx` suffix glued onto every DOCX file's last token
   (`"restaurants.docx"` instead of `"restaurants"`), and no handling for
   a `" (1)"` duplicate-download suffix, so the exact-duplicate
   `Complete_Merged (1).docx` produced a garbage token. Fixed with
   `Path(filename).stem` (strips whatever extension is actually present)
   and a small regex for the `(N)` suffix. Both were invisible at the
   5-file, PDF-only corpus this bug was written against - exactly why
   testing against messier real data matters.
2. **Confirmed, not just predicted**: user chose to keep all ~10 near-
   duplicate itinerary drafts as a deliberate stress test. Checked
   directly - **8 of 13 itinerary-related files have zero distinctive
   filename tokens** (`Clickable`, both `Complete_Merged` copies,
   `Day_1_Polished_Numbered`, `Day_2_Polished_Numbered`, `DesertTheme`,
   `Final`, `v6_Final`, `v7_Final`). Metadata filtering's disambiguating
   power - the Addendum 6 fix - genuinely collapses on this cluster; there
   isn't enough distinctive vocabulary in 8 of these filenames to tell
   them apart. Expected, given the mechanism (Addendum 6 flagged this
   exact blind spot on a 2-file case; this is the same limit at 8 files).

**Real query against the cluster, unfiltered because the filter can't
fire**: retrieved chunks from 4 different draft files at once (`Final`,
`v7_Final`, `v6_Final`, `DesertTheme_TIMINGS_BOOKINGS`) for one question -
**answer stayed correct and consistent** (7:00-8:00 PM, matches every
earlier single-source test). The drafts agree on this particular fact, so
mixing them didn't cause the contamination Addendum 5-7 found on the
resume pair. Not proof they always agree - a fact that differs across
drafts (a changed date, a renumbered day) would very plausibly reproduce
that failure mode, untested here.

**TTFT**: first query against the new corpus (cold context, no prefix-
cache carryover from the old 5-file corpus) cost 5.4s - over the 3s
budget. Second query, same session: 2.2s, back in the measured baseline
range. One-time cost per context change, not a standing regression from
corpus size.

**Not yet tested**: OCR gap (9 scanned pages across 3 PDFs, flagged not
silently dropped, content still invisible to retrieval), and whether a
genuinely conflicting fact across itinerary drafts reproduces the
cross-document contamination failure mode at this larger scale - the
next thing worth checking before calling multi-draft handling safe.

## Addendum 11 — semantic cache + confidence escalation: built, calibrated, and one real miss found

Built `rag/cache.py` and `rag/confidence.py`, both calibrated against real
data instead of guessed thresholds, both wired into `service/app.py`.

**Cache threshold (0.93)**: measured real embedding similarity first -
exact repeat = 1.00, a genuine paraphrase = 0.92, a *different* question on
the same topic = 0.80. Set the threshold above the paraphrase score, not
between it and the same-topic score, because a wrong cached answer is worse
than a cache miss. Consequence, seen live: a loose paraphrase ("time listed
for the tournament" vs "reporting time") will safely miss and get a real
answer; only close paraphrases and near-exact repeats hit.

**Confidence signal - rejected the obvious one, tested before building**:
tried retrieval top-score as the escalation trigger first. Calibrated it
against real questions and killed it before writing any code: a plausible-
but-wrong question ("chess tournament prize money") scored *higher* (0.78)
than a genuinely answerable one (Aadhaar DOB, 0.64) - retrieval score
measures topical similarity, not whether the fact is present. Built
`is_low_confidence()` instead: flags an answer whose stated numbers don't
appear verbatim in the chunks it was retrieved from (the actual mechanism
behind the Addendum 5 "20 vs 19 years" bug - a computed number usually
doesn't match a quoted one) plus light hedge-phrase detection. Found and
fixed a real bug in the first version before trusting it: a citation's own
page number (`[source p.1]`) was being checked as if it were a claimed
fact, false-positiving on nearly every answer - fixed by stripping bracket
citations before extracting numbers, caught by testing against a known-
correct answer, not by inspection.

**Full 16-question second eval set** (`harness/questions_v2.json`,
covering the new 31-file corpus - chess tournament, tax report, court
order, whitepaper, game design doc, school worksheets, and a deliberate
itinerary-draft-conflict probe) run against the live service, not
`bench.run` directly, specifically to exercise cache + router + escalation
together (`bench/run_service.py`). Results:

- **Cache confirmed working**: repeating the first 3 questions returned
  `lane: cache`, `cache_sim: 1.0` for all three, correctly skipping
  retrieval and generation.
- **3 of my own "unanswerable" questions were mislabeled**, not system
  failures - built from a ~200-char peek at each document instead of
  reading them properly. The chess tournament's prize fund turned out to
  be itemized non-monetary prizes (trophies, watches), the game design doc
  explicitly specifies its backend stack on page 22, and the court order
  does contain an operative ruling. The system found all three correctly;
  my ground truth was wrong. Same lesson the original design doc gave
  about the first eval set, ignored under time pressure on the second one.
- **One escalation fired, and it was a real, explainable, non-hallucination
  case**: "Dinner: 8:30 PM to 10:30 PM" flagged `ungrounded numbers: ['10',
  '30']`. Traced it - the source PDF's stylized menu heading extracts as
  `"B R E A K F A S T ( 8 : 3 0 A M T O 1 0 : 3 0 A M )"` (space between
  every character, a common artifact of decorative/kerned PDF text), so
  "30" as a two-digit unit never appears in the extracted context text at
  all, only isolated "3" and "0". A real, understood limitation of a
  string-matching check, not evidence the check is broken - flagging a
  correct-but-oddly-formatted answer for a second look is a defensible
  failure mode for a cheap safety net to have.
- **The one that matters**: the itinerary-draft-conflict probe question
  ("Do the 'Optimized' and 'DesertTheme' drafts describe the same Day 1
  schedule?") got a **confirmed, confident hallucination** - "Yes... Both
  include arrival, check-in, a City Palace visit, Bagore Ki Haveli, and
  dinner at Ambrai/Upré/Raas Leela," citing only the `Optimized` file.
  Checked `DesertTheme`'s actual Day 1 text directly: no City Palace, no
  named restaurants - none of that is in it. The model had (at most) the
  `Optimized` draft's chunk in context and asserted its content also
  described the other file, unprompted. **Neither of this session's two
  new safety nets caught it**: the router didn't send it deep, because
  `DesertTheme` has zero distinctive filename tokens (the same collapse
  documented in Addendum 10) and my question happened not to use a trigger
  word ("describe the same" rather than "compare" or "differ"); the
  confidence check didn't fire, because the hallucination is a qualitative
  conflation of *descriptive* content, not an invented *number* - outside
  what `ungrounded_numbers()` was built to catch.

**Two real gaps, not papered over**: retrieval recall degrades at this
larger, more redundant corpus in ways that sometimes fail safely (Camp-vs-
itinerary date comparison correctly said "I don't have the itinerary's
dates" rather than guessing - a genuine miss, but the honest kind) and
sometimes doesn't (the chess-officials list silently dropped one of three
named people, incomplete without flagging it as such). And the
confidence check's blind spot is structural, not a bug to patch: a
number-grounding check will never catch a hallucination that isn't a
number. The next thing worth building, now that it's been demonstrated
rather than hypothesized, is a grounding check for *claims*, not just
digits - does each cited document actually appear among the retrieved
chunks used to answer about it - which the router already half-computes
via `matched_sources()` and could cross-check post-hoc against what the
answer actually cites.

## Addendum 12 — claims-grounding check, closing the Addendum 11 gap

Added `missing_named_sources()` to `rag/confidence.py`: reuses
`idx.matched_sources()` (already built for metadata filtering) to get the
documents a question explicitly names, and checks that set against the
documents actually retrieved. If the question names 2+ documents but
retrieval only covered a subset - exactly what happened in the confirmed
hallucination - it's flagged. Wired into `service/app.py`'s escalation
path alongside the numeric check. Self-check reproduces the actual
Addendum 11 case with the real retrieved-sources list (all 5 chunks from
`Optimized.pdf`, `DesertTheme.pdf` named but absent) and confirms it now
fires - first version of the test accidentally passed for the wrong
reason (empty context tripped the numeric check instead), caught by
checking the returned reason string, not just the boolean.

## Addendum 13 — the draft-conflict question, finally tested: silent majority-vote, both lanes

The open question left since Addendum 10: does a *genuinely conflicting*
fact across two itinerary drafts (not just a missing document, which
Addendum 11/12 already covered) reproduce a faithfulness failure? Needed a
real conflict first - found one by inspection, not assumption: the PDF
family of drafts (`Clickable`, `Final`, `Optimized`, `v6_Final`,
`v7_Final`, `DesertTheme*`) states Day 1 as **"Thu, 28 Nov"**; the DOCX
family (`Complete_Merged`, `Day_1_Polished_Numbered`,
`Day_1_Final_Clickable_Restaurants`, `Day_1_Final_Search_Links`) states
the same Day 1 as **"Thu, 28 Dec"**. Same trip, same day label, different
month - a real, verifiable disagreement between documents, not a
constructed one.

Confirmed both families retrieve together before testing anything else:
"What is the exact date of Day 1...?" pulled 4 Dec-family chunks and 1
Nov-family chunk (`Optimized.pdf`) into the same top-5.

**Result, fast lane**: *"...is Thursday, 28 Dec [Day_1_Polished_Numbered.docx
p.1]."* Confident, single answer, zero acknowledgment that a directly
contradicting source was sitting in the same context. `low_confidence:
false` - neither existing check fired.

**Result, deep lane (forced via `force_lane`, 47s, full reasoning
budget)**: identical 5 sources, identical answer - *"28 December"* - same
silent pick, same lack of acknowledgment. More thinking time did not
surface the conflict. This rules out "the fast model just isn't careful
enough" as the explanation - it's not a capability gap, it's that nothing
in the pipeline ever asks the question "do my sources agree with each
other," at either model size.

**Why neither existing safety net catches this, precisely**:
- `ungrounded_numbers()` (Addendum 5) checks whether a stated number
  appears *somewhere* in the retrieved text. "28" appears in both the Nov
  and Dec chunks, so the check is satisfied - it was never designed to
  notice that "28" is attached to two different, contradictory month
  labels across sources.
- `missing_named_sources()` (Addendum 12) checks whether every document
  the *question* explicitly names was actually retrieved. This question
  never names a specific draft, so there's nothing to cross-check against
  - the check has no opinion about documents disagreeing with each other,
  only about a named document being silently absent.

Both checks are answering "is this claim grounded/complete" - correctly,
for what they were built to catch (Addendum 5's computed-not-quoted
numbers, Addendum 11's conflation-across-absent-sources). Neither one
asks "do the *present*, correctly-retrieved sources contradict each
other," which is a genuinely different failure mode this question was
specifically designed to isolate, and did.

**What would actually catch this, unbuilt**: a same-slot contradiction
check - for chunks retrieved together, extract comparable claims tagged
with the same apparent referent (a date attached to "Day 1," a name
attached to "father," a number attached to "years of experience") and
flag when two retrieved chunks assert different values for the same slot,
regardless of what the model does with them. This is a different
mechanism from both existing checks - it operates on the *chunks*, before
generation, not on the *answer*, after. Worth prioritizing over the
reranker fix still sitting in Open Items below, since this is a confirmed
gap and the reranker's is a hypothesized one.

## Addendum 14 — the contradiction check: built, verified live, and one thing it doesn't do

Built `conflicting_day_dates()` in `rag/confidence.py` per Addendum 13's
proposal: a narrow, deliberate first instance (Day-N-to-date claims only,
not a general claim-extraction engine) of "do retrieved chunks disagree,"
wired into `is_low_confidence()` as a third signal. Self-check reproduces
the real Nov/Dec text verbatim plus a negative case (different Day N's
must never be flagged as conflicting with each other).

**Verified against the live failure, not just the unit test**: re-ran the
exact question from Addendum 13 against the running service.
`low_confidence: true`, with the precise reason - *"sources disagree on
Day 1's date: 28 Dec=Day_1_Polished_Numbered.docx, 28 Nov=Optimized.pdf"* -
and it auto-escalated. Detection confirmed working end to end.

**Then checked what the escalation actually produces, since Addendum 13
already showed the deep lane alone doesn't fix this**: the verification
job's answer was, again, *"Thursday, 28 December"* - no mention of the
conflict. **Detecting the contradiction and fixing the answer are two
different problems, and this only solves the first one.** The escalation
mechanism re-runs the identical question through the deep lane with no
indication of *why* it's being re-run or *what* disagreement was found -
the deep model has no more reason to surface the conflict than the fast
one did, because nothing tells it one exists. The `low_confidence_reason`
is visible in the API response and to whoever reads it, which has real
value (an operator or the eval harness can see exactly why this answer is
suspect) - but the end user talking to the fast lane gets the same
silently-wrong-shaped answer either way.

**What would close this, unbuilt**: feed the detected conflict into the
escalation prompt itself - "sources disagree on Day 1's date (28 Nov per
X, 28 Dec per Y); answer honestly, citing the disagreement" - rather than
just re-asking the bare question on a bigger model. This is a small,
specific change (one extra sentence in `build_prompt()` when
`conflicting_day_dates()` fires) versus the open-ended "how would a model
ever know to check" problem detection was solving - worth doing before
extending the contradiction pattern to other fact types, since a detector
that doesn't inform the fix it triggers is only half the value.

## Addendum 15 — closing the Addendum 14 gap: tell the model, don't just flag it

Addendum 14 detected the Day-1 date conflict correctly but didn't fix the
answer - escalation re-asked the same bare question and got the same
silent pick. Fix: `build_prompt()` (`rag/generate/prompt.py`) now takes an
optional `conflicts` argument and appends an explicit notice - "your
sources disagree... do not silently pick one" - naming every conflicting
value and its source. `Pipeline.query()` computes
`conflicting_day_dates(hits)` right after retrieval, before the prompt is
built, so the *first* answer gets the fix, not just a later re-verification.

**Verified live, both lanes, on the exact question that exposed the gap**:

| | Before (Addendum 14) | After |
|---|---|---|
| Fast lane | "28 Dec" - silent pick | *"...disputed among the sources. [Day_1_Polished_Numbered.docx] says 28 Dec. However, [Optimized.pdf] states 28 Nov. Thus, the sources disagree."* |
| Deep lane (escalation) | "28 December" - identical silent pick | *"The sources disagree... .docx files state 28 Dec... Optimized.pdf states 28 Nov... No other documents provide a date for Day 1."* |

Both lanes now name both values and both sources, unprompted beyond the
injected notice. Still flags `low_confidence: true` and still escalates -
that's fine, detection and the honesty fix are independent; the escalation
now just reinforces an already-correct answer instead of quietly
re-committing the same error on a bigger model.

**Regression set re-run clean** (13/13, no errors, no empty answers) -
important because this change touches every prompt-build step, not just
conflicting ones. One question ("what flight number...", unrelated to
dates) incidentally retrieved conflicting Day-1 chunks and got the notice
anyway; the model correctly judged the conflict irrelevant to what was
asked, mentioned it in one clause, and still gave the correct refusal on
the actual question. Not a false-positive problem worth tightening -
the model's own relevance judgment absorbed the noise without being
derailed by it.

## Addendum 16 — structure-aware chunking: built, tested, real bug caught, no clear win

Built `rag/ingest/structured.py` as an opt-in alternative
(`Config.chunk_strategy = "structured"`, default stays `"sliding"` -
prototype before adopting, per the chunk-size discussion this addendum
answers): PDF paragraphs via pymupdf's own `get_text("blocks")` layout
analysis instead of blind token-count slicing; DOCX sections via python-
docx's actual `Heading 1/2/3` style metadata (thrown away entirely by the
existing flat-text `parse_docx`); tables kept as one unit each, never
sliced mid-row. Units are greedily packed up to the token budget; only a
single oversized unit falls back to the ordinary sliding window.

**A real bug caught by testing against the actual Bullforce whitepaper,
not the unit tests**: the first version let packed units span a page
boundary under one page-number tag - a chunk citing `page: 1` while half
its content was actually page 2's table of contents. Wrong page numbers
undermine the one thing this project's citations exist for (verifiability).
Fixed by flushing the pack buffer on any page change, confirmed against
the real document (chunk count 18->9, each chunk now provably single-
page), and added as a regression test with the exact scenario.

**Also caught defensively**: `python-docx` returns `paragraph.style` as
`None` for some paragraphs in the real corpus (`Legal_Risk_Analysis_
Shop9A3_Rajani.docx`) - crashed a naive `.style.name` check immediately;
handled before it reached the actual parser.

**Regression set re-run, full honest comparison against sliding-window,
not just a pass/fail**: 13/13 clean on both, one measurable efficiency win,
and two questions worth reporting precisely rather than rounding to "it's
better":

- **Efficiency, real**: 82,318 embedded tokens vs sliding-window's 93,913
  (~12% fewer) - packing whole units produces zero redundant overlap,
  where the sliding window's 20% overlap duplicates content near every
  boundary. Fewer tokens to embed, same corpus.
- **11 of 13 questions**: same substance, wording variance only - the
  normal run-to-run variance already documented in Addendum 2.
- **One question improved, but not for the reason chunking strategy
  should get credit for**: "do the two resumes agree on Reliance Jio
  dates" - sliding-window retrieved *zero* resume chunks (same known
  "resumes"-plural-doesn't-match-"resume"-singular retrieval-ranking
  weakness documented back when `matched_sources()` was first built - two
  unrelated documents, a legal filing and a court order, score higher by
  raw cosine similarity than the actual resumes for this generic
  phrasing, confirmed reproducible right now, not drift). Structured
  chunking's different unit boundaries happened to let one resume chunk
  win a top-5 slot this time, and the model answered correctly citing
  only that one chunk - technically still asserting agreement about a
  *second* resume it never saw, the same class of issue Addendum 11 named,
  just landing on the right answer this time. Not a robust fix; a
  different roll on the same underlying weakness.
- **One question got worse**: "Mumbai pick-up points" - sliding-window's
  top-5 happened to include the one correct chunk (`Camp_Ethnosphere...
  p.1`) alongside noise, giving a correct-but-messy answer; structured's
  top-5 missed it entirely, giving an honest refusal instead. Arguably the
  *safer* failure mode (refusing beats a wrong number), but it's a miss
  against ground truth either way.

**Honest verdict**: not a clear win or loss on this regression set - a
real, measured efficiency improvement, and confirmation that *which*
five chunks win a top-5 ranking is sensitive to exactly how content got
packed, in both directions, on a corpus this size. The known "resumes"-
plural retrieval weakness is unaffected by chunking strategy either way -
that's a `matched_sources()` matching-rule problem, not a chunk-boundary
problem, and remains open regardless of which chunker is used.

## Open items for next pass

- Re-run the matrix on a corpus large enough (thousands of chunks) that
  `embed_chunks` is more than 1% of ingest time, so batch/device actually
  get tested.
- ~~The real latency lever is `generate`, not retrieval: try a smaller
  model~~ — done in Addendum 2. `qwen3:4b` via `llamacpp` is the best
  result so far (2.4s TTFT, 43.6 tok/s). Next: go smaller still
  (`qwen3:1.7b`, `qwen3:0.6b`) and see where answer quality on the
  multihop/aggregation questions actually starts to break — that's the
  real floor, not raw speed, which will keep improving.
- Lower `max_tokens` / smaller `top_k` (fewer retrieved chunks) as a
  cheaper lever than model size — prefill scales with context tokens, and
  this doesn't cost any quality on the embed side, unlike shrinking the model.
- Software/hardware ceiling: this machine (M3 Pro) is compute-bound on
  prefill, not bandwidth-bound — quantization won't move TTFT further,
  only model size or context size will. A discrete NVIDIA GPU
  (vLLM/TensorRT-LLM) is the only path shown so far that plausibly closes
  the remaining ~5-10x gap to hosted APIs; nothing in the Apple Silicon
  lineup (Max/Ultra) closes it, only narrows it.
- ~~`harness/spans.py`'s `_rss_mb()` had an inverted unit-detection
  heuristic~~ — fixed (now keys off `platform.system()` instead of
  guessing from magnitude). Didn't affect any latency finding above; old
  result files still have RSS ~1024x too large, new runs are correct.
