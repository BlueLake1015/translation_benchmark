# translation_benchmark

Test and benchmark harness for open-weight **machine translation models** on
**real-world film/video subtitles (`.srt`)**. The primary evaluation direction
is **English → Korean**, but any language pair supported by the models works.

- Python **3.11**
- Translates `.srt` files while preserving cue indices and timings
- **Context-aware translation**: models that support context get a rolling
  window of previous (source → target) subtitle pairs, so pronouns, register
  (반말/존댓말), names, and running references stay consistent across a scene
- Speed (segments/s, chars/s) + quality (chrF++, BLEU, optional COMET) reports
  in JSON and Markdown

## Install

Requires Python 3.11.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .              # core: srt parsing, metrics, CLI
pip install -e '.[models]'    # + torch/transformers (real model inference)
pip install -e '.[vllm]'      # + vLLM (optimized serving, chat models, GPU)
pip install -e '.[ct2]'       # + CTranslate2 (CPU NLLB backend)
pip install -e '.[comet]'     # + COMET neural metric
pip install -e '.[dev]'       # + pytest
```

### Offline install

For air-gapped machines, two scripts handle the package side (model weights
are covered by the `models/` directory above):

```bash
# On a connected machine: download all wheels into offline/
scripts/download_offline.sh                # full stack: dev,models,ct2,vllm (~10 GB)
scripts/download_offline.sh dev            # CPU-only test profile (~35 MB)

# On the target machine (same OS/arch/Python 3.11): venv + install, no network
scripts/install_offline.sh .venv           # extras must match the download
scripts/install_offline.sh .venv dev
```

`offline/` ends up with pip/setuptools/wheel (venv bootstrap), the project
wheel itself, and every dependency wheel — `install_offline.sh` uses
`--no-index --find-links offline/` only. Wheels are platform-specific:
download on a machine matching the offline target (linux x86_64, CPython
3.11).

## Testing

The test suite targets **English → Korean** and runs anywhere — no GPU, no
model downloads. It exercises the full pipeline (`.srt` parsing → context
threading → translation → `.srt` writing → benchmark reports) against a
real-world-style film subtitle fixture
([night_shift.en.srt](tests/data/night_shift.en.srt) with a Korean reference
[night_shift.ko.srt](tests/data/night_shift.ko.srt)) using a deterministic
dummy backend; real model backends are construction-tested without loading
weights.

```bash
pip install -e '.[dev]'
pytest
```

### ⚡ Real model translation runs (outside pytest)

Real-model runs are deliberately **not part of the test suite** — they live
in the standalone [scripts/run_models.py](scripts/run_models.py):

```bash
python scripts/run_models.py                          # every model, every engine
python scripts/run_models.py tower-plus-9b --input movie.en.srt --output-dir /tmp/out
python scripts/run_models.py --engine transformers    # one engine only
python scripts/run_models.py qwen3-32b --quant awq
```

It sweeps every selected model across **all of its supported engines**
(falling back to a downloaded quant variant like `qwen3-32b@awq` if only
that is present) and prints a per-(model, engine) result summary —
**SUCCESS** (cues, speed, guard flags), **SKIP** (weights or engine extra
missing — never an error), or **FAIL** (with the reason). Each output is
named **`tests/test_result/<model-key>.<engine>.<input-stem>.<tgt>.srt`**,
so a model's vLLM and transformers results sit side by side for review.

Each (model, engine) runs in **its own subprocess** so GPU memory is fully
reclaimed between runs — some in-process loads (e.g. AWQ via gptqmodel)
don't release on unload, which would otherwise starve the next vLLM server.
Logs still stream live; `--no-isolate` runs everything in one process.

Stage weights first with `scripts/download_models.py`
(+ `scripts/convert_ct2.py` for Tier 4).

## Supported models

### Tier 1 — Frontier open-weight (server-class hardware)

| Model | Key | Context | VRAM | Notes |
|---|---|---|---|---|
| TranslateGemma 27B | `translategemma-27b` | ✅ (~128k tokens) | ~16 GB at 4-bit | 3.09 MetricX on WMT24++; best dedicated translation specialist |

### Tier 2 — Strong, single high-end GPU (16–24 GB)

| Model | Key | Context | VRAM | Notes |
|---|---|---|---|---|
| Qwen3-32B | `qwen3-32b` | ✅ (~32k tokens) | ~20 GB at 4-bit | Best general-purpose option at this size; strongest indirect hallucination evidence in the field |
| TranslateGemma 12B | `translategemma-12b` | ✅ (~128k tokens) | ~8 GB at 4-bit | Beats a 27B generalist baseline at under half the size |
| Tower+ 9B | `tower-plus-9b` | ✅ (~8k tokens) | ~6 GB at 4-bit | 84.38 XCOMET on its 24 supported pairs, best in its weight class among open models; **Korean explicitly supported** |
| Qwen3-14B | `qwen3-14b` | ✅ (~32k tokens) | ~10 GB at 4-bit | Slightly below the three above; excellent quality-per-GB |

### Tier 3 — Lightweight (laptop / small GPU)

| Model | Key | Context | VRAM | Notes |
|---|---|---|---|---|
| TranslateGemma 4B | `translategemma-4b` | ✅ (~128k tokens) | ~3 GB quantized | Matches 12B-class generalist quality |
| TowerInstruct-7B-v0.2 | `towerinstruct-7b-v0.2` | ✅ (~4k tokens) | ~5 GB at 4-bit | Previous Tower generation; superseded but still decent for its 10 languages including Korean |

### Tier 4 — Draft engines (⚠️ sentence-level only)

> **Note:** Tier 4 models do **not** support document context or
> instructions. They are encoder-decoder MT systems that translate **each
> subtitle line in isolation** — no rolling dialogue window, no consistency
> across cues, no register control. Use them for fast first-pass drafts only;
> the harness automatically skips context injection for them.

| Model | Key | Context | Hardware | Notes |
|---|---|---|---|---|
| MADLAD-400 10B | `madlad400-10b` | ❌ sentence-level | ~7 GB at int8 | Wide language coverage, literal output; first-pass drafts only |
| NLLB-200 3.3B | `nllb200-3.3b` | ❌ sentence-level | CPU-viable via CTranslate2 | Cheapest to run; flattest dialogue quality |

Hugging Face repo ids are defaults in
[registry.py](src/translation_benchmark/models/registry.py) and can be
overridden per run with `--hf-id` (e.g. to point at a quantized checkpoint).

## Model storage: the `models/` directory

All model weights live under the project-local [models/](models/) directory
by default — not the global Hugging Face cache. Override the location with
`--models-dir` or the `TB_MODELS_DIR` environment variable.

Layout: base weights in `models/<model-key>/`, pre-quantized variant repos
in `models/<model-key>@<variant>/` (e.g. `models/qwen3-14b@awq`), CTranslate2
conversions in `models/<model-key>-ct2/`. Model keys are listed by
`tb list-models`. The directory is git-ignored; weights never get committed.

- `models/<model-key>/` containing `config.json` is treated as a complete
  local copy and loaded directly (fully offline). The bundled downloader
  fills this layout for you:

  ```bash
  python scripts/download_models.py --dry-run         # show plan + sizes
  python scripts/download_models.py                   # all 9 models (default)
  python scripts/download_models.py tower-plus-9b translategemma-4b
  python scripts/download_models.py --tier 3 --tier 4
  ```

  Already-present models are skipped (`--force` re-downloads); gated repos
  (TranslateGemma) need an accepted license plus `--token` / `HF_TOKEN` /
  `huggingface-cli login`. Equivalent manual commands:

  ```bash
  huggingface-cli download Unbabel/Tower-Plus-9B --local-dir models/tower-plus-9b
  huggingface-cli download google/translategemma-4b-it --local-dir models/translategemma-4b
  ```

- **Quantization variants** are downloaded too (by default — filter with
  `--quant full`, `--quant awq`, …). Pre-quantized repos land in
  `models/<model-key>@<variant>/` (e.g. `models/qwen3-14b@awq`).

- **Exception:** for the Qwen3 models the full-precision (bf16) base weights
  are skipped by default — the official AWQ/FP8 variants cover usage at a
  fraction of the size. Opt in explicitly with `--quant full`.

- Otherwise the model is downloaded from Hugging Face **into `models/`**
  (used as the download cache), so weights always stay inside the project.

CTranslate2 conversions (the default engine for Tier 4) live there too, at
`models/<model-key>-ct2`. The bundled converter handles them from the
locally downloaded base weights:

```bash
python scripts/convert_ct2.py --dry-run   # plan; flags missing base weights
python scripts/convert_ct2.py             # all ct2-capable models, int8
python scripts/convert_ct2.py nllb200-3.3b --quantization int8_float16
```

(equivalent manual command: `ct2-transformers-converter --model
google/madlad400-10b-mt --output_dir models/madlad400-10b-ct2
--quantization int8`)

### Quantization at usage time

Every `tb translate` / `tb benchmark` run accepts `--quant`:

| `--quant` | What it does | Extra download? |
|---|---|---|
| *(default)* | Base weights, `torch_dtype="auto"` | — |
| `4bit` / `8bit` | On-the-fly bitsandbytes quantization of the base weights | no |
| `awq`, `fp8` (Qwen3-32B/14B) | Loads the official pre-quantized repo from `models/<key>@<variant>` | yes |

`tb list-models` shows the accepted `--quant` values per model; an unknown
variant fails fast with the available list. An explicit `--hf-id` overrides
the variant repo. NLLB's CTranslate2 int8 path is separate (`--ct2-dir`).

Loading an `awq`/`gptq` repo on the **transformers** engine needs
`gptqmodel` (bundled in the `.[models]` extra); the **vllm** engine has
native AWQ/FP8 kernels and needs nothing extra.

## Inference engines

Every model is served by its **optimized engine by default**; plain
transformers is the explicit fallback (`--engine transformers`):

| Models | Default engine | Fallback | Notes |
|---|---|---|---|
| Chat models (Tiers 1–3) | `vllm` | `transformers` | Spawns a `vllm serve` process on the local model dir and talks the OpenAI-compatible API; the server is terminated on unload and at exit. Supports `--quant 4bit/awq/fp8` (not `8bit`). Qwen/Tower use the chat endpoint (server-side templates); TranslateGemma's template needs custom content fields that vLLM's chat endpoint strips (≤0.22), so its requests render the checkpoint template client-side and go to `/v1/completions` |
| MADLAD-400, NLLB-200 (Tier 4) | `ct2` (CTranslate2) | `transformers` | int8, CPU-viable; needs a one-time `ct2-transformers-converter` conversion into `models/<key>-ct2` (a clear error tells you the exact command if missing) |

`tb list-models` shows the engines per model. Both chat engines share the
same prompts and rolling-context handling, so quality comparisons across
engines are apples-to-apples. Note that context-aware document translation
is sequential by nature (each line's context includes the previous
translation), so vLLM's batching shines most with `--context-window 0` or on
long files; otherwise the win is faster prefill/decode.

```bash
tb benchmark movie.en.srt -m qwen3-32b --quant awq -r movie.ko.srt   # vLLM by default
tb translate movie.en.srt -m madlad400-10b                            # CTranslate2 by default
tb translate movie.en.srt -m tower-plus-9b --engine transformers      # explicit fallback
```

## Hallucination mitigation

LLM translators hallucinate in characteristic ways — runaway repetition
loops, meta text ("Here is the translation:"), source copy-through, wrong
output language, invented content, dropped content. TranslateGemma-class
specialists are not immune. The harness mitigates at three levels (on by
default; disable with `--no-guard`):

1. **Decoding caps** — chat engines limit generation length proportionally
   to each source line, so a repetition loop is cut early instead of
   running to the token limit; greedy decoding throughout.
2. **Per-line detection** ([guards.py](src/translation_benchmark/guards.py)) —
   after cleaning echoed labels/quotes/commentary, each line is checked for:
   empty output, source copy, repetition loops, length explosion (>3×) or
   truncation (<0.15×), wrong output script (e.g. a Korean target with
   almost no Hangul), and meta/refusal text.
3. **Retry + context quarantine** — a flagged line is retried once
   *without* context (a poisoned window is the main driver of propagated
   hallucinations, and changing the prompt is the only lever under greedy
   decoding); whichever attempt has fewer issues wins. Lines that stay
   flagged are **never pushed into the rolling context**, so one bad line
   cannot poison the rest of the scene.

Findings are surfaced everywhere: `tb translate` prints a warning listing
flagged cue numbers for review, and `tb benchmark` reports a **Flagged**
column plus per-code counts (`issue_counts`) in `report.json` — so models
can be compared on hallucination rate, not just chrF++.

## How context is used

For every context-capable model (Tiers 1–3), the harness translates a
subtitle file **line by line in document order**, feeding each request a
rolling window of the most recent source → target pairs (default: 8 pairs,
capped at 2,400 characters — both configurable via `--context-window` and the
API). The model's own previous outputs become the context for the next line,
mirroring how a human subtitler works through a scene:

```
Previous subtitle lines (English -> Korean):
English: You're late, detective.
Korean: 늦었군, 형사.
English: Traffic. And a body on the Fifth Street bridge.
Korean: 차가 막혔습니다. 그리고 5번가 다리에서 시체가 나왔고요.

Translate the next subtitle line from English to Korean. Output only the translation.
English: The same bridge as last month?
```

Each chat-model family gets a **dedicated prompt matching its training
distribution** ([chat.py](src/translation_benchmark/models/chat.py)) —
off-format prompts measurably degrade specialists and are themselves a
hallucination trigger:

- `translategemma` — minimal canonical translation format, **no system
  prompt** (specialists drift when wrapped in meta-instructions)
- `tower` — the exact `English: … / Korean:` pattern Tower was tuned on,
  no system prompt
- `qwen` — detailed system prompt + instruction (generalists reward
  instruction-heavy prompting); thinking mode disabled

The tuned style is registry data (`prompt_style`); override it per run with
`--prompt-style` to A/B a prompt change against your reference file —
guards are model-agnostic, so flag counts stay comparable across styles.
Tier 4 models bypass all of this and receive bare sentences in batches.

## Usage

```bash
# Show the model lineup (tiers, context support, VRAM)
tb list-models

# Translate a subtitle file (English -> Korean), preserving timings
tb translate movie.en.srt -m tower-plus-9b -s en -t ko -o movie.ko.srt

# Use a bigger context window, on-the-fly 4-bit quantization
tb translate movie.en.srt -m translategemma-27b --context-window 16 --quant 4bit

# Use an official pre-quantized variant repo (Qwen ships AWQ and FP8)
tb translate movie.en.srt -m qwen3-14b --quant awq

# Tier 4 draft engine on CPU via CTranslate2 (default lookup: models/nllb200-3.3b-ct2)
ct2-transformers-converter --model facebook/nllb-200-3.3B \
    --output_dir models/nllb200-3.3b-ct2 --quantization int8
tb translate movie.en.srt -m nllb200-3.3b

# Benchmark several models against a reference subtitle file
tb benchmark tests/data/night_shift.en.srt \
    -m translategemma-4b -m tower-plus-9b -m nllb200-3.3b \
    -s en -t ko \
    -r tests/data/night_shift.ko.srt \
    --output-dir results
```

`tb benchmark` writes `results/report.json` and `results/report.md` with a
leaderboard:

| Model | Tier | Context | Segments | Time (s) | Seg/s | chrF++ | BLEU | Status |
|---|---|---|---|---|---|---|---|---|
| … | … | yes / sentence-level | … | … | … | … | … | ok |

Timing measures translation throughput only — model load/download happens
before the clock starts. chrF++ is the primary quality metric (tokenizer-free,
appropriate for Korean); BLEU uses character tokenization for ko/ja/zh.

## Project layout

```
src/translation_benchmark/
├── cli.py              # tb list-models | translate | benchmark
├── subtitles.py        # .srt parse/clean/write (markup stripping, timing-safe)
├── context.py          # rolling (source, target) context window
├── langs.py            # name / FLORES-200 / MADLAD code mappings
├── models/
│   ├── registry.py     # the 9-model lineup + factory
│   ├── base.py         # translator interface, document-order translation
│   ├── paths.py        # local models/ directory resolution
│   ├── chat.py         # TranslateGemma / Qwen3 / Tower (context-aware)
│   ├── vllm_engine.py  # optional vLLM serving for the chat models
│   ├── seq2seq.py      # MADLAD-400, NLLB-200 (+CTranslate2) — sentence-level
│   └── dummy.py        # deterministic backend for tests
└── benchmark/
    ├── runner.py       # timing + scoring + JSON/Markdown reports
    └── metrics.py      # chrF++, BLEU, optional COMET
```
