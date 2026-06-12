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

Per-family prompt formats (TranslateGemma/Qwen3 generic instruction style,
Tower's tuned `English: … / Korean:` pattern, Qwen3 thinking mode disabled)
live in [chat.py](src/translation_benchmark/models/chat.py). Tier 4 models
bypass all of this and receive bare sentences in batches.

## Install

Requires Python 3.11.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .              # core: srt parsing, metrics, CLI
pip install -e '.[models]'    # + torch/transformers (real model inference)
pip install -e '.[ct2]'       # + CTranslate2 (CPU NLLB backend)
pip install -e '.[comet]'     # + COMET neural metric
pip install -e '.[dev]'       # + pytest
```

## Usage

```bash
# Show the model lineup (tiers, context support, VRAM)
tb list-models

# Translate a subtitle file (English -> Korean), preserving timings
tb translate movie.en.srt -m tower-plus-9b -s en -t ko -o movie.ko.srt

# Use a bigger context window, 4-bit quantization
tb translate movie.en.srt -m translategemma-27b --context-window 16 --load-in-4bit

# Tier 4 draft engine on CPU via CTranslate2
ct2-transformers-converter --model facebook/nllb-200-3.3B \
    --output_dir nllb-ct2 --quantization int8
tb translate movie.en.srt -m nllb200-3.3b --ct2-dir nllb-ct2

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
│   ├── chat.py         # TranslateGemma / Qwen3 / Tower (context-aware)
│   ├── seq2seq.py      # MADLAD-400, NLLB-200 (+CTranslate2) — sentence-level
│   └── dummy.py        # deterministic backend for tests
└── benchmark/
    ├── runner.py       # timing + scoring + JSON/Markdown reports
    └── metrics.py      # chrF++, BLEU, optional COMET
```
