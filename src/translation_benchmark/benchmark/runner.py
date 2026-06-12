"""Benchmark runner: translate a subtitle file with one or more models and
report speed plus (when a reference .srt is given) quality metrics."""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass

from translation_benchmark import guards
from translation_benchmark.benchmark import metrics
from translation_benchmark.models.base import BaseTranslator
from translation_benchmark.subtitles import SubtitleLine, prepare_for_translation


@dataclass
class BenchmarkResult:
    model_key: str
    display_name: str
    tier: int
    supports_context: bool
    num_segments: int
    source_chars: int
    wall_seconds: float
    segments_per_second: float
    chars_per_second: float
    chrf: float | None  # chrF++, 0-100; None without a reference
    bleu: float | None
    error: str | None = None
    translations: list[str] | None = None
    # Hallucination guard findings (None when guards are disabled or on error)
    flagged_segments: int | None = None
    issue_counts: dict[str, int] | None = None


def run_benchmark(
    translator: BaseTranslator,
    lines: list[SubtitleLine],
    src_lang: str,
    tgt_lang: str,
    reference_texts: list[str] | None = None,
    max_context_pairs: int = 8,
    keep_translations: bool = True,
    guard: bool = True,
) -> BenchmarkResult:
    """Benchmark one model on one subtitle document.

    Model loading happens before the clock starts: we measure translation
    throughput, not download/initialization time.
    """
    spec = translator.spec
    texts = [prepare_for_translation(line.text) for line in lines]
    source_chars = sum(len(text) for text in texts)

    base = dict(
        model_key=spec.key,
        display_name=spec.display_name,
        tier=spec.tier,
        supports_context=spec.supports_context,
        num_segments=len(texts),
        source_chars=source_chars,
    )

    try:
        translator.load()
        start = time.perf_counter()
        translations = translator.translate_document(
            texts, src_lang, tgt_lang, max_context_pairs=max_context_pairs, guard=guard
        )
        wall = time.perf_counter() - start
    except Exception:
        return BenchmarkResult(
            **base,
            wall_seconds=0.0,
            segments_per_second=0.0,
            chars_per_second=0.0,
            chrf=None,
            bleu=None,
            error=traceback.format_exc(limit=3),
        )

    chrf_score = bleu_score = None
    if reference_texts is not None:
        refs = [prepare_for_translation(text) for text in reference_texts]
        chrf_score = metrics.chrf(translations, refs)
        bleu_score = metrics.bleu(translations, refs, tgt_lang=tgt_lang)

    flagged = issue_counts = None
    if guard:
        per_line = getattr(translator, "last_issues", [])
        flagged = sum(1 for issues in per_line if issues)
        issue_counts = guards.summarize(per_line)

    return BenchmarkResult(
        **base,
        wall_seconds=wall,
        segments_per_second=len(texts) / wall if wall > 0 else 0.0,
        chars_per_second=source_chars / wall if wall > 0 else 0.0,
        chrf=chrf_score,
        bleu=bleu_score,
        translations=translations if keep_translations else None,
        flagged_segments=flagged,
        issue_counts=issue_counts,
    )


def results_to_json(results: list[BenchmarkResult]) -> str:
    return json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2)


def results_to_markdown(results: list[BenchmarkResult]) -> str:
    """Markdown leaderboard, best chrF++ first (errored runs last)."""
    rows = sorted(
        results, key=lambda r: (r.error is not None, -(r.chrf if r.chrf is not None else -1))
    )
    out = [
        "| Model | Tier | Context | Segments | Time (s) | Seg/s | chrF++ | BLEU | Flagged | Status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        fmt = lambda value: f"{value:.2f}" if value is not None else "-"
        status = "ERROR" if r.error else "ok"
        context = "yes" if r.supports_context else "sentence-level"
        flagged = "-" if r.flagged_segments is None else str(r.flagged_segments)
        out.append(
            f"| {r.display_name} | {r.tier} | {context} | {r.num_segments} "
            f"| {fmt(r.wall_seconds)} | {fmt(r.segments_per_second)} "
            f"| {fmt(r.chrf)} | {fmt(r.bleu)} | {flagged} | {status} |"
        )
    return "\n".join(out)
