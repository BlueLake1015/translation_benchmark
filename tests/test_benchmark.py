"""Benchmark runner and metrics tests, English -> Korean."""
import json

from translation_benchmark.benchmark import metrics
from translation_benchmark.benchmark.runner import (
    results_to_json,
    results_to_markdown,
    run_benchmark,
)
from translation_benchmark.models.registry import create_translator
from translation_benchmark.subtitles import prepare_for_translation

from conftest import SRC_LANG, TGT_LANG


def test_chrf_perfect_match_on_korean_reference(ko_lines):
    refs = [prepare_for_translation(line.text) for line in ko_lines]
    assert metrics.chrf(refs, refs) == 100.0


def test_bleu_uses_char_tokenizer_for_korean(ko_lines):
    import pytest

    refs = [prepare_for_translation(line.text) for line in ko_lines]
    assert metrics.bleu(refs, refs, tgt_lang=TGT_LANG) == pytest.approx(100.0)


def test_metrics_reject_misaligned_corpora():
    import pytest

    with pytest.raises(ValueError):
        metrics.chrf(["a"], ["a", "b"])


def test_run_benchmark_with_reference(en_lines, ko_lines):
    translator = create_translator("dummy")
    reference_texts = [line.text for line in ko_lines]
    result = run_benchmark(
        translator, en_lines, SRC_LANG, TGT_LANG, reference_texts=reference_texts
    )

    assert result.error is None
    assert result.model_key == "dummy"
    assert result.num_segments == 18
    assert result.wall_seconds > 0
    assert result.segments_per_second > 0
    assert result.chars_per_second > 0
    # Dummy output shares no Korean characters with the reference: low but valid.
    assert result.chrf is not None and 0 <= result.chrf < 30
    assert result.bleu is not None and 0 <= result.bleu < 30
    assert len(result.translations) == 18


def test_run_benchmark_without_reference(en_lines):
    result = run_benchmark(create_translator("dummy"), en_lines, SRC_LANG, TGT_LANG)
    assert result.error is None
    assert result.chrf is None and result.bleu is None


def test_run_benchmark_captures_errors(en_lines):
    translator = create_translator("dummy")
    translator.translate_batch = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    result = run_benchmark(translator, en_lines, SRC_LANG, TGT_LANG)
    assert result.error is not None and "boom" in result.error


def test_reports(en_lines, ko_lines):
    reference_texts = [line.text for line in ko_lines]
    result = run_benchmark(
        create_translator("dummy"), en_lines, SRC_LANG, TGT_LANG,
        reference_texts=reference_texts,
    )
    payload = json.loads(results_to_json([result]))
    assert payload[0]["model_key"] == "dummy"
    assert payload[0]["num_segments"] == 18

    markdown = results_to_markdown([result])
    assert "| Model |" in markdown
    assert "Dummy (test-only)" in markdown
    assert "ok" in markdown
