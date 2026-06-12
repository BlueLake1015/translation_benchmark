"""Hallucination guards: detection units and mitigation behavior (EN -> KO)."""
import pytest

from translation_benchmark import guards
from translation_benchmark.benchmark.runner import run_benchmark
from translation_benchmark.models.dummy import DummyTranslator
from translation_benchmark.models.registry import create_translator, get_spec

from conftest import SRC_LANG, TGT_LANG

GOOD_KO = "늦었군, 형사."


# --- detection units ---------------------------------------------------------


def test_clean_output_strips_labels_quotes_and_commentary():
    assert guards.clean_output("Korean: 늦었군, 형사.", "Korean") == GOOD_KO
    assert guards.clean_output('"늦었군, 형사."', "Korean") == GOOD_KO
    assert (
        guards.clean_output("늦었군, 형사.\n\n(Note: informal register)", "Korean") == GOOD_KO
    )
    assert guards.clean_output("  늦었군, 형사.  ", "Korean") == GOOD_KO


def test_good_korean_line_passes_all_checks():
    issues = guards.find_issues("You're late, detective.", GOOD_KO, SRC_LANG, TGT_LANG)
    assert issues == []


def test_detects_empty_and_source_copy():
    assert [i.code for i in guards.find_issues("Hello there.", "  ", "en", "ko")] == ["empty"]
    issues = guards.find_issues(
        "You're late, detective.", "You're late, detective.", "en", "ko"
    )
    assert "source_copy" in [i.code for i in issues]


def test_detects_repetition_loops():
    assert guards.has_repetition_loop("같은 같은 같은 같은 같은 다리")
    assert guards.has_repetition_loop("the bridge the bridge the bridge the bridge")
    assert guards.has_repetition_loop("ㅋ" * 15)  # character run, no whitespace
    # Legitimate repetition stays clean.
    assert not guards.has_repetition_loop("같은 다리, 같은 구경, 주머니엔 같은 쪽지까지.")


def test_detects_length_explosion_and_truncation():
    src = "Get some sleep. Tomorrow we visit the archive."
    codes = [i.code for i in guards.find_issues(src, "네" * 200, SRC_LANG, TGT_LANG)]
    assert "too_long" in codes
    codes = [i.code for i in guards.find_issues(src, "네.", SRC_LANG, TGT_LANG)]
    assert "too_short" in codes


def test_detects_wrong_output_language():
    issues = guards.find_issues(
        "You're late, detective.", "Du bist spät dran, Detective.", SRC_LANG, TGT_LANG
    )
    assert "wrong_language" in [i.code for i in issues]
    # Mixed lines with proper nouns kept in Latin are fine.
    assert (
        guards.find_issues("Reyes is late.", "레예스 형사가 늦었어요, Reyes요.", "en", "ko") == []
    )


def test_detects_meta_text():
    issues = guards.find_issues(
        "You're late, detective.",
        "Sure, here is the Korean translation: 늦었군.",
        SRC_LANG,
        TGT_LANG,
    )
    assert "meta_text" in [i.code for i in issues]


def test_max_new_tokens_scales_with_source():
    assert guards.max_new_tokens_for("Hi.", 256) == 64  # floor
    assert guards.max_new_tokens_for("x" * 60, 256) == 136
    assert guards.max_new_tokens_for("x" * 500, 256) == 256  # configured cap


# --- mitigation behavior -----------------------------------------------------


class ScriptedTranslator(DummyTranslator):
    """Returns queued outputs in order; records contexts like the dummy."""

    def __init__(self, outputs):
        super().__init__(get_spec("dummy"))
        self.outputs = list(outputs)

    def translate_batch(self, texts, src_lang, tgt_lang, contexts=None):
        if contexts is not None:
            self.seen_contexts.extend(contexts)
        return [self.outputs.pop(0) for _ in texts]


def test_flagged_line_retried_without_context_and_better_retry_kept():
    translator = ScriptedTranslator(
        [
            GOOD_KO,  # line 1: clean -> enters context
            "다리 다리 다리 다리 다리",  # line 2 with context: repetition loop
            "차가 막혔습니다.",  # line 2 retry without context: clean
        ]
    )
    out = translator.translate_document(["You're late.", "Traffic."], "en", "ko")
    assert out == [GOOD_KO, "차가 막혔습니다."]
    assert translator.last_issues == [[], []]
    # The retry call must have carried an empty context window.
    assert translator.seen_contexts[-1] == []


def test_unrecovered_line_is_quarantined_from_context():
    translator = ScriptedTranslator(
        [
            GOOD_KO,  # line 1: clean
            "bridge bridge bridge bridge",  # line 2: bad (with context)
            "bridge bridge bridge bridge",  # line 2 retry: still bad
            "지난달이랑 같은 다리인가?",  # line 3: clean
        ]
    )
    out = translator.translate_document(
        ["You're late.", "The bridge.", "Same bridge as last month?"], "en", "ko"
    )
    assert len(out) == 3
    assert [i.code for i in translator.last_issues[1]] != []
    # Line 3's context contains only line 1 — the flagged line never entered.
    final_context = translator.seen_contexts[-1]
    assert [pair.target for pair in final_context] == [GOOD_KO]


def test_guard_disabled_keeps_raw_output_and_full_context():
    translator = ScriptedTranslator([GOOD_KO, "bridge bridge bridge bridge"])
    out = translator.translate_document(["A.", "B."], "en", "ko", guard=False)
    assert out[1] == "bridge bridge bridge bridge"
    assert translator.last_issues == [[], []]


def test_benchmark_reports_guard_findings(en_lines):
    # The dummy's "[ko] ..." output is wrong-language for Korean: every line flags.
    result = run_benchmark(create_translator("dummy"), en_lines, SRC_LANG, TGT_LANG)
    assert result.flagged_segments == len(en_lines)
    assert result.issue_counts.get("wrong_language") == len(en_lines)


def test_benchmark_guard_off_reports_nothing(en_lines):
    result = run_benchmark(
        create_translator("dummy"), en_lines, SRC_LANG, TGT_LANG, guard=False
    )
    assert result.flagged_segments is None
    assert result.issue_counts is None
