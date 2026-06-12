"""End-to-end English->Korean pipeline tests on the real-world subtitle fixture,
using the deterministic dummy backend (no GPU or model download needed)."""
from translation_benchmark.models.chat import build_user_prompt
from translation_benchmark.models.registry import create_translator
from translation_benchmark.context import ContextPair
from translation_benchmark.subtitles import (
    load_srt,
    prepare_for_translation,
    save_srt,
    translated_copy,
)

from conftest import SRC_LANG, TGT_LANG


def test_document_translation_threads_context(en_lines):
    translator = create_translator("dummy")
    texts = [prepare_for_translation(line.text) for line in en_lines]
    out = translator.translate_document(texts, SRC_LANG, TGT_LANG, max_context_pairs=4)

    assert len(out) == len(texts)
    assert out[1] == "[ko] You're late, detective."

    # First line gets no context; later lines get a growing, then capped, window.
    assert translator.seen_contexts[0] == []
    assert len(translator.seen_contexts[1]) == 1
    assert len(translator.seen_contexts[10]) == 4  # capped at max_context_pairs
    # Context holds the model's own previous output, newest last.
    assert translator.seen_contexts[2][-1] == ContextPair(
        source="You're late, detective.", target="[ko] You're late, detective."
    )


def test_context_window_zero_disables_context(en_lines):
    translator = create_translator("dummy")
    texts = [prepare_for_translation(line.text) for line in en_lines]
    translator.translate_document(texts, SRC_LANG, TGT_LANG, max_context_pairs=0)
    # Batched path: no per-line contexts recorded at all.
    assert translator.seen_contexts == []


def test_translated_srt_output(tmp_path, en_lines):
    translator = create_translator("dummy")
    texts = [prepare_for_translation(line.text) for line in en_lines]
    translations = translator.translate_document(texts, SRC_LANG, TGT_LANG)

    out_path = tmp_path / "night_shift.ko.srt"
    save_srt(out_path, translated_copy(en_lines, translations))

    reloaded = load_srt(out_path)
    assert len(reloaded) == len(en_lines)
    assert all(line.text.startswith("[ko] ") for line in reloaded)
    assert [(line.start, line.end) for line in reloaded] == [
        (line.start, line.end) for line in en_lines
    ]


def test_chat_prompt_includes_context_and_languages():
    context = [ContextPair("You're late, detective.", "늦었군, 형사.")]
    prompt = build_user_prompt(
        "translategemma", "Traffic.", "English", "Korean", context
    )
    assert "English: You're late, detective." in prompt
    assert "Korean: 늦었군, 형사." in prompt
    assert "from English to Korean" in prompt
    assert prompt.rstrip().endswith("English: Traffic.")


def test_tower_prompt_uses_tower_format():
    prompt = build_user_prompt("tower", "Get some sleep.", "English", "Korean", None)
    assert "Translate the following text from English into Korean." in prompt
    assert prompt.endswith("English: Get some sleep.\nKorean:")
