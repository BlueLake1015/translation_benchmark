"""End-to-end English->Korean pipeline tests on the real-world subtitle fixture,
using the deterministic dummy backend (no GPU or model download needed)."""
import pytest

from translation_benchmark.models.chat import ChatTranslator, build_messages
from translation_benchmark.models.registry import create_translator, get_spec
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
    # guard=False: this test checks raw context threading; the dummy's fake
    # "[ko] ..." output would otherwise be quarantined as wrong-language.
    out = translator.translate_document(
        texts, SRC_LANG, TGT_LANG, max_context_pairs=4, guard=False
    )

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


def test_translategemma_prompt_is_minimal_specialist_format():
    # Specialist: no system prompt, canonical translation format, context block.
    context = [ContextPair("You're late, detective.", "늦었군, 형사.")]
    messages = build_messages(get_spec("translategemma-4b"), "Traffic.", "en", "ko", context)
    assert [m["role"] for m in messages] == ["user"]
    content = messages[0]["content"]
    assert "English: You're late, detective." in content
    assert "Korean: 늦었군, 형사." in content
    assert content.endswith("Translate from English to Korean:\nTraffic.")


def test_tower_prompt_uses_tuned_format():
    # Tower was tuned on this exact pattern; no system prompt.
    messages = build_messages(get_spec("tower-plus-9b"), "Get some sleep.", "en", "ko", None)
    assert [m["role"] for m in messages] == ["user"]
    assert "Translate the following text from English into Korean." in messages[0]["content"]
    assert messages[0]["content"].endswith("English: Get some sleep.\nKorean:")


def test_qwen_prompt_keeps_detailed_system_instructions():
    # Generalist: instruction-heavy prompting pays off here.
    messages = build_messages(get_spec("qwen3-14b"), "Traffic.", "en", "ko", None)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "subtitle translator" in messages[0]["content"]


def test_prompt_style_override_for_ab_testing():
    spec = get_spec("translategemma-4b")
    default = build_messages(spec, "Traffic.", "en", "ko", None)
    overridden = build_messages(spec, "Traffic.", "en", "ko", None, prompt_style="qwen")
    assert [m["role"] for m in default] == ["user"]
    assert [m["role"] for m in overridden] == ["system", "user"]
    # The override threads through the translator construction too.
    translator = ChatTranslator(spec, prompt_style="tower")
    messages = translator.build_messages("Traffic.", "en", "ko", None)
    assert messages[0]["content"].endswith("English: Traffic.\nKorean:")


def test_unknown_prompt_style_rejected():
    with pytest.raises(ValueError, match="Unknown prompt style"):
        build_messages(get_spec("qwen3-14b"), "Hi.", "en", "ko", None, prompt_style="llama")
