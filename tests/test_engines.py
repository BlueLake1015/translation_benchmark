"""Engine selection and defaults — no GPU, vllm, or ctranslate2 install needed."""
import pytest

from translation_benchmark.models.chat import ChatTranslator, build_messages
from translation_benchmark.models.registry import create_translator, get_spec, list_specs
from translation_benchmark.models.seq2seq import (
    MadladCT2Translator,
    MadladTranslator,
    NLLBCT2Translator,
    NLLBTranslator,
)
from translation_benchmark.models.vllm_engine import VLLMChatTranslator


def test_optimized_engines_are_the_defaults():
    # Chat models default to vLLM, Tier 4 to CTranslate2 — construction must
    # not import the heavy stacks; only load() does.
    assert isinstance(create_translator("tower-plus-9b"), VLLMChatTranslator)
    assert isinstance(create_translator("qwen3-32b"), VLLMChatTranslator)
    assert isinstance(create_translator("translategemma-27b"), VLLMChatTranslator)
    assert isinstance(create_translator("madlad400-10b"), MadladCT2Translator)
    assert isinstance(create_translator("nllb200-3.3b"), NLLBCT2Translator)


def test_translategemma_serves_on_vllm_via_completions_path():
    # vLLM's chat endpoint strips TranslateGemma's required custom content
    # fields, so its requests go through client-side template rendering +
    # /v1/completions; the other families use the server-side chat endpoint.
    for key in ("translategemma-27b", "translategemma-12b", "translategemma-4b"):
        assert get_spec(key).engines == ("vllm", "transformers")
    assert create_translator("translategemma-4b")._style() == "translategemma"
    assert create_translator("tower-plus-9b")._style() == "tower"
    # The override flag switches the request path too.
    assert create_translator("tower-plus-9b", prompt_style="translategemma")._style() == (
        "translategemma"
    )


def test_transformers_fallback_is_selectable():
    assert isinstance(
        create_translator("tower-plus-9b", engine="transformers"), ChatTranslator
    )
    assert isinstance(
        create_translator("madlad400-10b", engine="transformers"), MadladTranslator
    )
    assert isinstance(
        create_translator("nllb200-3.3b", engine="transformers"), NLLBTranslator
    )


def test_spec_declares_supported_engines():
    # Engine support is per-model registry data, not derived from the backend.
    assert get_spec("qwen3-32b").engines == ("vllm", "transformers")
    assert get_spec("madlad400-10b").engines == ("ct2", "transformers")
    assert get_spec("dummy").engines == ("transformers",)
    for spec in list_specs():
        assert spec.default_engine() == spec.supported_engines()[0]


def test_every_declared_engine_is_dispatchable():
    from translation_benchmark.models.registry import ENGINES, list_specs as all_specs

    for spec in all_specs(include_test=True):
        assert spec.engines, f"{spec.key} declares no engines"
        for engine in spec.engines:
            assert engine in ENGINES, f"{spec.key} declares unknown engine {engine!r}"


def test_ct2_dir_defaults_under_models_dir():
    translator = create_translator("madlad400-10b", models_dir="/data/weights")
    assert translator.ct2_dir == "/data/weights/madlad400-10b-ct2"


def test_use_ct2_alias_still_works():
    translator = create_translator("nllb200-3.3b", use_ct2=True)
    assert isinstance(translator, NLLBCT2Translator)


def test_vllm_engine_accepts_quant_variants():
    translator = create_translator("qwen3-32b", quant="awq")
    assert isinstance(translator, VLLMChatTranslator)
    assert translator.quant == "awq"


def test_vllm_engine_rejects_8bit():
    with pytest.raises(ValueError, match="does not support 8bit"):
        create_translator("qwen3-32b", quant="8bit")


def test_engine_model_mismatches_rejected():
    with pytest.raises(ValueError, match="cannot serve"):
        create_translator("madlad400-10b", engine="vllm")
    with pytest.raises(ValueError, match="cannot serve"):
        create_translator("tower-plus-9b", engine="ct2")
    with pytest.raises(ValueError, match="cannot serve"):
        create_translator("dummy", engine="vllm")


def test_unknown_engine_rejected():
    with pytest.raises(ValueError, match="Unknown engine"):
        create_translator("tower-plus-9b", engine="tgi")


def test_engines_share_prompt_building():
    # Both chat engines must produce identical messages for the same input.
    spec = get_spec("tower-plus-9b")
    shared = build_messages(spec, "You're late, detective.", "en", "ko", None)
    via_chat = ChatTranslator(spec).build_messages("You're late, detective.", "en", "ko", None)
    assert shared == via_chat
    assert "English: You're late, detective." in shared[-1]["content"]
