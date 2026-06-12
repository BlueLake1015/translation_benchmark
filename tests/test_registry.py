import pytest

from translation_benchmark.models.dummy import DummyTranslator
from translation_benchmark.models.registry import create_translator, get_spec, list_specs

EXPECTED_LINEUP = {
    1: {"translategemma-27b"},
    2: {"qwen3-32b", "translategemma-12b", "tower-plus-9b", "qwen3-14b"},
    3: {"translategemma-4b", "towerinstruct-7b-v0.2"},
    4: {"madlad400-10b", "nllb200-3.3b"},
}


def test_full_lineup_present():
    for tier, keys in EXPECTED_LINEUP.items():
        assert {spec.key for spec in list_specs(tier=tier)} == keys
    assert len(list_specs()) == 9


def test_tier4_is_sentence_level_only():
    for spec in list_specs(tier=4):
        assert spec.supports_context is False
        assert spec.approx_context_tokens is None
        assert "sentence-level" in spec.notes.lower()


def test_tiers_1_to_3_support_context():
    for tier in (1, 2, 3):
        for spec in list_specs(tier=tier):
            assert spec.supports_context is True
            assert spec.approx_context_tokens


def test_dummy_excluded_from_default_listing():
    assert all(spec.key != "dummy" for spec in list_specs())
    assert any(spec.key == "dummy" for spec in list_specs(include_test=True))


def test_get_spec_unknown_key():
    with pytest.raises(ValueError, match="Unknown model"):
        get_spec("gpt-7")


def test_create_dummy_translator():
    translator = create_translator("dummy")
    assert isinstance(translator, DummyTranslator)
    assert translator.supports_context


def test_real_backends_instantiate_without_heavy_deps():
    # Construction must not import torch/transformers; only load() does.
    for key in ("translategemma-27b", "qwen3-32b", "tower-plus-9b", "madlad400-10b",
                "nllb200-3.3b"):
        translator = create_translator(key)
        assert translator.spec.key == key
