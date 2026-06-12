"""Quantization variant resolution on ModelSpec."""
import pytest

from translation_benchmark.models.registry import get_spec


def test_default_is_base_weights():
    spec = get_spec("tower-plus-9b")
    for quant in (None, "", "none", "full"):
        plan = spec.resolve_quant(quant)
        assert plan.hf_id == spec.hf_id
        assert plan.dir_key == spec.key
        assert plan.runtime is None


def test_runtime_quants_reuse_base_repo():
    spec = get_spec("translategemma-27b")
    for quant in ("4bit", "8bit"):
        plan = spec.resolve_quant(quant)
        assert plan.hf_id == spec.hf_id
        assert plan.dir_key == spec.key
        assert plan.runtime == quant


def test_repo_variant_points_at_quantized_repo():
    spec = get_spec("qwen3-32b")
    plan = spec.resolve_quant("awq")
    assert plan.hf_id == "Qwen/Qwen3-32B-AWQ"
    assert plan.dir_key == "qwen3-32b@awq"
    assert plan.runtime is None


def test_quant_variants_listing():
    assert get_spec("qwen3-14b").quant_variants() == ["none", "4bit", "8bit", "awq", "fp8"]
    assert get_spec("madlad400-10b").quant_variants() == ["none", "4bit", "8bit"]
    assert get_spec("dummy").quant_variants() == ["none"]


def test_unknown_variant_raises_with_available_list():
    with pytest.raises(ValueError, match="available: none, 4bit, 8bit"):
        get_spec("tower-plus-9b").resolve_quant("awq")


def test_seq2seq_supports_runtime_quant():
    plan = get_spec("nllb200-3.3b").resolve_quant("8bit")
    assert plan.runtime == "8bit"
