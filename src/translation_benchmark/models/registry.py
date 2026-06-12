"""The benchmark's model lineup, mirroring the tier table in README.md.

Hugging Face repo ids are defaults and can be overridden per run with
``--hf-id`` (useful for quantized or mirrored checkpoints). Context window
sizes are approximate, taken from each model's base architecture.
"""
from __future__ import annotations

from translation_benchmark.models.base import BaseTranslator, ModelSpec

TIER_LABELS = {
    0: "Test-only",
    1: "Frontier open-weight (server-class hardware)",
    2: "Strong, single high-end GPU (16-24 GB)",
    3: "Lightweight (laptop / small GPU)",
    4: "Draft engines (sentence-level, no context/instructions)",
}

MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in [
        # ----- Tier 1 -------------------------------------------------------
        ModelSpec(
            key="translategemma-27b",
            display_name="TranslateGemma 27B",
            hf_id="google/translategemma-27b-it",
            tier=1,
            tier_label=TIER_LABELS[1],
            params_b=27,
            backend="chat",
            prompt_style="translategemma",
            supports_context=True,
            approx_context_tokens=128_000,
            vram_note="~16 GB at 4-bit",
            notes="3.09 MetricX on WMT24++; best dedicated translation specialist.",
        ),
        # ----- Tier 2 -------------------------------------------------------
        ModelSpec(
            key="qwen3-32b",
            display_name="Qwen3-32B",
            hf_id="Qwen/Qwen3-32B",
            tier=2,
            tier_label=TIER_LABELS[2],
            params_b=32,
            backend="chat",
            prompt_style="qwen",
            supports_context=True,
            approx_context_tokens=32_768,
            vram_note="~20 GB at 4-bit",
            notes="Best general-purpose option at this size; strongest indirect "
            "hallucination evidence in the field.",
        ),
        ModelSpec(
            key="translategemma-12b",
            display_name="TranslateGemma 12B",
            hf_id="google/translategemma-12b-it",
            tier=2,
            tier_label=TIER_LABELS[2],
            params_b=12,
            backend="chat",
            prompt_style="translategemma",
            supports_context=True,
            approx_context_tokens=128_000,
            vram_note="~8 GB at 4-bit",
            notes="Beats a 27B generalist baseline at under half the size.",
        ),
        ModelSpec(
            key="tower-plus-9b",
            display_name="Tower+ 9B",
            hf_id="Unbabel/Tower-Plus-9B",
            tier=2,
            tier_label=TIER_LABELS[2],
            params_b=9,
            backend="chat",
            prompt_style="tower",
            supports_context=True,
            approx_context_tokens=8_192,
            vram_note="~6 GB at 4-bit",
            notes="84.38 XCOMET on its 24 supported pairs; best in its weight class "
            "among open models; Korean explicitly supported.",
        ),
        ModelSpec(
            key="qwen3-14b",
            display_name="Qwen3-14B",
            hf_id="Qwen/Qwen3-14B",
            tier=2,
            tier_label=TIER_LABELS[2],
            params_b=14,
            backend="chat",
            prompt_style="qwen",
            supports_context=True,
            approx_context_tokens=32_768,
            vram_note="~10 GB at 4-bit",
            notes="Slightly below the three above; excellent quality-per-GB.",
        ),
        # ----- Tier 3 -------------------------------------------------------
        ModelSpec(
            key="translategemma-4b",
            display_name="TranslateGemma 4B",
            hf_id="google/translategemma-4b-it",
            tier=3,
            tier_label=TIER_LABELS[3],
            params_b=4,
            backend="chat",
            prompt_style="translategemma",
            supports_context=True,
            approx_context_tokens=128_000,
            vram_note="~3 GB quantized",
            notes="Matches 12B-class generalist quality; runs on laptops.",
        ),
        ModelSpec(
            key="towerinstruct-7b-v0.2",
            display_name="TowerInstruct-7B-v0.2",
            hf_id="Unbabel/TowerInstruct-7B-v0.2",
            tier=3,
            tier_label=TIER_LABELS[3],
            params_b=7,
            backend="chat",
            prompt_style="tower",
            supports_context=True,
            approx_context_tokens=4_096,
            vram_note="~5 GB at 4-bit",
            notes="Previous Tower generation; superseded but still decent for its "
            "10 languages including Korean.",
        ),
        # ----- Tier 4 (sentence-level only) ---------------------------------
        ModelSpec(
            key="madlad400-10b",
            display_name="MADLAD-400 10B",
            hf_id="google/madlad400-10b-mt",
            tier=4,
            tier_label=TIER_LABELS[4],
            params_b=10,
            backend="seq2seq-madlad",
            prompt_style="",
            supports_context=False,
            approx_context_tokens=None,
            vram_note="~7 GB at int8",
            notes="Wide language coverage, literal output; first-pass drafts only. "
            "Sentence-level: cannot use document context or instructions.",
        ),
        ModelSpec(
            key="nllb200-3.3b",
            display_name="NLLB-200 3.3B",
            hf_id="facebook/nllb-200-3.3B",
            tier=4,
            tier_label=TIER_LABELS[4],
            params_b=3.3,
            backend="seq2seq-nllb",
            prompt_style="",
            supports_context=False,
            approx_context_tokens=None,
            vram_note="CPU-viable via CTranslate2 (int8)",
            notes="Cheapest to run; flattest dialogue quality. Sentence-level: "
            "cannot use document context or instructions.",
        ),
        # ----- Test-only ----------------------------------------------------
        ModelSpec(
            key="dummy",
            display_name="Dummy (test-only)",
            hf_id="",
            tier=0,
            tier_label=TIER_LABELS[0],
            params_b=0,
            backend="dummy",
            prompt_style="",
            supports_context=True,
            approx_context_tokens=None,
            vram_note="none",
            notes="Deterministic fake translator for tests and pipeline smoke runs.",
        ),
    ]
}


def get_spec(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        raise ValueError(
            f"Unknown model {key!r}. Available: {', '.join(sorted(MODELS))}"
        ) from None


def list_specs(tier: int | None = None, include_test: bool = False) -> list[ModelSpec]:
    specs = [
        spec
        for spec in MODELS.values()
        if (include_test or spec.tier != 0) and (tier is None or spec.tier == tier)
    ]
    return sorted(specs, key=lambda spec: (spec.tier, -spec.params_b))


def create_translator(key: str, device: str = "auto", **kwargs) -> BaseTranslator:
    """Instantiate the right backend for a model key.

    kwargs are forwarded to the backend (e.g. hf_id=, load_in_4bit=, ct2_dir=).
    Pass use_ct2=True with nllb200-3.3b to use the CTranslate2 backend.
    """
    spec = get_spec(key)
    if spec.backend == "chat":
        from translation_benchmark.models.chat import ChatTranslator

        return ChatTranslator(spec, device=device, **kwargs)
    if spec.backend == "seq2seq-madlad":
        from translation_benchmark.models.seq2seq import MadladTranslator

        return MadladTranslator(spec, device=device, **kwargs)
    if spec.backend == "seq2seq-nllb":
        if kwargs.pop("use_ct2", False):
            from translation_benchmark.models.seq2seq import NLLBCT2Translator

            return NLLBCT2Translator(spec, device=device, **kwargs)
        from translation_benchmark.models.seq2seq import NLLBTranslator

        return NLLBTranslator(spec, device=device, **kwargs)
    if spec.backend == "dummy":
        from translation_benchmark.models.dummy import DummyTranslator

        return DummyTranslator(spec, device=device, **kwargs)
    raise ValueError(f"Unknown backend {spec.backend!r} for model {key!r}")
