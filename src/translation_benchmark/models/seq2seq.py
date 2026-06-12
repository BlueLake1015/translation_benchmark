"""Sentence-level seq2seq backends (Tier 4 draft engines).

MADLAD-400 and NLLB-200 are encoder-decoder MT models: no chat template, no
instructions, and crucially NO document context — every subtitle line is
translated in isolation. They are useful as fast first-pass draft engines
only. NLLB additionally has an optional CTranslate2 backend that is viable
on CPU.
"""
from __future__ import annotations

from translation_benchmark.context import ContextPair
from translation_benchmark.langs import get_language
from translation_benchmark.models.base import BaseTranslator, ModelSpec
from translation_benchmark.models.paths import models_dir_or_default, resolve_model_source


class _HFSeq2SeqTranslator(BaseTranslator):
    """Shared transformers loading/generation for MADLAD and NLLB."""

    def __init__(
        self,
        spec: ModelSpec,
        device: str = "auto",
        hf_id: str | None = None,
        models_dir: str | None = None,
        quant: str | None = None,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(spec, device=device, **kwargs)
        self.hf_override = hf_id
        self.models_dir = models_dir
        self.quant = quant
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _source(self) -> tuple[str, dict]:
        plan = self.spec.resolve_quant(self.quant)
        return resolve_model_source(
            self.hf_override or plan.hf_id, plan.dir_key, self.models_dir
        )

    def _quantization_kwargs(self) -> dict:  # pragma: no cover - requires torch
        plan = self.spec.resolve_quant(self.quant)
        if not plan.runtime:
            return {}
        from transformers import BitsAndBytesConfig

        config = (
            BitsAndBytesConfig(load_in_4bit=True)
            if plan.runtime == "4bit"
            else BitsAndBytesConfig(load_in_8bit=True)
        )
        return {"quantization_config": config}

    def _load(self) -> None:  # pragma: no cover - requires model download
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The inference stack is not installed. Run: pip install 'translation-benchmark[models]'"
            ) from exc
        source, extra = self._source()
        self._tokenizer = AutoTokenizer.from_pretrained(source, **extra)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            source, device_map=self.device, torch_dtype="auto",
            **self._quantization_kwargs(), **extra,
        )
        self._model.eval()

    def unload(self) -> None:  # pragma: no cover
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def _generate(self, prompts: list[str], **generate_kwargs) -> list[str]:  # pragma: no cover
        import torch

        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True).to(
            self._model.device
        )
        with torch.no_grad():
            output = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, **generate_kwargs
            )
        return [
            text.strip()
            for text in self._tokenizer.batch_decode(output, skip_special_tokens=True)
        ]


class MadladTranslator(_HFSeq2SeqTranslator):
    """MADLAD-400: target language is selected with a ``<2xx>`` source prefix."""

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        contexts: list[list[ContextPair]] | None = None,
    ) -> list[str]:  # pragma: no cover - requires model download
        self.load()
        tgt = get_language(tgt_lang).code
        prompts = [f"<2{tgt}> {text}" for text in texts]
        return self._generate(prompts)


class NLLBTranslator(_HFSeq2SeqTranslator):
    """NLLB-200 via transformers, using FLORES-200 language codes."""

    def _load(self) -> None:  # pragma: no cover - requires model download
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The inference stack is not installed. Run: pip install 'translation-benchmark[models]'"
            ) from exc
        # src_lang must be set at tokenizer load time for correct encoding.
        self._src_flores = None
        self._tokenizer_cls = AutoTokenizer
        source, extra = self._source()
        self._tokenizer_source = (source, extra)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            source, device_map=self.device, torch_dtype="auto",
            **self._quantization_kwargs(), **extra,
        )
        self._model.eval()

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        contexts: list[list[ContextPair]] | None = None,
    ) -> list[str]:  # pragma: no cover - requires model download
        self.load()
        src = get_language(src_lang).flores
        tgt = get_language(tgt_lang).flores
        if self._tokenizer is None or self._src_flores != src:
            source, extra = self._tokenizer_source
            self._tokenizer = self._tokenizer_cls.from_pretrained(source, src_lang=src, **extra)
            self._src_flores = src
        bos = self._tokenizer.convert_tokens_to_ids(tgt)
        return self._generate(texts, forced_bos_token_id=bos)


class _CT2Translator(BaseTranslator):
    """Shared CTranslate2 serving — the optimized engine for Tier 4.

    Expects a one-time CTranslate2 conversion of the model under the models
    directory (default location: ``models/<key>-ct2``), e.g.:
        ct2-transformers-converter --model facebook/nllb-200-3.3B \
            --output_dir models/nllb200-3.3b-ct2 --quantization int8
    Pass ``ct2_dir`` to point somewhere else. The original repo's tokenizer
    is still used (downloaded into models/ if not present).
    """

    def __init__(
        self,
        spec: ModelSpec,
        device: str = "auto",
        ct2_dir: str | None = None,
        models_dir: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(spec, device=device, **kwargs)
        self.models_dir = models_dir
        self.ct2_dir = ct2_dir or str(models_dir_or_default(models_dir) / f"{spec.key}-ct2")
        self._translator = None
        self._tokenizer = None

    def _load(self) -> None:  # pragma: no cover - requires converted model
        try:
            import ctranslate2
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "CTranslate2 backend not installed. Run: pip install 'translation-benchmark[ct2]'"
            ) from exc
        from pathlib import Path

        if not Path(self.ct2_dir).is_dir():
            raise RuntimeError(
                f"No CTranslate2 model at {self.ct2_dir}. Convert once with:\n"
                f"  ct2-transformers-converter --model {self.spec.hf_id} "
                f"--output_dir {self.ct2_dir} --quantization int8\n"
                "or run with --engine transformers."
            )
        device = self.device if self.device in ("cpu", "cuda", "auto") else "cpu"
        self._translator = ctranslate2.Translator(self.ct2_dir, device=device)
        source, extra = resolve_model_source(self.spec.hf_id, self.spec.key, self.models_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(source, **extra)

    def unload(self) -> None:  # pragma: no cover
        self._translator = None
        self._tokenizer = None
        self._loaded = False

    def _encode(self, text: str) -> list[str]:  # pragma: no cover
        return self._tokenizer.convert_ids_to_tokens(self._tokenizer.encode(text))

    def _decode(self, tokens: list[str]) -> str:  # pragma: no cover
        return self._tokenizer.decode(
            self._tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True
        ).strip()


class MadladCT2Translator(_CT2Translator):
    """MADLAD-400 (T5-family) via CTranslate2; keeps the ``<2xx>`` prefix."""

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        contexts: list[list[ContextPair]] | None = None,
    ) -> list[str]:  # pragma: no cover - requires converted model
        self.load()
        tgt = get_language(tgt_lang).code
        source_tokens = [self._encode(f"<2{tgt}> {text}") for text in texts]
        results = self._translator.translate_batch(source_tokens)
        return [self._decode(result.hypotheses[0]) for result in results]


class NLLBCT2Translator(_CT2Translator):
    """NLLB-200 via CTranslate2 — int8 on CPU, the cheapest way to run Tier 4."""

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        contexts: list[list[ContextPair]] | None = None,
    ) -> list[str]:  # pragma: no cover - requires converted model
        self.load()
        src = get_language(src_lang).flores
        tgt = get_language(tgt_lang).flores
        self._tokenizer.src_lang = src
        source_tokens = [self._encode(text) for text in texts]
        results = self._translator.translate_batch(
            source_tokens, target_prefix=[[tgt]] * len(texts)
        )
        # Drop the language-code prefix from each hypothesis.
        return [self._decode(result.hypotheses[0][1:]) for result in results]
