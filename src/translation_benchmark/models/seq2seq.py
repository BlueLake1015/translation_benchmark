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


class _HFSeq2SeqTranslator(BaseTranslator):
    """Shared transformers loading/generation for MADLAD and NLLB."""

    def __init__(
        self,
        spec: ModelSpec,
        device: str = "auto",
        hf_id: str | None = None,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(spec, device=device, **kwargs)
        self.hf_id = hf_id or spec.hf_id
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:  # pragma: no cover - requires model download
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The inference stack is not installed. Run: pip install 'translation-benchmark[models]'"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.hf_id, device_map=self.device, torch_dtype="auto"
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
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.hf_id, device_map=self.device, torch_dtype="auto"
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
            self._tokenizer = self._tokenizer_cls.from_pretrained(self.hf_id, src_lang=src)
            self._src_flores = src
        bos = self._tokenizer.convert_tokens_to_ids(tgt)
        return self._generate(texts, forced_bos_token_id=bos)


class NLLBCT2Translator(BaseTranslator):
    """NLLB-200 via CTranslate2 — int8 on CPU, the cheapest way to run Tier 4.

    Expects a CTranslate2 conversion of the model, e.g.:
        ct2-transformers-converter --model facebook/nllb-200-3.3B \
            --output_dir nllb-200-3.3B-ct2 --quantization int8
    Pass the converted directory as ``ct2_dir``.
    """

    def __init__(
        self, spec: ModelSpec, device: str = "auto", ct2_dir: str | None = None, **kwargs
    ) -> None:
        super().__init__(spec, device=device, **kwargs)
        if not ct2_dir:
            raise ValueError("NLLBCT2Translator requires ct2_dir (converted model directory)")
        self.ct2_dir = ct2_dir
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
        device = "cpu" if self.device in ("auto", "cpu") else self.device
        self._translator = ctranslate2.Translator(self.ct2_dir, device=device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.hf_id)

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
        source_tokens = [
            self._tokenizer.convert_ids_to_tokens(self._tokenizer.encode(text))
            for text in texts
        ]
        results = self._translator.translate_batch(
            source_tokens, target_prefix=[[tgt]] * len(texts)
        )
        out = []
        for result in results:
            tokens = result.hypotheses[0][1:]  # drop the language-code prefix
            out.append(
                self._tokenizer.decode(
                    self._tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True
                ).strip()
            )
        return out
