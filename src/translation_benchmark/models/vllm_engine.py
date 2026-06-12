"""vLLM engine for the chat models (TranslateGemma, Qwen3, Tower).

Same prompts and context handling as the transformers ChatTranslator, but
served by vLLM: paged attention, continuous batching, and native AWQ/FP8
kernels. Context-aware document translation is inherently sequential (each
line's context includes the previous translation), so the win there is
faster prefill/decode per request; with --context-window 0 whole batches
are translated concurrently.

Not available for the Tier 4 seq2seq models — vLLM does not serve these
encoder-decoder architectures (NLLB's CPU path is CTranslate2 instead).
"""
from __future__ import annotations

from translation_benchmark import guards
from translation_benchmark.context import ContextPair
from translation_benchmark.models.base import BaseTranslator, ModelSpec
from translation_benchmark.models.chat import build_messages
from translation_benchmark.models.paths import resolve_model_source


class VLLMChatTranslator(BaseTranslator):
    def __init__(
        self,
        spec: ModelSpec,
        device: str = "auto",
        hf_id: str | None = None,
        models_dir: str | None = None,
        quant: str | None = None,
        prompt_style: str | None = None,
        max_new_tokens: int = 256,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(spec, device=device, **kwargs)
        # Validate the variant up front (fails fast, no vLLM import needed).
        plan = spec.resolve_quant(quant)
        if plan.runtime == "8bit":
            raise ValueError(
                "The vLLM engine does not support 8bit (bitsandbytes int8); "
                "use --quant 4bit, awq, or fp8."
            )
        self.hf_override = hf_id
        self.models_dir = models_dir
        self.quant = quant
        self.prompt_style = prompt_style
        self.max_new_tokens = max_new_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self._llm = None

    def _load(self) -> None:  # pragma: no cover - requires GPU + vllm
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is not installed. Run: pip install 'translation-benchmark[vllm]'"
            ) from exc

        plan = self.spec.resolve_quant(self.quant)
        source, extra = resolve_model_source(
            self.hf_override or plan.hf_id, plan.dir_key, self.models_dir
        )
        llm_kwargs: dict = {
            "model": source,
            "dtype": "auto",
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }
        if extra.get("cache_dir"):
            llm_kwargs["download_dir"] = extra["cache_dir"]
        if plan.runtime == "4bit":
            llm_kwargs["quantization"] = "bitsandbytes"
        # AWQ/FP8 variant repos are autodetected from the checkpoint config.
        if self.max_model_len:
            llm_kwargs["max_model_len"] = self.max_model_len
        self._llm = LLM(**llm_kwargs)

    def unload(self) -> None:  # pragma: no cover - requires GPU + vllm
        self._llm = None
        self._loaded = False
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        contexts: list[list[ContextPair]] | None = None,
    ) -> list[str]:  # pragma: no cover - requires GPU + vllm
        self.load()
        from vllm import SamplingParams

        conversations = [
            build_messages(
                self.spec,
                text,
                src_lang,
                tgt_lang,
                contexts[i] if contexts else None,
                prompt_style=self.prompt_style,
            )
            for i, text in enumerate(texts)
        ]
        chat_kwargs: dict = {}
        if self.spec.prompt_style == "qwen":
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
        # Per-line caps relative to source length cut repetition loops early.
        params = [
            SamplingParams(
                temperature=0.0,
                max_tokens=guards.max_new_tokens_for(text, self.max_new_tokens),
            )
            for text in texts
        ]
        outputs = self._llm.chat(conversations, params, **chat_kwargs)
        return [output.outputs[0].text.strip() for output in outputs]
