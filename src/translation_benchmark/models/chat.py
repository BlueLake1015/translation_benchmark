"""Context-aware chat/causal-LM backend (TranslateGemma, Qwen3, Tower).

These models are instruction-tuned, so each subtitle line is translated via
a prompt that includes a rolling window of previous (source, target) pairs.
Heavy dependencies (torch/transformers) are imported lazily so the rest of
the package works without them.
"""
from __future__ import annotations

from translation_benchmark.context import ContextPair
from translation_benchmark.langs import get_language
from translation_benchmark.models.base import BaseTranslator, ModelSpec

SYSTEM_PROMPT = (
    "You are a professional subtitle translator. Translate film dialogue "
    "faithfully and idiomatically, preserving tone, register, and speaker "
    "consistency with the previous lines. Output only the translation, with "
    "no quotes, notes, or explanations."
)


def build_user_prompt(
    style: str,
    text: str,
    src_name: str,
    tgt_name: str,
    context: list[ContextPair] | None,
) -> str:
    """Render the per-line prompt for a given model family."""
    parts: list[str] = []
    if context:
        parts.append(f"Previous subtitle lines ({src_name} -> {tgt_name}):")
        for pair in context:
            parts.append(f"{src_name}: {pair.source}")
            parts.append(f"{tgt_name}: {pair.target}")
        parts.append("")

    if style == "tower":
        # TowerInstruct / Tower+ models were tuned on this exact pattern.
        parts.append(f"Translate the following text from {src_name} into {tgt_name}.")
        parts.append(f"{src_name}: {text}")
        parts.append(f"{tgt_name}:")
    else:
        # TranslateGemma and Qwen3 follow generic instructions well.
        parts.append(
            f"Translate the next subtitle line from {src_name} to {tgt_name}. "
            "Output only the translation."
        )
        parts.append(f"{src_name}: {text}")
    return "\n".join(parts)


class ChatTranslator(BaseTranslator):
    def __init__(
        self,
        spec: ModelSpec,
        device: str = "auto",
        hf_id: str | None = None,
        max_new_tokens: int = 256,
        load_in_4bit: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(spec, device=device, **kwargs)
        self.hf_id = hf_id or spec.hf_id
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:  # pragma: no cover - requires GPU/model download
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The inference stack is not installed. Run: pip install 'translation-benchmark[models]'"
            ) from exc

        kwargs: dict = {"device_map": self.device, "torch_dtype": "auto"}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.hf_id, **kwargs)
        self._model.eval()

    def unload(self) -> None:  # pragma: no cover
        self._model = None
        self._tokenizer = None
        self._loaded = False
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass

    def build_messages(
        self, text: str, src_lang: str, tgt_lang: str, context: list[ContextPair] | None
    ) -> list[dict]:
        src_name = get_language(src_lang).name
        tgt_name = get_language(tgt_lang).name
        user = build_user_prompt(self.spec.prompt_style, text, src_name, tgt_name, context)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        contexts: list[list[ContextPair]] | None = None,
    ) -> list[str]:  # pragma: no cover - requires GPU/model download
        self.load()
        import torch

        results: list[str] = []
        for i, text in enumerate(texts):
            context = contexts[i] if contexts else None
            messages = self.build_messages(text, src_lang, tgt_lang, context)
            template_kwargs: dict = {"add_generation_prompt": True}
            if self.spec.prompt_style == "qwen":
                # Qwen3 hybrid thinking would burn tokens on reasoning traces;
                # subtitle translation wants the direct answer.
                template_kwargs["enable_thinking"] = False
            inputs = self._tokenizer.apply_chat_template(
                messages, tokenize=True, return_tensors="pt", **template_kwargs
            ).to(self._model.device)
            with torch.no_grad():
                output = self._model.generate(
                    inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_tokens = output[0][inputs.shape[-1] :]
            results.append(self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return results
