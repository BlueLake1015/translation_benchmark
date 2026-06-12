"""Context-aware chat/causal-LM backend (TranslateGemma, Qwen3, Tower).

Each model family gets a DEDICATED prompt matching its training
distribution — off-format prompts measurably degrade specialists and are
themselves a hallucination trigger:

- ``translategemma``: minimal canonical translation format, no system
  prompt. A translation specialist drifts when wrapped in
  meta-instructions, and Gemma chat templates fold system text into the
  user turn anyway.
- ``tower``: the exact pattern Tower/TowerInstruct were tuned on, no
  system prompt — these models were not trained to follow meta-instructions.
- ``qwen``: detailed system prompt + instruction. Generalists are where
  instruction-heavy prompting pays off.

The model's tuned style comes from its registry spec (``prompt_style``);
``--prompt-style`` overrides it per run for A/B testing prompt changes
against a reference. Hallucination guards stay model-agnostic, so flag
counts remain comparable across prompt styles.

Heavy dependencies (torch/transformers) are imported lazily so the rest of
the package works without them.
"""
from __future__ import annotations

from translation_benchmark import guards
from translation_benchmark.context import ContextPair
from translation_benchmark.langs import get_language
from translation_benchmark.models.base import BaseTranslator, ModelSpec
from translation_benchmark.models.paths import resolve_model_source

SYSTEM_PROMPT = (
    "You are a professional subtitle translator. Translate film dialogue "
    "faithfully and idiomatically, preserving tone, register, and speaker "
    "consistency with the previous lines. Translate exactly what is said: "
    "never add, omit, or explain content, and never repeat words that are "
    "not repeated in the source. If a line is just a name, number, or "
    "interjection, translate it directly. Output only the translation, with "
    "no quotes, notes, or explanations."
)


def _context_block(
    src_name: str, tgt_name: str, context: list[ContextPair]
) -> list[str]:
    lines = [f"Previous subtitle lines ({src_name} -> {tgt_name}):"]
    for pair in context:
        lines.append(f"{src_name}: {pair.source}")
        lines.append(f"{tgt_name}: {pair.target}")
    lines.append("")
    return lines


def _messages_translategemma(text, src_name, tgt_name, context) -> list[dict]:
    parts: list[str] = []
    if context:
        parts.extend(_context_block(src_name, tgt_name, context))
    parts.append(f"Translate from {src_name} to {tgt_name}:")
    parts.append(text)
    return [{"role": "user", "content": "\n".join(parts)}]


def _messages_tower(text, src_name, tgt_name, context) -> list[dict]:
    parts: list[str] = []
    if context:
        parts.extend(_context_block(src_name, tgt_name, context))
    parts.append(f"Translate the following text from {src_name} into {tgt_name}.")
    parts.append(f"{src_name}: {text}")
    parts.append(f"{tgt_name}:")
    return [{"role": "user", "content": "\n".join(parts)}]


def _messages_qwen(text, src_name, tgt_name, context) -> list[dict]:
    parts: list[str] = []
    if context:
        parts.extend(_context_block(src_name, tgt_name, context))
    parts.append(
        f"Translate the next subtitle line from {src_name} to {tgt_name}. "
        "Output only the translation."
    )
    parts.append(f"{src_name}: {text}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]


PROMPT_STYLES = {
    "translategemma": _messages_translategemma,
    "tower": _messages_tower,
    "qwen": _messages_qwen,
}


def build_messages(
    spec: ModelSpec,
    text: str,
    src_lang: str,
    tgt_lang: str,
    context: list[ContextPair] | None,
    prompt_style: str | None = None,
) -> list[dict]:
    """Chat messages for one subtitle line — shared by all chat engines.

    Uses the model's tuned style from the registry unless overridden.
    """
    style = prompt_style or spec.prompt_style or "qwen"
    try:
        builder = PROMPT_STYLES[style]
    except KeyError:
        raise ValueError(
            f"Unknown prompt style {style!r}; available: {', '.join(sorted(PROMPT_STYLES))}"
        ) from None
    src_name = get_language(src_lang).name
    tgt_name = get_language(tgt_lang).name
    return builder(text, src_name, tgt_name, context)


class ChatTranslator(BaseTranslator):
    def __init__(
        self,
        spec: ModelSpec,
        device: str = "auto",
        hf_id: str | None = None,
        models_dir: str | None = None,
        quant: str | None = None,
        prompt_style: str | None = None,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(spec, device=device, **kwargs)
        self.hf_override = hf_id
        self.models_dir = models_dir
        self.quant = quant
        self.prompt_style = prompt_style
        self.max_new_tokens = max_new_tokens
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

        plan = self.spec.resolve_quant(self.quant)
        kwargs: dict = {"device_map": self.device, "torch_dtype": "auto"}
        if plan.runtime:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = (
                BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                if plan.runtime == "4bit"
                else BitsAndBytesConfig(load_in_8bit=True)
            )
        # An explicit --hf-id wins over the variant repo from the registry.
        source, extra = resolve_model_source(
            self.hf_override or plan.hf_id, plan.dir_key, self.models_dir
        )
        self._tokenizer = AutoTokenizer.from_pretrained(source, **extra)
        self._model = AutoModelForCausalLM.from_pretrained(source, **kwargs, **extra)
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
        return build_messages(
            self.spec, text, src_lang, tgt_lang, context, prompt_style=self.prompt_style
        )

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
                    # Cap relative to the source line so repetition loops are
                    # cut early instead of running to the global limit.
                    max_new_tokens=guards.max_new_tokens_for(text, self.max_new_tokens),
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_tokens = output[0][inputs.shape[-1] :]
            results.append(self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return results
