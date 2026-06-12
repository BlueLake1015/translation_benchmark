"""Translator interface shared by all backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from translation_benchmark.context import ContextBuilder, ContextPair


@dataclass(frozen=True)
class ModelSpec:
    key: str  # CLI identifier, e.g. "translategemma-27b"
    display_name: str
    hf_id: str  # default Hugging Face repo; overridable with --hf-id
    tier: int  # 1-4 per the project spec; 0 = test-only dummy
    tier_label: str
    params_b: float  # parameter count, billions
    backend: str  # "chat" | "seq2seq-madlad" | "seq2seq-nllb" | "dummy"
    prompt_style: str  # "translategemma" | "qwen" | "tower" | "" for seq2seq
    supports_context: bool
    approx_context_tokens: int | None  # approximate context window; None = sentence-level
    vram_note: str
    notes: str


class BaseTranslator(ABC):
    """A loaded (or lazily loadable) translation model.

    Context-aware backends translate one line at a time with a rolling
    window of previous (source, target) pairs. Sentence-level backends
    (Tier 4) ignore context entirely and may batch lines for throughput.
    """

    def __init__(self, spec: ModelSpec, device: str = "auto", **kwargs) -> None:
        self.spec = spec
        self.device = device
        self._loaded = False

    @property
    def supports_context(self) -> bool:
        return self.spec.supports_context

    def load(self) -> None:
        """Load weights. Idempotent; called automatically on first translate."""
        if not self._loaded:
            self._load()
            self._loaded = True

    def _load(self) -> None:  # pragma: no cover - overridden by heavy backends
        pass

    def unload(self) -> None:  # pragma: no cover - overridden by heavy backends
        self._loaded = False

    @abstractmethod
    def translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        contexts: list[list[ContextPair]] | None = None,
    ) -> list[str]:
        """Translate a batch of independent texts.

        ``contexts``, when given, holds one context window per text; backends
        that do not support context ignore it.
        """

    def translate_document(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        max_context_pairs: int = 8,
        max_context_chars: int = 2400,
        batch_size: int = 8,
    ) -> list[str]:
        """Translate an ordered document (e.g. all cues of a subtitle file).

        Context-aware models go line by line, feeding their own previous
        output back in as context. Sentence-level models translate in
        batches of ``batch_size`` with no context.
        """
        self.load()
        if not self.supports_context or max_context_pairs == 0:
            out: list[str] = []
            for i in range(0, len(texts), batch_size):
                out.extend(self.translate_batch(texts[i : i + batch_size], src_lang, tgt_lang))
            return out

        builder = ContextBuilder(max_pairs=max_context_pairs, max_chars=max_context_chars)
        out = []
        for text in texts:
            translation = self.translate_batch(
                [text], src_lang, tgt_lang, contexts=[builder.pairs()]
            )[0]
            out.append(translation)
            builder.push(text, translation)
        return out
