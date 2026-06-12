"""Translator interface shared by all backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from translation_benchmark import guards
from translation_benchmark.context import ContextBuilder, ContextPair
from translation_benchmark.langs import get_language

# On-the-fly bitsandbytes quantization of the base weights — no extra download.
RUNTIME_QUANTS = ("4bit", "8bit")
_HF_BACKENDS = ("chat", "seq2seq-madlad", "seq2seq-nllb")


@dataclass(frozen=True)
class QuantPlan:
    """How to load a model for a requested quantization variant."""

    hf_id: str  # repo to load (base or a pre-quantized variant repo)
    dir_key: str  # directory name under models/ ("<key>" or "<key>@<variant>")
    runtime: str | None  # bitsandbytes mode applied at load time, or None


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
    # Officially published pre-quantized repos: (variant name, repo id)
    quant_repos: tuple[tuple[str, str], ...] = ()
    # Whether the downloader fetches the full-precision base weights by
    # default. False where pre-quantized variants cover normal usage
    # (explicit --quant full still downloads them).
    download_full_by_default: bool = True
    # Engines that can serve THIS model, optimized first (= the default).
    # Per-model data, not derived from the backend: engine support differs
    # between models even within a family.
    engines: tuple[str, ...] = ("transformers",)

    def supported_engines(self) -> tuple[str, ...]:
        return self.engines

    def default_engine(self) -> str:
        return self.engines[0]

    def quant_variants(self) -> list[str]:
        """All accepted --quant values for this model."""
        variants = ["none"]
        if self.backend in _HF_BACKENDS:
            variants.extend(RUNTIME_QUANTS)
        variants.extend(name for name, _ in self.quant_repos)
        return variants

    def resolve_quant(self, quant: str | None) -> QuantPlan:
        if not quant or quant in ("none", "full"):
            return QuantPlan(self.hf_id, self.key, None)
        repos = dict(self.quant_repos)
        if quant in repos:
            return QuantPlan(repos[quant], f"{self.key}@{quant}", None)
        if quant in RUNTIME_QUANTS and self.backend in _HF_BACKENDS:
            return QuantPlan(self.hf_id, self.key, quant)
        raise ValueError(
            f"Model {self.key!r} has no quantization variant {quant!r}; "
            f"available: {', '.join(self.quant_variants())}"
        )


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
        guard: bool = True,
    ) -> list[str]:
        """Translate an ordered document (e.g. all cues of a subtitle file).

        Context-aware models go line by line, feeding their own previous
        output back in as context. Sentence-level models translate in
        batches of ``batch_size`` with no context.

        With ``guard`` on (default), every line passes hallucination checks
        (see guards.py): flagged lines are retried once without context,
        and lines that stay flagged are quarantined from the rolling
        context so they cannot poison later lines. Per-line findings are
        left on ``self.last_issues`` for reporting.
        """
        self.load()
        self.last_issues: list[list[guards.Issue]] = [[] for _ in texts]
        tgt_name = get_language(tgt_lang).name

        if not self.supports_context or max_context_pairs == 0:
            out: list[str] = []
            for i in range(0, len(texts), batch_size):
                out.extend(self.translate_batch(texts[i : i + batch_size], src_lang, tgt_lang))
            if guard:
                out = [guards.clean_output(text, tgt_name) for text in out]
                self.last_issues = [
                    guards.find_issues(src, hyp, src_lang, tgt_lang)
                    for src, hyp in zip(texts, out)
                ]
            return out

        builder = ContextBuilder(max_pairs=max_context_pairs, max_chars=max_context_chars)
        out = []
        for idx, text in enumerate(texts):
            context = builder.pairs()
            translation = self.translate_batch(
                [text], src_lang, tgt_lang, contexts=[context]
            )[0]
            issues: list[guards.Issue] = []
            if guard:
                translation = guards.clean_output(translation, tgt_name)
                issues = guards.find_issues(text, translation, src_lang, tgt_lang)
                if issues and context:
                    # Retry without context: a poisoned window is the main
                    # driver of propagated hallucinations, and changing the
                    # prompt is the only lever under greedy decoding.
                    retry = guards.clean_output(
                        self.translate_batch([text], src_lang, tgt_lang, contexts=[[]])[0],
                        tgt_name,
                    )
                    retry_issues = guards.find_issues(text, retry, src_lang, tgt_lang)
                    if len(retry_issues) < len(issues):
                        translation, issues = retry, retry_issues
            out.append(translation)
            self.last_issues[idx] = issues
            if not guard or not issues:  # quarantine flagged lines
                builder.push(text, translation)
        return out
