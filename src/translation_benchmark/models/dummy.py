"""Deterministic fake translator so tests and benchmark plumbing run anywhere.

It produces a stable pseudo-translation (``[tgt] text``) and records the
context windows it was given, letting tests assert that context-aware
document translation actually threads previous pairs through.
"""
from __future__ import annotations

from translation_benchmark.context import ContextPair
from translation_benchmark.models.base import BaseTranslator, ModelSpec


class DummyTranslator(BaseTranslator):
    def __init__(self, spec: ModelSpec, device: str = "auto", **kwargs) -> None:
        super().__init__(spec, device=device, **kwargs)
        self.seen_contexts: list[list[ContextPair]] = []

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        contexts: list[list[ContextPair]] | None = None,
    ) -> list[str]:
        if contexts is not None:
            self.seen_contexts.extend(contexts)
        return [f"[{tgt_lang}] {text}" for text in texts]
