"""Rolling dialogue context for context-aware models.

Subtitles are a document, not a bag of sentences: pronouns, register
(formal/informal speech), names, and running jokes all depend on what was
said before. Context-capable models are therefore fed a rolling window of
the most recent (source, target) pairs so the next line is translated
consistently with what came before.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextPair:
    source: str
    target: str


class ContextBuilder:
    """Keeps the most recent translated pairs, bounded by count and size.

    ``max_pairs`` bounds how many previous subtitle lines are shown to the
    model; ``max_chars`` bounds the total character budget (most recent pairs
    are kept) so a long scene cannot blow up the prompt.
    """

    def __init__(self, max_pairs: int = 8, max_chars: int = 2400) -> None:
        if max_pairs < 0 or max_chars < 0:
            raise ValueError("max_pairs and max_chars must be non-negative")
        self.max_pairs = max_pairs
        self.max_chars = max_chars
        self._pairs: deque[ContextPair] = deque(maxlen=max_pairs or None)

    def push(self, source: str, target: str) -> None:
        if self.max_pairs == 0:
            return
        self._pairs.append(ContextPair(source=source, target=target))

    def pairs(self) -> list[ContextPair]:
        """Most recent pairs, newest last, trimmed to the character budget."""
        kept: list[ContextPair] = []
        budget = self.max_chars
        for pair in reversed(self._pairs):
            cost = len(pair.source) + len(pair.target)
            if kept and cost > budget:
                break
            kept.append(pair)
            budget -= cost
        kept.reverse()
        return kept

    def render(self, src_name: str, tgt_name: str) -> str:
        """Human-readable context block for inclusion in a chat prompt."""
        pairs = self.pairs()
        if not pairs:
            return ""
        lines = [f"Previous subtitle lines ({src_name} -> {tgt_name}):"]
        for pair in pairs:
            lines.append(f"{src_name}: {pair.source}")
            lines.append(f"{tgt_name}: {pair.target}")
        return "\n".join(lines)
