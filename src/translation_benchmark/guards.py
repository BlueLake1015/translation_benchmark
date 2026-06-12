"""Hallucination detection and mitigation for MT outputs.

LLM translators fail in characteristic ways: runaway repetition loops,
meta text ("Here is the translation:"), copying the source through,
answering in the wrong language, inventing content (length explosion), or
dropping it (severe truncation). Mitigation strategy, applied per line in
``BaseTranslator.translate_document``:

1. Clean obvious wrapping (echoed "Korean:" labels, quotes, commentary
   after the first line).
2. Run deterministic checks; each failure is an ``Issue`` with a code.
3. Context-aware engines retry a flagged line once WITHOUT context —
   poisoned context is the main driver of propagated hallucinations and
   changing the prompt is the only lever for greedy decoding.
4. Lines still flagged never enter the rolling context window
   (quarantine), and are reported per run (``flagged`` in benchmarks).

Additionally, chat engines cap generation length relative to the source
line, which stops repetition loops from running to the token limit.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

# Unicode ranges of the dominant script for script-bound target languages.
SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "ko": ((0xAC00, 0xD7A3), (0x1100, 0x11FF), (0x3130, 0x318F)),
    "ja": ((0x3040, 0x309F), (0x30A0, 0x30FF), (0x4E00, 0x9FFF)),
    "zh": ((0x4E00, 0x9FFF), (0x3400, 0x4DBF)),
    "ru": ((0x0400, 0x04FF),),
    "uk": ((0x0400, 0x04FF),),
    "ar": ((0x0600, 0x06FF), (0x0750, 0x077F)),
    "hi": ((0x0900, 0x097F),),
    "th": ((0x0E00, 0x0E7F),),
}

META_MARKERS = (
    "sure,",
    "sure!",
    "here is",
    "here's",
    "the translation",
    "translation:",
    "i cannot",
    "i can't",
    "i'm sorry",
    "as an ai",
    "note:",
)


@dataclass(frozen=True)
class Issue:
    code: str  # "empty" | "source_copy" | "repetition" | "too_long" | "too_short" | "wrong_language" | "meta_text"
    detail: str = ""


def clean_output(text: str, tgt_name: str) -> str:
    """Strip common non-translation wrapping without touching the content.

    Sources are flattened to a single line before translation, so anything
    after the first output line is commentary or an unrequested alternative.
    """
    out = text.strip()
    for line in out.splitlines():
        if line.strip():
            out = line.strip()
            break
    for label in (f"{tgt_name}:", "Translation:"):
        if out.lower().startswith(label.lower()):
            out = out[len(label) :].strip()
    if len(out) >= 2 and out[0] in "\"'“‘" and out[-1] in "\"'”’":
        out = out[1:-1].strip()
    return out


def has_repetition_loop(text: str, max_phrase: int = 4, min_repeats: int = 4) -> bool:
    """True for degenerate loops: a word/phrase repeated consecutively, or a
    long single-character run (catches no-whitespace scripts)."""
    if re.search(r"(.)\1{9,}", text):
        return True
    tokens = text.split()
    for size in range(1, max_phrase + 1):
        for offset in range(size):
            repeats, prev = 1, None
            for i in range(offset, len(tokens) - size + 1, size):
                block = tuple(tokens[i : i + size])
                if block == prev:
                    repeats += 1
                    if repeats >= min_repeats:
                        return True
                else:
                    repeats, prev = 1, block
    return False


def _script_ratio(text: str, tgt_lang: str) -> float | None:
    ranges = SCRIPT_RANGES.get(tgt_lang)
    if not ranges:
        return None  # Latin-script target: no reliable cheap check
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 4:
        return None  # too short to judge (numbers, names, interjections)
    in_script = sum(1 for c in letters if any(lo <= ord(c) <= hi for lo, hi in ranges))
    return in_script / len(letters)


def find_issues(source: str, translation: str, src_lang: str, tgt_lang: str) -> list[Issue]:
    """Deterministic per-line checks. Bounds are deliberately generous —
    flagging a good line (which triggers retry/quarantine) costs more than
    missing a borderline one."""
    issues: list[Issue] = []
    stripped = translation.strip()

    if not stripped:
        return [Issue("empty")]
    if src_lang != tgt_lang and stripped.lower() == source.strip().lower():
        issues.append(Issue("source_copy"))
    if has_repetition_loop(stripped):
        issues.append(Issue("repetition"))

    if len(source) >= 12:
        ratio = len(stripped) / len(source)
        if ratio > 3.0 and len(stripped) > 40:
            issues.append(Issue("too_long", f"{ratio:.1f}x source length"))
        elif ratio < 0.15:
            issues.append(Issue("too_short", f"{ratio:.2f}x source length"))
    if len(stripped) > 400:
        if not any(issue.code == "too_long" for issue in issues):
            issues.append(Issue("too_long", f"{len(stripped)} chars"))

    ratio = _script_ratio(stripped, tgt_lang)
    if ratio is not None and ratio < 0.3:
        issues.append(Issue("wrong_language", f"only {ratio:.0%} target-script letters"))

    lowered = stripped.lower()
    if any(lowered.startswith(marker) for marker in META_MARKERS):
        issues.append(Issue("meta_text", stripped[:40]))

    return issues


def max_new_tokens_for(source: str, configured: int) -> int:
    """Generation cap proportional to the source line: a subtitle translation
    never legitimately needs ~2 tokens per source character, so a runaway
    loop is cut early instead of burning to the global limit."""
    return max(64, min(configured, 16 + 2 * len(source)))


def summarize(per_line_issues: list[list[Issue]]) -> dict[str, int]:
    """Issue-code counts across a document."""
    return dict(Counter(issue.code for issues in per_line_issues for issue in issues))
