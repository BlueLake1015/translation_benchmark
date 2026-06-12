"""Reading, cleaning, and writing SubRip (.srt) subtitle files."""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path

import srt as srt_lib

_TAG_RE = re.compile(r"</?\s*(i|b|u|font)[^>]*>", re.IGNORECASE)  # HTML-style tags
_ASS_RE = re.compile(r"\{\\[^}]*\}")  # ASS override tags like {\an8}


@dataclass(frozen=True)
class SubtitleLine:
    index: int
    start: timedelta
    end: timedelta
    text: str

    def with_text(self, text: str) -> "SubtitleLine":
        return replace(self, text=text)


def strip_markup(text: str) -> str:
    """Remove HTML-style and ASS formatting tags."""
    return _ASS_RE.sub("", _TAG_RE.sub("", text))


def flatten(text: str) -> str:
    """Collapse a multi-line cue into one line for translation."""
    return " ".join(part.strip() for part in text.splitlines() if part.strip())


def prepare_for_translation(text: str) -> str:
    return flatten(strip_markup(text)).strip()


def load_srt(path: str | Path) -> list[SubtitleLine]:
    content = Path(path).read_text(encoding="utf-8-sig")
    return [
        SubtitleLine(index=sub.index, start=sub.start, end=sub.end, text=sub.content)
        for sub in srt_lib.parse(content)
    ]


def save_srt(path: str | Path, lines: list[SubtitleLine]) -> None:
    subs = [
        srt_lib.Subtitle(index=line.index, start=line.start, end=line.end, content=line.text)
        for line in lines
    ]
    Path(path).write_text(srt_lib.compose(subs), encoding="utf-8")


def translated_copy(lines: list[SubtitleLine], translations: list[str]) -> list[SubtitleLine]:
    """New cue list with identical indices/timings and translated text."""
    if len(lines) != len(translations):
        raise ValueError(
            f"Got {len(translations)} translations for {len(lines)} subtitle lines"
        )
    return [line.with_text(text) for line, text in zip(lines, translations)]
