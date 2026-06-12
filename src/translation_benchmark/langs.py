"""Language code mappings for the different model backends.

Chat models (TranslateGemma, Qwen3, Tower) take plain English language names
in the prompt. MADLAD-400 uses ``<2xx>`` target prefixes on ISO 639-1 codes.
NLLB-200 uses FLORES-200 codes.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str  # ISO 639-1, used on the CLI and by MADLAD prefixes
    name: str  # English name, used in chat prompts
    flores: str  # FLORES-200 code, used by NLLB


LANGUAGES: dict[str, Language] = {
    lang.code: lang
    for lang in [
        Language("en", "English", "eng_Latn"),
        Language("ko", "Korean", "kor_Hang"),
        Language("ja", "Japanese", "jpn_Jpan"),
        Language("zh", "Chinese (Simplified)", "zho_Hans"),
        Language("es", "Spanish", "spa_Latn"),
        Language("fr", "French", "fra_Latn"),
        Language("de", "German", "deu_Latn"),
        Language("it", "Italian", "ita_Latn"),
        Language("pt", "Portuguese", "por_Latn"),
        Language("ru", "Russian", "rus_Cyrl"),
        Language("ar", "Arabic", "arb_Arab"),
        Language("hi", "Hindi", "hin_Deva"),
        Language("vi", "Vietnamese", "vie_Latn"),
        Language("th", "Thai", "tha_Thai"),
        Language("id", "Indonesian", "ind_Latn"),
        Language("tr", "Turkish", "tur_Latn"),
        Language("pl", "Polish", "pol_Latn"),
        Language("nl", "Dutch", "nld_Latn"),
        Language("uk", "Ukrainian", "ukr_Cyrl"),
    ]
}


def get_language(code: str) -> Language:
    try:
        return LANGUAGES[code.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown language code {code!r}. Known codes: {', '.join(sorted(LANGUAGES))}"
        ) from None
