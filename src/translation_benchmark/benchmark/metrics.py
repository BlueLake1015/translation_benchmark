"""Quality metrics for hypothesis vs. reference subtitle texts.

chrF++ is the primary metric (robust at corpus and segment level, no
tokenizer dependence — important for Korean). BLEU is reported for
familiarity. COMET (optional extra) is the strongest correlate of human
judgment but needs torch.
"""
from __future__ import annotations


def _check_lengths(hypotheses: list[str], references: list[str]) -> None:
    if len(hypotheses) != len(references):
        raise ValueError(
            f"Hypotheses ({len(hypotheses)}) and references ({len(references)}) differ in length"
        )
    if not hypotheses:
        raise ValueError("Cannot score an empty corpus")


def chrf(hypotheses: list[str], references: list[str]) -> float:
    """Corpus chrF++ (chrF with word order 2), 0-100."""
    _check_lengths(hypotheses, references)
    from sacrebleu.metrics import CHRF

    return CHRF(word_order=2).corpus_score(hypotheses, [references]).score


def bleu(hypotheses: list[str], references: list[str], tgt_lang: str = "ko") -> float:
    """Corpus BLEU, 0-100. Uses a char-friendly tokenizer for ko/ja/zh."""
    _check_lengths(hypotheses, references)
    from sacrebleu.metrics import BLEU

    tokenize = "char" if tgt_lang in ("ko", "ja", "zh") else "13a"
    return BLEU(tokenize=tokenize).corpus_score(hypotheses, [references]).score


def comet(
    hypotheses: list[str], references: list[str], sources: list[str]
) -> float:  # pragma: no cover - heavy optional dependency
    """COMET-22 score, 0-1. Requires: pip install 'translation-benchmark[comet]'."""
    _check_lengths(hypotheses, references)
    from comet import download_model, load_from_checkpoint

    model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    data = [
        {"src": src, "mt": hyp, "ref": ref}
        for src, hyp, ref in zip(sources, hypotheses, references)
    ]
    return model.predict(data, progress_bar=False).system_score
