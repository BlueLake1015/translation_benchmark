"""Command-line interface: tb list-models | translate | benchmark."""
from __future__ import annotations

from pathlib import Path

import click

from translation_benchmark.benchmark.runner import (
    results_to_json,
    results_to_markdown,
    run_benchmark,
)
from translation_benchmark.models.registry import create_translator, list_specs
from translation_benchmark.subtitles import (
    load_srt,
    prepare_for_translation,
    save_srt,
    translated_copy,
)


@click.group()
def main() -> None:
    """Benchmark open-weight MT models on film/video subtitles (.srt)."""


@main.command("list-models")
@click.option("--tier", type=int, default=None, help="Only show one tier (1-4).")
@click.option("--all", "include_test", is_flag=True, help="Include the test-only dummy model.")
def list_models(tier: int | None, include_test: bool) -> None:
    """Show the supported model lineup."""
    current_tier = None
    for spec in list_specs(tier=tier, include_test=include_test):
        if spec.tier != current_tier:
            current_tier = spec.tier
            click.echo(f"\nTier {spec.tier} — {spec.tier_label}")
        context = (
            f"context (~{spec.approx_context_tokens // 1000}k tokens)"
            if spec.supports_context and spec.approx_context_tokens
            else ("context" if spec.supports_context else "SENTENCE-LEVEL ONLY")
        )
        click.echo(f"  {spec.key:<22} {spec.display_name:<22} {context:<28} {spec.vram_note}")
        click.echo(f"  {'':<22} {spec.notes}")


def _common_model_kwargs(hf_id: str | None, load_in_4bit: bool, use_ct2: bool, ct2_dir: str | None):
    kwargs: dict = {}
    if hf_id:
        kwargs["hf_id"] = hf_id
    if load_in_4bit:
        kwargs["load_in_4bit"] = True
    if use_ct2:
        kwargs["use_ct2"] = True
    if ct2_dir:
        kwargs["use_ct2"] = True
        kwargs["ct2_dir"] = ct2_dir
    return kwargs


@main.command()
@click.argument("input_srt", type=click.Path(exists=True, dir_okay=False))
@click.option("--model", "-m", required=True, help="Model key (see: tb list-models).")
@click.option("--src", "-s", default="en", show_default=True, help="Source language code.")
@click.option("--tgt", "-t", default="ko", show_default=True, help="Target language code.")
@click.option("--output", "-o", type=click.Path(dir_okay=False), default=None,
              help="Output .srt path [default: <input>.<tgt>.srt].")
@click.option("--context-window", default=8, show_default=True,
              help="Previous subtitle pairs fed as context (context-aware models only).")
@click.option("--device", default="auto", show_default=True)
@click.option("--hf-id", default=None, help="Override the Hugging Face repo id.")
@click.option("--load-in-4bit", is_flag=True, help="4-bit quantization (chat models).")
@click.option("--ct2-dir", default=None, help="CTranslate2 model dir (NLLB only).")
def translate(
    input_srt: str,
    model: str,
    src: str,
    tgt: str,
    output: str | None,
    context_window: int,
    device: str,
    hf_id: str | None,
    load_in_4bit: bool,
    ct2_dir: str | None,
) -> None:
    """Translate a subtitle file, preserving timings."""
    translator = create_translator(
        model, device=device, **_common_model_kwargs(hf_id, load_in_4bit, False, ct2_dir)
    )
    lines = load_srt(input_srt)
    texts = [prepare_for_translation(line.text) for line in lines]
    if not translator.supports_context and context_window:
        click.echo(
            f"note: {translator.spec.display_name} is sentence-level; context is ignored.",
            err=True,
        )
    translations = translator.translate_document(
        texts, src, tgt, max_context_pairs=context_window
    )
    out_path = output or str(Path(input_srt).with_suffix(f".{tgt}.srt"))
    save_srt(out_path, translated_copy(lines, translations))
    click.echo(f"Wrote {len(translations)} cues to {out_path}")


@main.command()
@click.argument("input_srt", type=click.Path(exists=True, dir_okay=False))
@click.option("--model", "-m", "models", multiple=True, required=True,
              help="Model key; repeat for several models.")
@click.option("--src", "-s", default="en", show_default=True)
@click.option("--tgt", "-t", default="ko", show_default=True)
@click.option("--reference", "-r", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Reference .srt in the target language (enables chrF++/BLEU).")
@click.option("--output-dir", type=click.Path(file_okay=False), default="results",
              show_default=True)
@click.option("--context-window", default=8, show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--hf-id", default=None, help="Override repo id (single-model runs).")
@click.option("--load-in-4bit", is_flag=True)
@click.option("--ct2-dir", default=None, help="CTranslate2 model dir (NLLB only).")
def benchmark(
    input_srt: str,
    models: tuple[str, ...],
    src: str,
    tgt: str,
    reference: str | None,
    output_dir: str,
    context_window: int,
    device: str,
    hf_id: str | None,
    load_in_4bit: bool,
    ct2_dir: str | None,
) -> None:
    """Benchmark one or more models on a subtitle file."""
    lines = load_srt(input_srt)
    reference_texts = None
    if reference:
        ref_lines = load_srt(reference)
        if len(ref_lines) != len(lines):
            raise click.ClickException(
                f"Reference has {len(ref_lines)} cues but input has {len(lines)}; "
                "they must align 1:1."
            )
        reference_texts = [line.text for line in ref_lines]

    results = []
    for key in models:
        click.echo(f"Benchmarking {key} ...")
        translator = create_translator(
            key, device=device, **_common_model_kwargs(hf_id, load_in_4bit, False, ct2_dir)
        )
        result = run_benchmark(
            translator,
            lines,
            src,
            tgt,
            reference_texts=reference_texts,
            max_context_pairs=context_window,
        )
        if result.error:
            click.echo(f"  ERROR:\n{result.error}", err=True)
        else:
            click.echo(
                f"  {result.num_segments} cues in {result.wall_seconds:.2f}s "
                f"({result.segments_per_second:.2f} seg/s)"
                + (f", chrF++ {result.chrf:.2f}" if result.chrf is not None else "")
            )
        results.append(result)
        translator.unload()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(results_to_json(results), encoding="utf-8")
    (out / "report.md").write_text(results_to_markdown(results) + "\n", encoding="utf-8")
    click.echo(f"\n{results_to_markdown(results)}")
    click.echo(f"\nReports written to {out / 'report.json'} and {out / 'report.md'}")


if __name__ == "__main__":
    main()
