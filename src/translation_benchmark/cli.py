"""Command-line interface: tb list-models | translate | benchmark."""
from __future__ import annotations

from pathlib import Path

import click

from translation_benchmark import guards
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
        engines = spec.supported_engines()
        engine_label = ", ".join([f"{engines[0]} (default)"] + list(engines[1:]))
        click.echo(f"  {spec.key:<22} {spec.display_name:<22} {context:<28} {spec.vram_note}")
        click.echo(f"  {'':<22} engines: {engine_label}")
        click.echo(f"  {'':<22} quant: {', '.join(spec.quant_variants())}")
        click.echo(f"  {'':<22} {spec.notes}")


def _common_model_kwargs(
    hf_id: str | None,
    quant: str | None,
    ct2_dir: str | None,
    models_dir: str | None = None,
    prompt_style: str | None = None,
):
    kwargs: dict = {}
    if hf_id:
        kwargs["hf_id"] = hf_id
    if models_dir:
        kwargs["models_dir"] = models_dir
    if quant:
        kwargs["quant"] = quant
    if ct2_dir:
        kwargs["ct2_dir"] = ct2_dir
    if prompt_style:
        kwargs["prompt_style"] = prompt_style
    return kwargs


PROMPT_STYLE_OPTION = click.option(
    "--prompt-style",
    type=click.Choice(["translategemma", "qwen", "tower"]),
    default=None,
    help="Override the prompt format for A/B testing [default: the model "
    "family's tuned prompt from the registry].",
)


MODELS_DIR_OPTION = click.option(
    "--models-dir",
    default="models",
    show_default=True,
    envvar="TB_MODELS_DIR",
    help="Local directory model weights are loaded from / downloaded into.",
)

QUANT_OPTION = click.option(
    "--quant",
    default=None,
    help="Quantization variant: 4bit/8bit (on-the-fly bitsandbytes) or a "
    "model-specific pre-quantized repo (e.g. awq, fp8 — see tb list-models).",
)

NO_GUARD_OPTION = click.option(
    "--no-guard",
    is_flag=True,
    help="Disable hallucination detection/mitigation (cleaning, "
    "retry-without-context, context quarantine).",
)

ENGINE_OPTION = click.option(
    "--engine",
    type=click.Choice(["transformers", "vllm", "ct2"]),
    default=None,
    help="Serving stack [default: the model's optimized engine — vllm for "
    "chat models, ct2 for Tier 4; transformers is the fallback].",
)


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
@MODELS_DIR_OPTION
@QUANT_OPTION
@ENGINE_OPTION
@PROMPT_STYLE_OPTION
@NO_GUARD_OPTION
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
    models_dir: str,
    quant: str | None,
    engine: str | None,
    prompt_style: str | None,
    no_guard: bool,
    ct2_dir: str | None,
) -> None:
    """Translate a subtitle file, preserving timings."""
    translator = create_translator(
        model,
        device=device,
        engine=engine,
        **_common_model_kwargs(hf_id, quant, ct2_dir, models_dir, prompt_style),
    )
    lines = load_srt(input_srt)
    texts = [prepare_for_translation(line.text) for line in lines]
    if not translator.supports_context and context_window:
        click.echo(
            f"note: {translator.spec.display_name} is sentence-level; context is ignored.",
            err=True,
        )
    translations = translator.translate_document(
        texts, src, tgt, max_context_pairs=context_window, guard=not no_guard
    )
    out_path = output or str(Path(input_srt).with_suffix(f".{tgt}.srt"))
    save_srt(out_path, translated_copy(lines, translations))
    click.echo(f"Wrote {len(translations)} cues to {out_path}")
    if not no_guard:
        per_line = getattr(translator, "last_issues", [])
        flagged = [(i, issues) for i, issues in enumerate(per_line) if issues]
        if flagged:
            counts = ", ".join(
                f"{code}: {n}" for code, n in sorted(guards.summarize(per_line).items())
            )
            click.echo(
                f"warning: guard flagged {len(flagged)}/{len(per_line)} lines "
                f"({counts}) — review cues "
                + ", ".join(str(lines[i].index) for i, _ in flagged[:10]),
                err=True,
            )


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
@MODELS_DIR_OPTION
@QUANT_OPTION
@ENGINE_OPTION
@PROMPT_STYLE_OPTION
@NO_GUARD_OPTION
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
    models_dir: str,
    quant: str | None,
    engine: str | None,
    prompt_style: str | None,
    no_guard: bool,
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
            key,
            device=device,
            engine=engine,
            **_common_model_kwargs(hf_id, quant, ct2_dir, models_dir, prompt_style),
        )
        result = run_benchmark(
            translator,
            lines,
            src,
            tgt,
            reference_texts=reference_texts,
            max_context_pairs=context_window,
            guard=not no_guard,
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
