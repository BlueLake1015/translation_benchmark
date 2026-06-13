#!/usr/bin/env python3
"""Manually run real models over a subtitle file (not part of the test suite).

Sweeps every selected model across ALL of its supported inference engines
(vllm / ct2 / transformers), translating the input .srt from local weights.
Each output carries both the model key and the engine in its filename:

    <output-dir>/<model-key>.<engine>.<input-stem>.<tgt>.srt

A (model, engine) pair is skipped when its local weights are absent (for
ct2: the converted models/<key>-ct2 directory; for vllm/transformers: the
base weights, or a downloaded quant variant). Pass --engine to restrict the
sweep to a single engine.

Each (model, engine) runs in its OWN SUBPROCESS so all GPU memory is
reclaimed between runs — some in-process loads (e.g. AWQ via gptqmodel) do
not release GPU memory on unload, which would otherwise starve the next
vLLM server. Pass --no-isolate to run in the current process instead.

Each run logs detailed, timestamped phases (resolve -> load -> translate ->
guard -> write) plus every cue's source -> translation by default; pass
--quiet to suppress the per-cue lines.

Examples:
    python scripts/run_models.py                      # every model, every engine
    python scripts/run_models.py tower-plus-9b qwen3-32b
    python scripts/run_models.py --engine transformers       # one engine only
    python scripts/run_models.py translategemma-4b --quiet    # phases only
    python scripts/run_models.py --input movie.en.srt --output-dir /tmp/out
    python scripts/run_models.py qwen3-32b --quant awq --no-guard

Stage weights first: python scripts/download_models.py [keys]
                     python scripts/convert_ct2.py        (Tier 4)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from translation_benchmark import guards
from translation_benchmark.models.base import ModelSpec
from translation_benchmark.models.paths import models_dir_or_default, resolve_model_source
from translation_benchmark.models.registry import create_translator, get_spec, list_specs
from translation_benchmark.subtitles import (
    load_srt,
    prepare_for_translation,
    save_srt,
    translated_copy,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "tests" / "data" / "night_shift.en.srt"
DEFAULT_OUTPUT_DIR = ROOT / "tests" / "test_result"


def local_run_kwargs(spec: ModelSpec, engine: str, models_dir: Path) -> dict | None:
    """Kwargs to run a model on an engine from local weights, or None if the
    required weights are not present for that engine."""
    base_present = (models_dir / spec.key / "config.json").is_file()
    if engine == "ct2":
        # The converted dir holds the weights; the tokenizer comes from the
        # base dir if present, else it is fetched (tiny) — so the base weights
        # being deleted after conversion is fine.
        converted = (models_dir / f"{spec.key}-ct2" / "model.bin").is_file()
        return {} if converted else None
    # transformers / vllm: base weights, or a downloaded pre-quantized variant.
    if base_present:
        return {}
    for name, _repo in spec.quant_repos:
        if (models_dir / f"{spec.key}@{name}" / "config.json").is_file():
            return {"quant": name}
    return None


def _first_line(exc: Exception) -> str:
    lines = str(exc).splitlines()
    summary = lines[0] if lines else ""
    return f"{type(exc).__name__}: {summary}" if summary else type(exc).__name__


def log(message: str) -> None:
    """Timestamped detail line, flushed immediately for live tailing."""
    print(f"  {time.strftime('%H:%M:%S')}  {message}", flush=True)


def run_one(spec, engine, args, lines, texts, stem, out_dir, models_dir) -> tuple[str, str]:
    """Translate the document with one (model, engine). Returns (status, detail);
    status is SUCCESS | FAIL | SKIP. Does all phase/per-cue logging."""
    label = f"{spec.key}/{engine}"
    kwargs = local_run_kwargs(spec, engine, models_dir)
    if kwargs is None:
        print(f"[skip] {label}: weights not under {models_dir}")
        return ("SKIP", f"weights not under {models_dir}")
    if args.quant:
        kwargs["quant"] = args.quant
    quant = kwargs.get("quant", "none")
    print(f"\n[run ] {label}  engine={engine} quant={quant} "
          f"context={args.context_window} guard={not args.no_guard}", flush=True)
    try:
        translator = create_translator(
            spec.key, engine=engine, models_dir=str(models_dir), **kwargs
        )
        try:
            src_path, _ = resolve_model_source(
                kwargs.get("hf_id") or spec.hf_id,
                f"{spec.key}@{quant}" if quant != "none" else spec.key,
                str(models_dir),
            )
            src_path = getattr(translator, "ct2_dir", src_path)  # ct2 weights dir
            log(f"loading weights ({src_path}) ...")
            t0 = time.perf_counter()
            translator.load()
            log(f"loaded in {time.perf_counter() - t0:.1f}s; "
                f"translating {len(texts)} cues ({sum(len(t) for t in texts)} chars) ...")
            start = time.perf_counter()
            translations = translator.translate_document(
                texts, args.src, args.tgt,
                max_context_pairs=args.context_window, guard=not args.no_guard,
            )
            elapsed = time.perf_counter() - start
            per_line = getattr(translator, "last_issues", [[]] * len(texts))
            flagged = sum(1 for issues in per_line if issues)
            log(f"translated in {elapsed:.1f}s ({len(texts) / elapsed:.1f} seg/s, "
                f"{sum(len(t) for t in texts) / elapsed:.0f} char/s)")
            if not args.no_guard:
                counts = guards.summarize(per_line)
                log(f"guard: {flagged}/{len(texts)} flagged"
                    + (f" ({counts})" if counts else ""))
            if not args.quiet:
                for line, src, hyp, issues in zip(lines, texts, translations, per_line):
                    mark = "!!" if issues else "  "
                    codes = " [" + ",".join(i.code for i in issues) + "]" if issues else ""
                    log(f"  {mark} #{line.index}: {src!r} -> {hyp!r}{codes}")
            out_path = out_dir / f"{spec.key}.{engine}.{stem}.{args.tgt}.srt"
            save_srt(out_path, translated_copy(lines, translations))
            log("unloading engine ...")
            detail = (
                f"{len(texts)} cues, {elapsed:.1f}s ({len(texts) / elapsed:.1f} seg/s), "
                f"{flagged} flagged -> {out_path.name}"
            )
            return ("SUCCESS", detail)
        finally:
            translator.unload()  # always close the engine (vLLM server etc.)
    except RuntimeError as exc:
        if "not installed" in str(exc):  # missing optional engine stack
            print(f"[skip] {label}: {exc}")
            return ("SKIP", _first_line(exc))
        print(f"[FAIL] {label}: {exc}", file=sys.stderr)
        return ("FAIL", _first_line(exc))
    except Exception as exc:
        print(f"[FAIL] {label}: {exc}", file=sys.stderr)
        return ("FAIL", _first_line(exc))


def run_isolated(spec, engine, args, models_dir) -> tuple[str, str]:
    """Run one (model, engine) in a fresh subprocess so its GPU memory is
    fully released on exit. Logs stream live (inherited stdout/stderr); the
    status/detail come back through a small JSON result file."""
    fd, result_file = tempfile.mkstemp(suffix=".json", prefix="tb-run-")
    os.close(fd)
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker-key", spec.key, "--worker-engine", engine,
        "--result-file", result_file,
        "--input", args.input, "--src", args.src, "--tgt", args.tgt,
        "--output-dir", args.output_dir, "--models-dir", str(models_dir),
        "--context-window", str(args.context_window),
    ]
    if args.quant:
        cmd += ["--quant", args.quant]
    if args.no_guard:
        cmd += ["--no-guard"]
    if args.quiet:
        cmd += ["--quiet"]
    try:
        proc = subprocess.run(cmd)  # inherits stdout/stderr -> live logs
        try:
            data = json.loads(Path(result_file).read_text())
            return (data["status"], data["detail"])
        except (OSError, ValueError, KeyError):
            return ("FAIL", f"worker produced no result (exit code {proc.returncode})")
    finally:
        Path(result_file).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("keys", nargs="*",
                        help="Model keys [default: every model with local weights].")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help=f"Input .srt [default: {DEFAULT_INPUT.relative_to(ROOT)}].")
    parser.add_argument("--src", default="en")
    parser.add_argument("--tgt", default="ko")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help=f"Output directory [default: {DEFAULT_OUTPUT_DIR.relative_to(ROOT)}].")
    parser.add_argument("--models-dir", default=None,
                        help="Models directory [default: models, or $TB_MODELS_DIR].")
    parser.add_argument("--quant", default=None, help="Quantization variant override.")
    parser.add_argument("--engine", default=None,
                        help="Restrict the sweep to one engine (vllm/ct2/transformers) "
                        "[default: every engine each model supports].")
    parser.add_argument("--context-window", type=int, default=8)
    parser.add_argument("--no-guard", action="store_true",
                        help="Disable hallucination detection/mitigation.")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress the per-cue source -> translation log "
                        "(phase/timing logs still print).")
    parser.add_argument("--no-isolate", action="store_true",
                        help="Run in the current process instead of one subprocess "
                        "per (model, engine).")
    # Internal: single-run worker invoked by run_isolated (hidden from --help).
    parser.add_argument("--worker-key", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-engine", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    models_dir = models_dir_or_default(args.models_dir).resolve()
    lines = load_srt(args.input)
    texts = [prepare_for_translation(line.text) for line in lines]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem.removesuffix(f".{args.src}")

    # Worker mode: run exactly one (model, engine) and report via result file.
    if args.worker_key:
        spec = get_spec(args.worker_key)
        status, detail = run_one(
            spec, args.worker_engine, args, lines, texts, stem, out_dir, models_dir
        )
        if args.result_file:
            Path(args.result_file).write_text(
                json.dumps({"status": status, "detail": detail})
            )
        return 0 if status != "FAIL" else 1

    # Parent sweep.
    specs = [get_spec(key) for key in args.keys] if args.keys else list_specs()
    results: list[tuple[str, str, str]] = []  # (label, status, detail)
    for spec in specs:
        engines = spec.supported_engines()
        if args.engine:
            engines = tuple(e for e in engines if e == args.engine)
            if not engines:
                label = f"{spec.key}/{args.engine}"
                print(f"[skip] {label}: engine not supported by this model")
                results.append((label, "SKIP", "engine not supported by this model"))
                continue
        for engine in engines:
            label = f"{spec.key}/{engine}"
            if args.no_isolate:
                status, detail = run_one(
                    spec, engine, args, lines, texts, stem, out_dir, models_dir
                )
            else:
                status, detail = run_isolated(spec, engine, args, models_dir)
            results.append((label, status, detail))

    print("\n=== Results ===")
    for label, status, detail in results:
        print(f"  {status:<8} {label:<32} {detail}")
    counts = {status: sum(1 for _, s, _ in results if s == status)
              for status in ("SUCCESS", "FAIL", "SKIP")}
    print(f"\n{counts['SUCCESS']} success, {counts['FAIL']} fail, {counts['SKIP']} skip")
    return 1 if counts["FAIL"] or not counts["SUCCESS"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
