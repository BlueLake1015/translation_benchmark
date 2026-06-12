#!/usr/bin/env python3
"""Convert models to CTranslate2 format under the local models/ directory.

Each conversion lands in ``models/<model-key>-ct2`` — the default lookup
path of the ct2 engine, which is the optimized (and default) engine for the
Tier 4 models. Conversions are one-time, offline, and non-destructive: the
base weights are read from ``models/<model-key>/`` and a new directory is
written. Already-converted models are skipped unless --force is given.

With no arguments, every ct2-capable model in the registry is converted.

Examples:
    python scripts/convert_ct2.py --dry-run     # show the plan
    python scripts/convert_ct2.py               # all ct2-capable models
    python scripts/convert_ct2.py nllb200-3.3b
    python scripts/convert_ct2.py --quantization int8_float16

Prerequisites: pip install 'translation-benchmark[ct2,models]' and the base
weights downloaded first (scripts/download_models.py). Note the converter
loads the full model into RAM (~40 GB for MADLAD-400 10B in fp32).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from translation_benchmark.models.base import ModelSpec
from translation_benchmark.models.paths import models_dir_or_default
from translation_benchmark.models.registry import get_spec, list_specs

QUANTIZATIONS = ("int8", "int8_float16", "int16", "float16", "bfloat16", "float32")


def ct2_capable_specs() -> list[ModelSpec]:
    return [spec for spec in list_specs() if "ct2" in spec.engines]


def select_specs(keys: list[str]) -> list[ModelSpec]:
    # No selection means every ct2-capable model.
    if not keys:
        return ct2_capable_specs()
    selected = []
    for key in keys:
        spec = get_spec(key)
        if "ct2" not in spec.engines:
            raise SystemExit(
                f"{key!r} does not support the ct2 engine "
                f"(supported: {', '.join(spec.engines)}). ct2-capable models: "
                + ", ".join(s.key for s in ct2_capable_specs())
            )
        selected.append(spec)
    return selected


def base_dir(spec: ModelSpec, models_dir: Path) -> Path:
    return models_dir / spec.key


def target_dir(spec: ModelSpec, models_dir: Path) -> Path:
    return models_dir / f"{spec.key}-ct2"


def is_converted(target: Path) -> bool:
    return (target / "model.bin").is_file()


def convert(spec: ModelSpec, models_dir: Path, quantization: str, force: bool) -> Path:
    from ctranslate2.converters import TransformersConverter

    out = target_dir(spec, models_dir)
    converter = TransformersConverter(str(base_dir(spec, models_dir)))
    converter.convert(str(out), quantization=quantization, force=force)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("keys", nargs="*",
                        help="Model keys (see: tb list-models) [default: all ct2-capable].")
    parser.add_argument("--quantization", default="int8", choices=QUANTIZATIONS,
                        help="CTranslate2 quantization [default: int8].")
    parser.add_argument("--models-dir", default=None,
                        help="Models directory [default: models, or $TB_MODELS_DIR].")
    parser.add_argument("--force", action="store_true",
                        help="Re-convert even if the ct2 model already exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print the conversion plan.")
    args = parser.parse_args(argv)

    models_dir = models_dir_or_default(args.models_dir)
    todo = []
    for spec in select_specs(args.keys):
        source = base_dir(spec, models_dir)
        target = target_dir(spec, models_dir)
        if is_converted(target) and not args.force:
            action = "skip (converted)"
        elif not (source / "config.json").is_file():
            action = "MISSING WEIGHTS"
        else:
            action = "convert"
            todo.append(spec)
        print(f"  [{action:>16}] {spec.key:<22} {source} -> {target}")

    missing = [
        spec.key
        for spec in select_specs(args.keys)
        if not (base_dir(spec, models_dir) / "config.json").is_file()
    ]
    if missing:
        print(
            f"\nBase weights missing for: {', '.join(missing)}. Download them first:\n"
            f"  python scripts/download_models.py {' '.join(missing)}",
            file=sys.stderr,
        )

    print(f"\n{len(todo)} conversion(s) ({args.quantization}) into {models_dir}/.")
    if args.dry_run or not todo:
        return 1 if missing else 0

    failures = []
    for spec in todo:
        print(f"\n=== {spec.key} ===")
        try:
            print(f"done: {convert(spec, models_dir, args.quantization, args.force)}")
        except ImportError:
            print(
                "FAILED: CTranslate2 stack not installed. Run: "
                "pip install 'translation-benchmark[ct2,models]'",
                file=sys.stderr,
            )
            return 1
        except Exception as exc:  # keep going; report at the end
            failures.append(spec.key)
            print(f"FAILED: {spec.key}: {exc}", file=sys.stderr)

    if failures or missing:
        if failures:
            print(f"\nFailed conversions: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("\nAll conversions complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
