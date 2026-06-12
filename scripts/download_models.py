#!/usr/bin/env python3
"""Download benchmark models into the local models/ directory.

Each model lands in ``models/<model-key>/`` and each pre-quantized variant
(e.g. Qwen's AWQ/FP8 repos) in ``models/<model-key>@<variant>/`` — the
layouts the harness loads directly, fully offline. Already-downloaded
items are skipped unless --force is given.

With no arguments, the full 9-model lineup INCLUDING all quantization
variants is downloaded. Runtime quantization (--quant 4bit/8bit at usage
time) needs no separate download; it reuses the base weights.

Examples:
    # See what the default (all models, all variants) would download
    python scripts/download_models.py --dry-run

    # Download everything
    python scripts/download_models.py

    # Two specific models (with their variants)
    python scripts/download_models.py qwen3-14b translategemma-4b

    # Base weights only / only the AWQ variants
    python scripts/download_models.py --quant full
    python scripts/download_models.py --quant awq

    # Everything that fits a laptop (Tier 3) plus the draft engines (Tier 4)
    python scripts/download_models.py --tier 3 --tier 4

Gated models (TranslateGemma requires accepting Google's license on the
Hugging Face page first) need an authenticated token: pass --token, set
HF_TOKEN, or run `huggingface-cli login` once.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from translation_benchmark.models.base import ModelSpec
from translation_benchmark.models.paths import models_dir_or_default
from translation_benchmark.models.registry import get_spec, list_specs

# Skip alternate weight formats; transformers loads the safetensors/bin shards.
DEFAULT_IGNORE_PATTERNS = ["*.gguf", "original/*"]
# Duplicate legacy weight copies (PyTorch .bin, TF, Flax, ...) — only skipped
# when the repo also ships .safetensors, otherwise they ARE the weights.
LEGACY_WEIGHT_PATTERNS = ["*.bin", "*.pth", "*.ckpt", "*.h5", "*.msgpack"]

# Rough download footprint in GB per billion params, by variant.
SIZE_FACTORS = {"full": 2.1, "fp8": 1.2}
DEFAULT_QUANT_FACTOR = 0.7  # 4-bit-ish repos (awq, gptq, ...)


@dataclass(frozen=True)
class DownloadItem:
    key: str  # model key
    variant: str  # "full" or a quant variant name
    repo_id: str
    dir_name: str  # directory under models/
    approx_gb: float


def select_specs(keys: list[str], tiers: list[int], all_models: bool = False) -> list[ModelSpec]:
    # No selection means everything: downloading the full lineup is the default.
    if all_models or (not keys and not tiers):
        return list_specs()
    selected: dict[str, ModelSpec] = {}
    for tier in tiers:
        for spec in list_specs(tier=tier):
            selected[spec.key] = spec
    for key in keys:
        spec = get_spec(key)
        selected[spec.key] = spec
    return sorted(selected.values(), key=lambda spec: (spec.tier, -spec.params_b))


def variant_items(spec: ModelSpec, quants: list[str] | None = None) -> list[DownloadItem]:
    """Downloadable items for a model: base weights plus pre-quantized repos.

    ``quants`` filters by variant name ("full" = the base weights); empty or
    None means the model's defaults — all variants, except that models with
    ``download_full_by_default=False`` (Qwen3: AWQ/FP8 cover usage) skip the
    full-precision base unless "full" is requested explicitly.
    """
    include_full = "full" in quants if quants else spec.download_full_by_default
    items = []
    if include_full:
        items.append(
            DownloadItem(
                key=spec.key,
                variant="full",
                repo_id=spec.hf_id,
                dir_name=spec.key,
                approx_gb=round(spec.params_b * SIZE_FACTORS["full"], 1),
            )
        )
    for name, repo in spec.quant_repos:
        factor = SIZE_FACTORS.get(name, DEFAULT_QUANT_FACTOR)
        items.append(
            DownloadItem(
                key=spec.key,
                variant=name,
                repo_id=repo,
                dir_name=f"{spec.key}@{name}",
                approx_gb=round(spec.params_b * factor, 1),
            )
        )
    if quants:
        items = [item for item in items if item.variant in quants]
    return items


def is_downloaded(target: Path) -> bool:
    return (target / "config.json").is_file()


def ignore_patterns_for(repo_files: list[str]) -> list[str]:
    """Ignore patterns for a repo, given its file listing.

    Repos that ship safetensors alongside legacy formats (e.g. a duplicate
    pytorch_model.bin copy of the same weights) only need the safetensors —
    skipping the duplicates roughly halves some downloads.
    """
    patterns = list(DEFAULT_IGNORE_PATTERNS)
    if any(name.endswith(".safetensors") for name in repo_files):
        patterns.extend(LEGACY_WEIGHT_PATTERNS)
    return patterns


def download(item: DownloadItem, models_dir: Path, token: str | None = None) -> Path:
    from huggingface_hub import list_repo_files, snapshot_download

    return Path(
        snapshot_download(
            repo_id=item.repo_id,
            local_dir=models_dir / item.dir_name,
            ignore_patterns=ignore_patterns_for(list_repo_files(item.repo_id, token=token)),
            token=token,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("keys", nargs="*",
                        help="Model keys (see: tb list-models) [default: all models].")
    parser.add_argument("--tier", type=int, action="append", default=[],
                        choices=[1, 2, 3, 4], help="Download a whole tier; repeatable.")
    parser.add_argument("--all", action="store_true",
                        help="Download the full 9-model lineup (the default when nothing is selected).")
    parser.add_argument("--quant", action="append", default=[],
                        help="Only these variants; repeatable ('full' = base weights, "
                        "or a variant name like awq/fp8) [default: all variants].")
    parser.add_argument("--models-dir", default=None,
                        help="Target directory [default: models, or $TB_MODELS_DIR].")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the model is already present.")
    parser.add_argument("--token", default=None,
                        help="Hugging Face token for gated repos [default: HF_TOKEN / cached login].")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print the download plan.")
    args = parser.parse_args(argv)

    models_dir = models_dir_or_default(args.models_dir)
    items = [
        item
        for spec in select_specs(args.keys, args.tier, args.all)
        for item in variant_items(spec, args.quant)
    ]
    if not items:
        print(f"No download items match --quant {args.quant}.", file=sys.stderr)
        return 1

    todo = []
    for item in items:
        present = is_downloaded(models_dir / item.dir_name)
        action = "skip (present)" if present and not args.force else "download"
        if action == "download":
            todo.append(item)
        print(f"  [{action:>14}] {item.dir_name:<28} {item.repo_id:<32} ~{item.approx_gb} GB")

    total = sum(item.approx_gb for item in todo)
    print(f"\n{len(todo)} download(s) into {models_dir}/ (~{total:.0f} GB total).")
    if args.dry_run or not todo:
        return 0

    failures = []
    for item in todo:
        print(f"\n=== {item.dir_name} ({item.repo_id}) ===")
        try:
            path = download(item, models_dir, token=args.token)
            print(f"done: {path}")
        except Exception as exc:  # keep going; report at the end
            failures.append((item.dir_name, exc))
            message = str(exc)
            print(f"FAILED: {item.dir_name}: {message}", file=sys.stderr)
            if "gated" in message.lower() or "401" in message or "403" in message:
                print(
                    f"hint: accept the license at https://huggingface.co/{item.repo_id} "
                    "and authenticate (--token, HF_TOKEN, or `huggingface-cli login`).",
                    file=sys.stderr,
                )

    if failures:
        print(f"\n{len(failures)} of {len(todo)} downloads failed: "
              + ", ".join(name for name, _ in failures), file=sys.stderr)
        return 1
    print("\nAll downloads complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
