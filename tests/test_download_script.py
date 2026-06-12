"""Tests for scripts/download_models.py (no network access)."""
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "download_models.py"
spec = importlib.util.spec_from_file_location("download_models", SCRIPT)
download_models = importlib.util.module_from_spec(spec)
sys.modules["download_models"] = download_models  # dataclasses needs this registered
spec.loader.exec_module(download_models)


def test_select_all_returns_full_lineup():
    specs = download_models.select_specs([], [], all_models=True)
    assert len(specs) == 9


def test_select_by_tier_and_key_deduplicates():
    specs = download_models.select_specs(["nllb200-3.3b"], [4], all_models=False)
    assert {s.key for s in specs} == {"madlad400-10b", "nllb200-3.3b"}


def test_select_nothing_defaults_to_all_models():
    specs = download_models.select_specs([], [], all_models=False)
    assert len(specs) == 9


def test_select_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        download_models.select_specs(["gpt-7"], [], all_models=False)


def test_qwen_default_skips_full_precision_base():
    # Qwen3 has AWQ/FP8 repos, so the bf16 base is not downloaded by default.
    spec = download_models.get_spec("qwen3-14b")
    items = download_models.variant_items(spec)
    assert [item.variant for item in items] == ["awq", "fp8"]
    assert items[0].dir_name == "qwen3-14b@awq"
    assert items[0].repo_id == "Qwen/Qwen3-14B-AWQ"


def test_explicit_full_still_downloads_qwen_base():
    spec = download_models.get_spec("qwen3-14b")
    items = download_models.variant_items(spec, ["full"])
    assert [item.variant for item in items] == ["full"]
    assert items[0].dir_name == "qwen3-14b"
    assert items[0].repo_id == "Qwen/Qwen3-14B"


def test_variant_items_quant_filter():
    spec = download_models.get_spec("qwen3-14b")
    assert [i.variant for i in download_models.variant_items(spec, ["awq"])] == ["awq"]
    assert [i.variant for i in download_models.variant_items(spec, ["full", "awq"])] == [
        "full",
        "awq",
    ]


def test_variant_items_model_without_quant_repos_keeps_full_by_default():
    spec = download_models.get_spec("nllb200-3.3b")
    items = download_models.variant_items(spec)
    assert [item.variant for item in items] == ["full"]


def test_legacy_weights_skipped_when_safetensors_present():
    patterns = download_models.ignore_patterns_for(
        ["config.json", "model.safetensors", "pytorch_model.bin", "tf_model.h5"]
    )
    assert "*.bin" in patterns and "*.h5" in patterns
    assert "*.gguf" in patterns  # defaults are kept


def test_legacy_weights_kept_when_they_are_the_only_weights():
    patterns = download_models.ignore_patterns_for(["config.json", "pytorch_model.bin"])
    assert "*.bin" not in patterns
    assert patterns == download_models.DEFAULT_IGNORE_PATTERNS


def test_is_downloaded_requires_config_json(tmp_path):
    target = tmp_path / "tower-plus-9b"
    assert not download_models.is_downloaded(target)
    target.mkdir()
    assert not download_models.is_downloaded(target)  # dir alone is not enough
    (target / "config.json").write_text("{}")
    assert download_models.is_downloaded(target)


def test_dry_run_downloads_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        download_models, "download", lambda *a, **k: pytest.fail("should not download")
    )
    # No model selection: defaults to the full lineup with all variants.
    code = download_models.main(["--dry-run", "--models-dir", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    # 7 base models (Qwen bf16 bases skipped by default) + 2x2 Qwen quant repos.
    assert "11 download(s)" in out
    assert "qwen3-32b@awq" in out and "qwen3-14b@fp8" in out
    assert " qwen3-32b " not in out  # bf16 base not in the default plan


def test_main_skips_present_and_downloads_missing(tmp_path, monkeypatch):
    present = tmp_path / "nllb200-3.3b"
    present.mkdir(parents=True)
    (present / "config.json").write_text("{}")

    downloaded = []
    monkeypatch.setattr(
        download_models,
        "download",
        lambda item, models_dir, token=None: downloaded.append(item.dir_name) or tmp_path,
    )
    code = download_models.main(
        ["nllb200-3.3b", "madlad400-10b", "--models-dir", str(tmp_path)]
    )
    assert code == 0
    assert downloaded == ["madlad400-10b"]


def test_main_quant_filter_downloads_only_that_variant(tmp_path, monkeypatch):
    downloaded = []
    monkeypatch.setattr(
        download_models,
        "download",
        lambda item, models_dir, token=None: downloaded.append(item.dir_name) or tmp_path,
    )
    code = download_models.main(
        ["qwen3-14b", "--quant", "awq", "--models-dir", str(tmp_path)]
    )
    assert code == 0
    assert downloaded == ["qwen3-14b@awq"]


def test_main_unmatched_quant_filter_errors(tmp_path, capsys):
    code = download_models.main(
        ["nllb200-3.3b", "--quant", "awq", "--models-dir", str(tmp_path)]
    )
    assert code == 1
    assert "No download items" in capsys.readouterr().err


def test_main_reports_failures(tmp_path, monkeypatch, capsys):
    def boom(item, models_dir, token=None):
        raise RuntimeError("403 Client Error: gated repo")

    monkeypatch.setattr(download_models, "download", boom)
    code = download_models.main(["translategemma-4b", "--models-dir", str(tmp_path)])
    assert code == 1
    err = capsys.readouterr().err
    assert "FAILED: translategemma-4b" in err
    assert "accept the license" in err
