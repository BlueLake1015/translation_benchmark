"""Tests for scripts/convert_ct2.py (no ctranslate2 install or weights needed)."""
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "convert_ct2.py"
spec = importlib.util.spec_from_file_location("convert_ct2", SCRIPT)
convert_ct2 = importlib.util.module_from_spec(spec)
sys.modules["convert_ct2"] = convert_ct2
spec.loader.exec_module(convert_ct2)


def _stage_weights(models_dir: Path, key: str) -> None:
    target = models_dir / key
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}")


def test_default_selection_is_all_ct2_capable_models():
    specs = convert_ct2.select_specs([])
    assert {s.key for s in specs} == {"madlad400-10b", "nllb200-3.3b"}


def test_non_ct2_model_rejected_with_hint():
    with pytest.raises(SystemExit, match="does not support the ct2 engine"):
        convert_ct2.select_specs(["qwen3-14b"])


def test_is_converted_requires_model_bin(tmp_path):
    assert not convert_ct2.is_converted(tmp_path)
    (tmp_path / "model.bin").write_text("")
    assert convert_ct2.is_converted(tmp_path)


def test_dry_run_converts_nothing(tmp_path, monkeypatch, capsys):
    _stage_weights(tmp_path, "nllb200-3.3b")
    _stage_weights(tmp_path, "madlad400-10b")
    monkeypatch.setattr(
        convert_ct2, "convert", lambda *a, **k: pytest.fail("should not convert")
    )
    code = convert_ct2.main(["--dry-run", "--models-dir", str(tmp_path)])
    assert code == 0
    assert "2 conversion(s)" in capsys.readouterr().out


def test_missing_base_weights_reported_and_skipped(tmp_path, monkeypatch, capsys):
    converted = []
    monkeypatch.setattr(
        convert_ct2,
        "convert",
        lambda spec, models_dir, quantization, force: converted.append(spec.key) or tmp_path,
    )
    _stage_weights(tmp_path, "nllb200-3.3b")  # madlad weights NOT staged
    code = convert_ct2.main(["--models-dir", str(tmp_path)])
    assert code == 1  # missing weights -> non-zero
    assert converted == ["nllb200-3.3b"]
    err = capsys.readouterr().err
    assert "madlad400-10b" in err and "download_models.py" in err


def test_skips_already_converted(tmp_path, monkeypatch):
    _stage_weights(tmp_path, "nllb200-3.3b")
    ct2_dir = tmp_path / "nllb200-3.3b-ct2"
    ct2_dir.mkdir()
    (ct2_dir / "model.bin").write_text("")

    converted = []
    monkeypatch.setattr(
        convert_ct2,
        "convert",
        lambda spec, models_dir, quantization, force: converted.append(spec.key) or tmp_path,
    )
    code = convert_ct2.main(["nllb200-3.3b", "--models-dir", str(tmp_path)])
    assert code == 0
    assert converted == []


def test_conversion_failure_reported(tmp_path, monkeypatch, capsys):
    _stage_weights(tmp_path, "nllb200-3.3b")

    def boom(spec, models_dir, quantization, force):
        raise RuntimeError("unsupported layer")

    monkeypatch.setattr(convert_ct2, "convert", boom)
    code = convert_ct2.main(["nllb200-3.3b", "--models-dir", str(tmp_path)])
    assert code == 1
    assert "FAILED: nllb200-3.3b" in capsys.readouterr().err
