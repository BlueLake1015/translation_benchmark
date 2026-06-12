"""CLI tests (English -> Korean) via click's test runner and the dummy model."""
import json

from click.testing import CliRunner

from translation_benchmark.cli import main
from translation_benchmark.subtitles import load_srt


def test_list_models_shows_all_tiers_and_sentence_level_note():
    result = CliRunner().invoke(main, ["list-models"])
    assert result.exit_code == 0
    for fragment in (
        "Tier 1", "Tier 2", "Tier 3", "Tier 4",
        "TranslateGemma 27B", "Qwen3-32B", "Tower+ 9B", "Qwen3-14B",
        "TranslateGemma 12B", "TranslateGemma 4B", "TowerInstruct-7B-v0.2",
        "MADLAD-400 10B", "NLLB-200 3.3B",
        "SENTENCE-LEVEL ONLY",
    ):
        assert fragment in result.output, fragment


def test_translate_command(tmp_path, en_srt_path):
    out = tmp_path / "out.ko.srt"
    result = CliRunner().invoke(
        main,
        ["translate", str(en_srt_path), "-m", "dummy", "-s", "en", "-t", "ko",
         "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    lines = load_srt(out)
    assert len(lines) == 18
    assert lines[0].text.startswith("[ko] ")


def test_benchmark_command(tmp_path, en_srt_path, ko_srt_path):
    out_dir = tmp_path / "results"
    result = CliRunner().invoke(
        main,
        ["benchmark", str(en_srt_path), "-m", "dummy", "-s", "en", "-t", "ko",
         "-r", str(ko_srt_path), "--output-dir", str(out_dir)],
    )
    assert result.exit_code == 0, result.output
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report[0]["model_key"] == "dummy"
    assert report[0]["chrf"] is not None
    assert (out_dir / "report.md").exists()


def test_benchmark_rejects_misaligned_reference(tmp_path, en_srt_path):
    bad_ref = tmp_path / "bad.srt"
    bad_ref.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n안녕하세요.\n", encoding="utf-8"
    )
    result = CliRunner().invoke(
        main,
        ["benchmark", str(en_srt_path), "-m", "dummy", "-r", str(bad_ref)],
    )
    assert result.exit_code != 0
    assert "must align 1:1" in result.output
