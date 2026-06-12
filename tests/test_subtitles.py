from datetime import timedelta

from translation_benchmark.subtitles import (
    flatten,
    load_srt,
    prepare_for_translation,
    save_srt,
    strip_markup,
    translated_copy,
)


def test_load_real_world_srt(en_lines):
    assert len(en_lines) == 18
    assert en_lines[0].index == 1
    assert en_lines[0].start == timedelta(seconds=1)
    assert en_lines[0].end == timedelta(seconds=3, milliseconds=500)
    assert "Rain hammered" in en_lines[0].text


def test_cues_are_ordered_and_non_overlapping(en_lines):
    for prev, cur in zip(en_lines, en_lines[1:]):
        assert prev.end <= cur.start
        assert cur.start < cur.end


def test_en_and_ko_fixtures_align(en_lines, ko_lines):
    assert len(en_lines) == len(ko_lines)
    for en, ko in zip(en_lines, ko_lines):
        assert en.start == ko.start
        assert en.end == ko.end


def test_strip_markup_removes_html_and_ass_tags():
    assert strip_markup("<i>Officially.</i>") == "Officially."
    assert strip_markup("{\\an8}So we have a pattern.") == "So we have a pattern."
    assert strip_markup('<font color="red">hi</font>') == "hi"


def test_flatten_collapses_multiline_cues():
    assert flatten("Traffic. And a body\non the Fifth Street bridge.") == (
        "Traffic. And a body on the Fifth Street bridge."
    )


def test_prepare_for_translation(en_lines):
    text = prepare_for_translation(en_lines[0].text)
    assert text == "Rain hammered the city like it owed money."


def test_save_round_trip(tmp_path, en_lines):
    out = tmp_path / "round_trip.srt"
    save_srt(out, en_lines)
    reloaded = load_srt(out)
    assert [line.text for line in reloaded] == [line.text for line in en_lines]
    assert [line.start for line in reloaded] == [line.start for line in en_lines]


def test_translated_copy_preserves_timing(en_lines):
    translations = [f"번역 {i}" for i in range(len(en_lines))]
    out = translated_copy(en_lines, translations)
    assert [line.text for line in out] == translations
    assert [(line.start, line.end) for line in out] == [
        (line.start, line.end) for line in en_lines
    ]
