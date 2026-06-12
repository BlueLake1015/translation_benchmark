from pathlib import Path

import pytest

from translation_benchmark.subtitles import load_srt

DATA_DIR = Path(__file__).parent / "data"

# The whole suite targets English -> Korean.
SRC_LANG = "en"
TGT_LANG = "ko"


@pytest.fixture
def en_srt_path() -> Path:
    return DATA_DIR / "night_shift.en.srt"


@pytest.fixture
def ko_srt_path() -> Path:
    return DATA_DIR / "night_shift.ko.srt"


@pytest.fixture
def en_lines(en_srt_path):
    return load_srt(en_srt_path)


@pytest.fixture
def ko_lines(ko_srt_path):
    return load_srt(ko_srt_path)
