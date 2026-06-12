"""Test and benchmark harness for open-weight MT models on film/video subtitles."""

__version__ = "0.1.0"

from translation_benchmark.models.registry import (  # noqa: F401
    MODELS,
    create_translator,
    get_spec,
    list_specs,
)
