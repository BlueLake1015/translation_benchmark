"""Local model storage.

All model weights live under the project-local ``models/`` directory by
default (override with ``--models-dir`` or the ``TB_MODELS_DIR`` env var):

- ``models/<model-key>/`` containing a ``config.json`` is treated as a fully
  downloaded local copy and loaded directly, e.g.::

      huggingface-cli download Unbabel/Tower-Plus-9B --local-dir models/tower-plus-9b

- otherwise the Hugging Face repo id is used with ``models/`` as the
  download cache, so weights still end up under ``models/``.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MODELS_DIR = "models"


def models_dir_or_default(models_dir: str | os.PathLike | None) -> Path:
    return Path(models_dir or os.environ.get("TB_MODELS_DIR") or DEFAULT_MODELS_DIR)


def resolve_model_source(
    hf_id: str, key: str, models_dir: str | os.PathLike | None = None
) -> tuple[str, dict]:
    """Where to load a model from.

    Returns ``(path_or_repo_id, extra_from_pretrained_kwargs)``: the local
    ``<models_dir>/<key>`` directory if it holds a downloaded model, else the
    repo id with ``cache_dir`` pointed at ``models_dir``.
    """
    base = models_dir_or_default(models_dir)
    local = base / key
    if (local / "config.json").is_file():
        return str(local), {}
    return hf_id, {"cache_dir": str(base)}
