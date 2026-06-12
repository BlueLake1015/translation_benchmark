from translation_benchmark.models.paths import models_dir_or_default, resolve_model_source


def test_defaults_to_models_directory(monkeypatch):
    monkeypatch.delenv("TB_MODELS_DIR", raising=False)
    assert str(models_dir_or_default(None)) == "models"


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("TB_MODELS_DIR", "/data/weights")
    assert str(models_dir_or_default(None)) == "/data/weights"
    # An explicit argument beats the env var.
    assert str(models_dir_or_default("elsewhere")) == "elsewhere"


def test_local_copy_is_loaded_directly(tmp_path):
    local = tmp_path / "tower-plus-9b"
    local.mkdir()
    (local / "config.json").write_text("{}")
    source, extra = resolve_model_source("Unbabel/Tower-Plus-9B", "tower-plus-9b", tmp_path)
    assert source == str(local)
    assert extra == {}


def test_missing_local_copy_falls_back_to_hub_with_local_cache(tmp_path):
    source, extra = resolve_model_source("Unbabel/Tower-Plus-9B", "tower-plus-9b", tmp_path)
    assert source == "Unbabel/Tower-Plus-9B"
    assert extra == {"cache_dir": str(tmp_path)}


def test_incomplete_local_dir_is_not_used(tmp_path):
    (tmp_path / "tower-plus-9b").mkdir()  # no config.json -> not a model
    source, extra = resolve_model_source("Unbabel/Tower-Plus-9B", "tower-plus-9b", tmp_path)
    assert source == "Unbabel/Tower-Plus-9B"
