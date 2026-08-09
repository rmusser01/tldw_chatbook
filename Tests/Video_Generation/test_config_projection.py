"""Regression coverage for persisted video settings reaching runtime config."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import tldw_chatbook.config as app_config
from tldw_chatbook.Video_Generation import adapter_registry, config as video_config
from tldw_chatbook.Video_Generation.adapter_registry import get_registry
from tldw_chatbook.Video_Generation.config import (
    get_video_generation_config,
    reset_video_generation_runtime,
)


def _video_toml(workflow: str) -> str:
    return f'''[video_generation]
default_backend = "comfyui"
enabled_backends = ["comfyui"]

[video_generation.comfyui]
base_url = "http://127.0.0.1:18188"
default_workflow = "{workflow}"
timeout_seconds = 321
'''


@contextmanager
def _scratch_video_config(tmp_path: Path, monkeypatch):
    """Point real config/runtime caches at a scratch profile, then restore them."""
    cache_state = {
        "config_cache": app_config._CONFIG_CACHE,
        "config_cache_source": app_config._CONFIG_CACHE_SOURCE,
        "settings_cache": app_config._SETTINGS_CACHE,
        "settings_cache_source": app_config._SETTINGS_CACHE_SOURCE,
        "settings": app_config.settings,
        "config_generation": app_config._CONFIG_GENERATION,
        "video_config_cache": video_config._config_cache,
        "registry": adapter_registry._registry,
    }
    config_path = tmp_path / "video-generation-config.toml"
    config_path.write_text(_video_toml("minimax_h3_t2v_spectrum.json"), encoding="utf-8")
    original_path = os.environ.get("TLDW_CONFIG_PATH")
    try:
        monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
        yield config_path, cache_state
    finally:
        if original_path is None:
            monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
        else:
            monkeypatch.setenv("TLDW_CONFIG_PATH", original_path)
        app_config._CONFIG_CACHE = cache_state["config_cache"]
        app_config._CONFIG_CACHE_SOURCE = cache_state["config_cache_source"]
        app_config._SETTINGS_CACHE = cache_state["settings_cache"]
        app_config._SETTINGS_CACHE_SOURCE = cache_state["settings_cache_source"]
        app_config.settings = cache_state["settings"]
        app_config._CONFIG_GENERATION = cache_state["config_generation"]
        video_config._config_cache = cache_state["video_config_cache"]
        adapter_registry._registry = cache_state["registry"]


def test_persisted_video_settings_project_to_runtime_and_refresh(tmp_path, monkeypatch):
    """A profile's global and nested video tables drive fresh runtime instances."""
    with _scratch_video_config(tmp_path, monkeypatch) as (config_path, cache_state):
        settings = app_config.load_settings()
        assert settings["video_generation"] == {
            "default_backend": "comfyui",
            "enabled_backends": ["comfyui"],
            "comfyui": {
                "base_url": "http://127.0.0.1:18188",
                "default_workflow": "minimax_h3_t2v_spectrum.json",
                "timeout_seconds": 321,
            },
        }

        reset_video_generation_runtime()
        first_config = get_video_generation_config()
        first_registry = get_registry()
        assert first_config.default_backend == "comfyui"
        assert first_config.enabled_backends == ["comfyui"]
        assert first_config.comfyui_base_url == "http://127.0.0.1:18188"
        assert first_config.comfyui_default_workflow == "minimax_h3_t2v_spectrum.json"
        assert first_config.comfyui_timeout_seconds == 321
        assert first_registry.resolve_backend(None) == "comfyui"

        config_path.write_text(_video_toml("minimax_h3_t2v_revised.json"), encoding="utf-8")
        app_config.load_settings(force_reload=True)
        reset_video_generation_runtime()
        second_config = get_video_generation_config()
        second_registry = get_registry()

        assert second_config is not first_config
        assert second_config.comfyui_default_workflow == "minimax_h3_t2v_revised.json"
        assert second_registry is not first_registry
        assert second_registry.resolve_backend(None) == "comfyui"

    assert app_config._CONFIG_CACHE is cache_state["config_cache"]
    assert app_config._CONFIG_CACHE_SOURCE is cache_state["config_cache_source"]
    assert app_config._SETTINGS_CACHE is cache_state["settings_cache"]
    assert app_config._SETTINGS_CACHE_SOURCE is cache_state["settings_cache_source"]
    assert app_config.settings is cache_state["settings"]
    assert app_config._CONFIG_GENERATION == cache_state["config_generation"]
    assert video_config._config_cache is cache_state["video_config_cache"]
    assert adapter_registry._registry is cache_state["registry"]
