from contextlib import contextmanager
import tomllib

from tldw_chatbook import config as config_module
from tldw_chatbook.config import API_MODELS_BY_PROVIDER, CONFIG_TOML_CONTENT


@contextmanager
def _temporary_config(tmp_path, monkeypatch, toml_text):
    """Load settings from an isolated scratch config and restore both caches."""
    config_path = tmp_path / "provider-model-defaults.toml"
    config_path.write_text(toml_text, encoding="utf-8")
    original_config_cache = config_module._CONFIG_CACHE
    original_config_cache_source = config_module._CONFIG_CACHE_SOURCE
    original_settings_cache = config_module._SETTINGS_CACHE
    original_settings_cache_source = config_module._SETTINGS_CACHE_SOURCE
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    try:
        yield config_module.load_settings(force_reload=True)
    finally:
        config_module._CONFIG_CACHE = original_config_cache
        config_module._CONFIG_CACHE_SOURCE = original_config_cache_source
        config_module._SETTINGS_CACHE = original_settings_cache
        config_module._SETTINGS_CACHE_SOURCE = original_settings_cache_source


def test_zai_provider_and_settings_defaults_exist():
    parsed = tomllib.loads(CONFIG_TOML_CONTENT)
    assert isinstance(parsed["providers"].get("ZAI"), list)
    zai_settings = parsed["api_settings"]["zai"]
    assert zai_settings["api_key_env_var"] == "ZAI_API_KEY"
    assert zai_settings["api_base_url"] == "https://api.z.ai/api/paas/v4"


def test_model_catalog_defaults_exist():
    parsed = tomllib.loads(CONFIG_TOML_CONTENT)
    section = parsed["model_catalog"]
    assert section["auto_refresh_enabled"] is True
    assert section["stale_after_hours"] == 24
    assert section["auto_refresh_disabled"] == []
    assert section["write_to_config"] == []


def test_bundled_provider_defaults_use_current_models():
    parsed = tomllib.loads(CONFIG_TOML_CONTENT)

    assert parsed["api_settings"]["deepseek"]["model"] == "deepseek-v4-flash"
    assert parsed["api_settings"]["anthropic"]["model"] == "claude-sonnet-5"
    assert parsed["api_settings"]["openai"]["model"] == "gpt-5.6-terra"
    assert parsed["chat_defaults"]["provider"] == "OpenAI"
    assert parsed["chat_defaults"]["model"] == "gpt-5.6-terra"

    model_capabilities = parsed["model_capabilities"]
    assert model_capabilities["models"]["gpt-5.6-terra"] == {
        "vision": True,
        "max_images": 10,
    }
    assert model_capabilities["models"]["claude-sonnet-5"] == {
        "vision": True,
        "max_images": 5,
    }

    providers = parsed["providers"]
    assert providers["DeepSeek"] == ["deepseek-v4-flash", "deepseek-v4-pro"]
    assert providers["Anthropic"][:4] == [
        "claude-sonnet-5",
        "claude-opus-5",
        "claude-fable-5",
        "claude-haiku-4-5",
    ]
    assert providers["OpenAI"][:3] == [
        "gpt-5.6-terra",
        "gpt-5.6-sol",
        "gpt-5.6-luna",
    ]
    assert "deepseek-chat" not in providers["DeepSeek"]
    assert "deepseek-reasoner" not in providers["DeepSeek"]

    for provider in ("DeepSeek", "Anthropic", "OpenAI"):
        assert API_MODELS_BY_PROVIDER[provider] == providers[provider]


def test_load_settings_uses_current_models_when_legacy_api_models_are_omitted(
    tmp_path, monkeypatch
):
    with _temporary_config(tmp_path, monkeypatch, "[API]\n") as settings:
        assert settings["anthropic_api"]["model"] == "claude-sonnet-5"
        assert settings["deepseek_api"]["model"] == "deepseek-v4-flash"
        assert settings["openai_api"]["model"] == "gpt-5.6-terra"


def test_load_settings_preserves_explicit_legacy_api_models(tmp_path, monkeypatch):
    explicit_models = {
        "anthropic_model": "user-anthropic-model",
        "deepseek_model": "user-deepseek-model",
        "openai_model": "user-openai-model",
    }
    config_text = "[API]\n" + "\n".join(
        f'{key} = "{model}"' for key, model in explicit_models.items()
    )

    with _temporary_config(tmp_path, monkeypatch, config_text) as settings:
        assert settings["anthropic_api"]["model"] == explicit_models["anthropic_model"]
        assert settings["deepseek_api"]["model"] == explicit_models["deepseek_model"]
        assert settings["openai_api"]["model"] == explicit_models["openai_model"]
