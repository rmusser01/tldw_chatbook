import tomllib

from tldw_chatbook.config import API_MODELS_BY_PROVIDER, CONFIG_TOML_CONTENT


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
