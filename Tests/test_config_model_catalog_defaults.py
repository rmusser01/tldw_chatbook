import tomllib

from tldw_chatbook.config import CONFIG_TOML_CONTENT


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
