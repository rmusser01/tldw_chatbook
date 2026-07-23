import pytest

@pytest.fixture(autouse=True)
def _reset_cache():
    from tldw_chatbook.Image_Generation import config as c
    c.reset_image_generation_config_cache()
    yield
    c.reset_image_generation_config_cache()

def test_defaults_when_unconfigured(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    # No TOML section, no env, no keyring: fall back to documented defaults.
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: {}, raising=False)
    monkeypatch.setattr(c, "_keyring_get", lambda backend: None, raising=False)  # avoid real keyring
    for var in ("OPENROUTER_API_KEY", "NOVITA_API_KEY", "TOGETHER_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert cfg.swarmui_base_url == c.DEFAULT_SWARMUI_BASE_URL
    assert cfg.max_width == c.DEFAULT_MAX_WIDTH
    assert cfg.openrouter_image_api_key in (None, "")  # unconfigured

def test_nested_toml_flattens_to_flat_fields(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    fake = {
        "default_backend": "swarmui",
        "enabled_backends": ["swarmui", "openrouter"],
        "swarmui": {"base_url": "http://example:9999"},
        "openrouter": {"default_model": "openai/gpt-image-1", "timeout_seconds": 42},
    }
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: fake, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert cfg.swarmui_base_url == "http://example:9999"
    assert cfg.openrouter_image_default_model == "openai/gpt-image-1"
    assert cfg.openrouter_image_timeout_seconds == 42
    assert cfg.enabled_backends == ["swarmui", "openrouter"]

def test_secret_precedence_env_over_config(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    fake = {"openrouter": {"api_key": "from-config"}}
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: fake, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    cfg = c.get_image_generation_config(reload=True)
    assert cfg.openrouter_image_api_key == "from-env"

def test_secret_from_keyring_populates_field(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: {}, raising=False)
    monkeypatch.delenv("NOVITA_API_KEY", raising=False)
    # keyring-only secret must land on the config field so listing.is_configured sees it (spec §4.2 step 5)
    monkeypatch.setattr(c, "_keyring_get", lambda backend: "kr-secret" if backend == "novita" else None, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert cfg.novita_image_api_key == "kr-secret"
