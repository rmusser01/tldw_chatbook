import pytest


@pytest.fixture(autouse=True)
def _reset():
    from tldw_chatbook.Image_Generation import config as c, adapter_registry as r
    c.reset_image_generation_config_cache()
    r.reset_registry()
    yield
    c.reset_image_generation_config_cache()
    r.reset_registry()


def test_keyring_populated_backend_reports_configured(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c, listing as L
    # enable openrouter; provide its key only via keyring (spec §4.2 step 5 -> is_configured must be True)
    monkeypatch.setattr(c, "_read_image_generation_toml",
                        lambda: {"enabled_backends": ["openrouter"], "default_backend": "openrouter"}, raising=False)
    for var in ("OPENROUTER_API_KEY",):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(c, "_keyring_get", lambda b: "kr" if b == "openrouter" else None, raising=False)
    c.get_image_generation_config(reload=True)
    entries = {e["name"]: e for e in L.list_image_models_for_catalog()}
    assert entries["openrouter"]["is_configured"] is True


def test_disabled_backends_excluded(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c, listing as L
    monkeypatch.setattr(c, "_read_image_generation_toml",
                        lambda: {"enabled_backends": ["swarmui"], "default_backend": "swarmui"}, raising=False)
    c.get_image_generation_config(reload=True)
    names = {e["name"] for e in L.list_image_models_for_catalog()}
    assert "novita" not in names and "swarmui" in names


# --- task-2 (fal/Gemini/Fireworks image backends): _is_*_configured --------
#
# These three backends have no adapter registered yet (adapters land in a
# later task), so they never appear via list_image_models_for_catalog()'s
# registry-driven enumeration here -- exercise the dispatch helpers directly,
# same shape as the sibling _is_openrouter_configured/etc. functions.

def test_is_fal_configured_true_when_key_present(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c, listing as L
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.setattr(c, "_read_image_generation_toml",
                        lambda: {"fal": {"api_key": "fake-fal-key"}}, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert L._is_fal_configured(cfg, True) is True


def test_is_fal_configured_false_when_key_missing(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c, listing as L
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: {}, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert L._is_fal_configured(cfg, True) is False


def test_is_fal_configured_false_when_disabled_even_with_key(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c, listing as L
    monkeypatch.setattr(c, "_read_image_generation_toml",
                        lambda: {"fal": {"api_key": "fake-fal-key"}}, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert L._is_fal_configured(cfg, False) is False


def test_is_gemini_configured_true_when_key_present(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c, listing as L
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(c, "_read_image_generation_toml",
                        lambda: {"gemini": {"api_key": "fake-gemini-key"}}, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert L._is_gemini_configured(cfg, True) is True


def test_is_gemini_configured_false_when_key_missing(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c, listing as L
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: {}, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert L._is_gemini_configured(cfg, True) is False


def test_is_fireworks_configured_true_when_key_present(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c, listing as L
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.setattr(c, "_read_image_generation_toml",
                        lambda: {"fireworks": {"api_key": "fake-fireworks-key"}}, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert L._is_fireworks_configured(cfg, True) is True


def test_is_fireworks_configured_false_when_key_missing(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c, listing as L
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: {}, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert L._is_fireworks_configured(cfg, True) is False
