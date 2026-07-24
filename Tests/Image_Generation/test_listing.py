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
