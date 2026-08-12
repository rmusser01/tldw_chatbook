import threading

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


# --- task-2 (fal/Gemini image backends): _is_*_configured ------------------
# Fireworks was dropped 2026-07-26 -- vendor deprecated image generation
# (see the design spec/plan docs' 2026-07-26 decision notes).
#
# These backends have no adapter registered yet (adapters land in a later
# task), so they never appear via list_image_models_for_catalog()'s
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


def test_comfyui_listing_is_local_only_and_requires_enabled_valid_resource(
    monkeypatch,
):
    import httpx

    from tldw_chatbook.Image_Generation import config as c, listing as L

    monkeypatch.setattr(
        c,
        "_read_image_generation_toml",
        lambda: {
            "enabled_backends": ["comfyui"],
            "comfyui": {"base_url": "http://127.0.0.1:8188"},
        },
    )
    monkeypatch.setattr(
        L, "_comfyui_workflow_resource_available", lambda: True, raising=False
    )
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("catalog listing must not construct a network client")
        ),
    )

    entries = {entry["name"]: entry for entry in L.list_image_models_for_catalog()}

    assert entries["comfyui"]["is_configured"] is True


def test_comfyui_listing_reports_unconfigured_without_packaged_resource(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c, listing as L

    monkeypatch.setattr(
        c,
        "_read_image_generation_toml",
        lambda: {
            "enabled_backends": ["comfyui"],
            "comfyui": {"base_url": "http://127.0.0.1:8188"},
        },
    )
    monkeypatch.setattr(
        L, "_comfyui_workflow_resource_available", lambda: False, raising=False
    )

    entries = {entry["name"]: entry for entry in L.list_image_models_for_catalog()}

    assert entries["comfyui"]["is_configured"] is False


def test_comfyui_listing_excludes_disabled_backend(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c, listing as L

    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: {})

    assert "comfyui" not in {
        entry["name"] for entry in L.list_image_models_for_catalog()
    }


def test_listing_resolves_formats_from_its_captured_registry(monkeypatch):
    from tldw_chatbook.Image_Generation import adapter_registry as registry
    from tldw_chatbook.Image_Generation import config as image_config
    from tldw_chatbook.Image_Generation import listing

    names_entered = threading.Event()
    release_names = threading.Event()
    entries = []
    errors = []

    class AdapterA:
        supported_formats = {"a-format"}

    class AdapterB:
        supported_formats = {"b-format"}

    monkeypatch.setattr(
        image_config,
        "_read_image_generation_toml",
        lambda: {
            "default_backend": "dynamic",
            "enabled_backends": ["dynamic"],
        },
    )
    image_config.reset_image_generation_runtime()
    registry_a = registry.get_registry()
    registry_a.register_adapter("dynamic", AdapterA)
    real_list_names = registry_a.list_backend_names

    def blocked_list_names(*, include_disabled=False):
        names_entered.set()
        assert release_names.wait(5)
        return real_list_names(include_disabled=include_disabled)

    monkeypatch.setattr(registry_a, "list_backend_names", blocked_list_names)

    def run_listing():
        try:
            entries.extend(listing.list_image_models_for_catalog())
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    listing_thread = threading.Thread(target=run_listing, daemon=True)
    listing_thread.start()
    assert names_entered.wait(5)

    image_config.reset_image_generation_runtime()
    registry_b = registry.get_registry()
    registry_b.register_adapter("dynamic", AdapterB)
    release_names.set()
    listing_thread.join(5)

    assert not listing_thread.is_alive()
    assert errors == []
    assert entries[0]["name"] == "dynamic"
    assert entries[0]["supported_formats"] == ["a-format"]
    image_config.reset_image_generation_runtime()
