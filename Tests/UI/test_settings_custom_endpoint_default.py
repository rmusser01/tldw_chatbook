"""Settings opens when the saved default provider is a custom endpoint.

TASK-33002.12: with ``[chat_defaults] provider = "custom-ep:<slug>"`` (the
ADR-146 registry), opening Settings killed the app -- readiness canonicalized
the id to ``custom_ep:<slug>``, which ``ProviderReadiness`` rejects with
``ValueError('Provider key is invalid.')``. Readiness for a registry id now
resolves through the registry, and a dangling slug reads as an honest
not-ready state.
"""

import pytest
from textual.widgets import Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
)
from Tests.UI.test_settings_configuration_hub import (
    _open_settings_category,
    _settle_settings_mount_storm,
)
from tldw_chatbook.Chat.console_session_settings import (
    build_console_settings_readiness,
    build_default_console_session_settings,
)
from tldw_chatbook.Chat.provider_readiness import get_provider_readiness

_ENTRY = {
    "display_name": "GPU box",
    "family": "llama_cpp",
    "base_url": "http://192.168.1.5:8080",
    "models": ["model-a"],
}


def _registry_default_app(provider: str):
    app = _build_test_app()
    app.app_config["custom_endpoints"] = {"gpu-box": dict(_ENTRY)}
    app.app_config["chat_defaults"] = {"provider": provider, "model": "model-a"}
    return app


def _text(screen, selector: str) -> str:
    widget = screen.query_one(selector, Static)
    return str(getattr(widget.renderable, "plain", widget.renderable))


def _all_static_text(screen) -> str:
    return "\n".join(
        str(getattr(w.renderable, "plain", w.renderable)) for w in screen.query(Static)
    )


@pytest.mark.asyncio
@private_profile_test
async def test_settings_opens_on_a_registry_default_and_names_the_entry(request):
    host = DestinationHarness(_registry_default_app("custom-ep:gpu-box"), "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings_mount_storm(pilot)
        screen = _active_destination_screen(host)
        overview = _text(screen, "#settings-overview-configuration")
        assert "Status: Ready" in overview, overview

        await _open_settings_category(pilot, "#settings-category-providers-models")
        await _settle_settings_mount_storm(pilot)
        rendered = _all_static_text(screen)
        assert "GPU box" in rendered, rendered
        assert "custom-ep:gpu-box" not in rendered, rendered


@pytest.mark.asyncio
@private_profile_test
async def test_settings_opens_on_a_dangling_registry_default_honestly(request):
    host = DestinationHarness(_registry_default_app("custom-ep:gone"), "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings_mount_storm(pilot)
        screen = _active_destination_screen(host)
        overview = _text(screen, "#settings-overview-configuration")
        assert "Not ready: Endpoint not found" in overview, overview

        await _open_settings_category(pilot, "#settings-category-providers-models")
        await _settle_settings_mount_storm(pilot)
        rendered = _all_static_text(screen)
        assert "Not ready · endpoint not found" in rendered, rendered


# --- readiness seam (no mount) -------------------------------------------


def _config(**entry_overrides):
    return {"custom_endpoints": {"gpu-box": {**_ENTRY, **entry_overrides}}}


@pytest.mark.parametrize("spelling", ["custom-ep:gpu-box", "custom_ep:gpu_box"])
def test_registry_readiness_is_the_entry_family_readiness(spelling):
    readiness = get_provider_readiness(spelling, _config(), environ={})
    assert readiness.ready is True
    assert readiness.provider == "GPU box"
    assert readiness.provider_key == "llama_cpp"
    assert readiness.api_key is None


def test_registry_readiness_is_ready_when_the_declared_credential_resolves():
    readiness = get_provider_readiness(
        "custom-ep:gpu-box",
        _config(api_key_env="GPU_KEY"),
        environ={"GPU_KEY": "sk-env-entry-key"},
    )
    assert readiness.ready is True


def test_registry_readiness_never_carries_the_family_slot_credential():
    """The ``[api_settings.custom]`` key belongs to the slot, not the entry."""
    config = _config(family="openai_compatible")
    config["api_settings"] = {"custom": {"api_key": "sk-slot-key-not-the-entry"}}
    assert get_provider_readiness("custom", config, environ={}).api_key
    readiness = get_provider_readiness("custom-ep:gpu-box", config, environ={})
    assert readiness.ready is True
    assert readiness.api_key is None
    assert readiness.api_key_source is None


def test_registry_readiness_blocks_an_unresolved_declared_credential():
    readiness = get_provider_readiness(
        "custom-ep:gpu-box", _config(api_key_env="GPU_KEY"), environ={}
    )
    assert readiness.ready is False
    assert readiness.reason == "Missing API key"
    assert "GPU_KEY" in readiness.user_message


def test_dangling_registry_readiness_is_an_honest_not_ready_state():
    readiness = get_provider_readiness("custom-ep:gone", _config(), environ={})
    assert readiness.ready is False
    assert readiness.reason == "Endpoint not found"
    assert readiness.configuration_issue == "endpoint_missing"
    assert "custom-ep:gone" in readiness.user_message


@pytest.mark.parametrize(
    ("provider", "ready", "reason"),
    [
        ("OpenAI", False, "Missing API key"),
        ("llama_cpp", True, "Ready"),
        ("not-a-provider", False, "Unknown provider"),
    ],
)
def test_ordinary_provider_readiness_is_unchanged(provider, ready, reason):
    readiness = get_provider_readiness(provider, _config(), environ={})
    assert (readiness.ready, readiness.reason) == (ready, reason)


@pytest.mark.parametrize(
    ("provider", "blocked"), [("custom-ep:gpu-box", False), ("custom-ep:gone", True)]
)
def test_home_console_readiness_seam_handles_a_registry_default(provider, blocked):
    """Home's readiness (``_home_console_provider_ready``) rides this seam."""
    config = {**_config(), "chat_defaults": {"provider": provider, "model": "model-a"}}
    settings = build_default_console_session_settings(config)
    readiness = build_console_settings_readiness(
        settings, app_config=config, environ={}, background_credentials=False
    )
    assert readiness.native_send_supported is not blocked
