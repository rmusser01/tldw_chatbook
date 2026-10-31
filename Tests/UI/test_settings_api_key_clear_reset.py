"""Settings ▸ Providers & Models: a dropped Clear does not come back (review
of #3057, R2 F1 = R1 #12).

Clear (ctrl+l) on the API key field stages the removal of a saved key and
remembers which draft and provider it was staged for, so that emptying the
field by hand is no edit. That memory was never dropped: after a provider
round trip, typing a character and deleting it staged the removal again and
Save deleted the key.
"""

from __future__ import annotations

import pytest
from textual.widgets import Input

from Tests.private_profile import private_profile_test
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_configuration_hub import (
    _capture_provider_settings_mutations,
)
from Tests.UI.test_settings_connect_rows import (
    _CLEAR_KEY,
    _FAKE_KEY,
    _SIZE,
    _open_providers,
    _text,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from Tests.UI.test_settings_provider_keyboard_journeys import _revert, _settle
from tldw_chatbook.UI.Screens.settings_screen import SettingsCategoryId

PROVIDERS_MODELS = SettingsCategoryId.PROVIDERS_MODELS
_ANTHROPIC_KEY = "sk-ant-api03-abcdefghijklmnop1234"


def _app():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "groq", "model": "llama-3.3"}
    app.app_config["api_settings"] = {
        "groq": {"api_key": _FAKE_KEY},
        "anthropic": {"api_key": _ANTHROPIC_KEY, "model": "claude-sonnet-4-6"},
        "openai": {"api_key": _FAKE_KEY, "model": "gpt-4.1"},
    }
    return app


async def _choose_provider(host, pilot, screen, name: str) -> None:
    screen.query_one("#settings-provider-search", Input).focus()
    await pilot.pause()
    await pilot.press(*name, "enter")
    await _settle(host, pilot)


async def _clear_the_key(pilot, screen) -> None:
    screen.query_one("#settings-provider-api-key", Input).focus()
    await pilot.pause()
    await pilot.press(_CLEAR_KEY)
    await pilot.pause()
    assert _text(screen, "#settings-provider-key-status") == "cleared *"


async def _type_and_delete_a_character(pilot, screen) -> None:
    screen.query_one("#settings-provider-api-key", Input).focus()
    await pilot.pause()
    await pilot.press("x", "backspace")
    await pilot.pause()


def _key_is_not_staged_for_removal(screen) -> None:
    draft = screen._settings_drafts.get(PROVIDERS_MODELS)
    assert draft is None or "api_key" not in draft.values
    assert _text(screen, "#settings-provider-key-status") == "saved in config"


@pytest.mark.asyncio
@private_profile_test
async def test_a_clear_dropped_by_a_provider_round_trip_does_not_come_back(
    request, monkeypatch
):
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    mutations = _capture_provider_settings_mutations(monkeypatch)
    app = _app()
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _choose_provider(host, pilot, screen, "anthropic")
        await _clear_the_key(pilot, screen)

        await _choose_provider(host, pilot, screen, "openai")
        await _choose_provider(host, pilot, screen, "anthropic")
        _key_is_not_staged_for_removal(screen)

        await _type_and_delete_a_character(pilot, screen)
        _key_is_not_staged_for_removal(screen)

        await pilot.press("escape", "s")
        await _settle(host, pilot)
        assert mutations, "the provider change is saved"
        assert all(
            "api_key" not in deleted.get("api_settings.anthropic", ())
            for _values, deleted in mutations
        ), mutations
        assert app.app_config["api_settings"]["anthropic"]["api_key"] == (
            _ANTHROPIC_KEY
        )


@pytest.mark.asyncio
@private_profile_test
async def test_a_reverted_clear_does_not_come_back(request, monkeypatch):
    """Revert drops the draft; an equal draft made later is a new one, but
    the memory is dropped with it rather than left to an identity check."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _clear_the_key(pilot, screen)
        assert screen._provider_api_key_cleared is not None

        await _revert(host, pilot, discard=True)
        assert screen._provider_api_key_cleared is None

        await _type_and_delete_a_character(pilot, screen)
        _key_is_not_staged_for_removal(screen)


@pytest.mark.asyncio
@private_profile_test
async def test_a_saved_clear_is_forgotten(request, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    _capture_provider_settings_mutations(monkeypatch)
    app = _app()
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _clear_the_key(pilot, screen)
        await pilot.press("escape", "s")
        await _settle(host, pilot)

        assert "api_key" not in app.app_config["api_settings"]["groq"]
        assert screen._provider_api_key_cleared is None
