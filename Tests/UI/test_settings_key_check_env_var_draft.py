"""Settings ▸ Providers & Models ▸ Key check: the verdict reads the credential
Save would keep (review of #3057, R2's card note; a1's report, item 6).

An Env var name typed over a saved key makes Save store the variable and
delete the key. The key 't' probes with already followed that, but the
readiness behind the Key check row still found the stored key: the row said
Ready from a key Save was about to delete, beside an API key row that says
"s removes the saved key". The same gap, from the other two edits: a key
typed over a saved Env var choice was not read, and a cleared key was still
found in the shell that Save was about to stop reading.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_connect_rows import (
    _CLEAR_KEY,
    _FAKE_KEY,
    _SIZE,
    _open_providers,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from Tests.UI.test_settings_provider_keyboard_journeys import _settle

_UNSET = "A2_UNSET_VAR"
_SET = "A2_SET_VAR"


def _verdict(screen) -> str:
    return str(screen.query_one("#settings-provider-readiness", Static).renderable)


async def _source_becomes(pilot, screen, source: str) -> None:
    """Wait for the edit to be staged: Save would then write ``source``."""
    try:
        async with asyncio.timeout(10):
            while screen._provider_current_credential_source("openai") != source:
                await pilot.pause(0.02)
    except TimeoutError as exc:
        raise AssertionError(f"the draft never chose {source!r}") from exc
    await pilot.pause()


@pytest.mark.asyncio
@private_profile_test
async def test_key_check_follows_an_env_var_typed_over_a_saved_key(
    request, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv(_UNSET, raising=False)
    monkeypatch.setenv(_SET, _FAKE_KEY)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {"api_key": _FAKE_KEY}}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        assert _verdict(screen).startswith("Ready"), _verdict(screen)
        env_var = screen.query_one("#settings-provider-credential-env-var", Input)

        env_var.value = _UNSET
        await _source_becomes(pilot, screen, "environment")
        assert screen._provider_current_draft_credential() is None
        assert _verdict(screen).startswith("Not ready"), _verdict(screen)

        env_var.value = _SET
        await _settle(host, pilot)
        assert screen._provider_current_draft_credential() == _FAKE_KEY
        assert _verdict(screen).startswith("Ready"), _verdict(screen)

        # Emptied again, Save keeps the stored key and so does the verdict.
        monkeypatch.delenv(_SET)
        env_var.value = ""
        await _source_becomes(pilot, screen, "stored")
        assert _verdict(screen).startswith("Ready"), _verdict(screen)


async def _open(monkeypatch, table: dict[str, str]):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": table}
    return _SettingsCssHarness(app, "settings")


@pytest.mark.asyncio
@private_profile_test
async def test_key_check_reads_a_key_typed_over_a_saved_env_var_choice(
    request, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv(_UNSET, raising=False)
    host = await _open(
        monkeypatch, {"api_key_env_var": _UNSET, "credential_source": "environment"}
    )

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        assert _verdict(screen).startswith("Not ready"), _verdict(screen)

        screen.query_one("#settings-provider-api-key", Input).value = _FAKE_KEY
        await _source_becomes(pilot, screen, "draft")
        assert _verdict(screen).startswith("Ready"), _verdict(screen)


@pytest.mark.asyncio
@private_profile_test
async def test_key_check_stops_reading_the_shell_once_the_key_is_cleared(
    request, monkeypatch
):
    """Clear with no Env var named saves "no credential": the provider's
    conventional variable is not read after that, so it is not read now."""
    monkeypatch.setenv("OPENAI_API_KEY", _FAKE_KEY)
    host = await _open(monkeypatch, {"api_key": _FAKE_KEY})

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        assert _verdict(screen).startswith("Ready"), _verdict(screen)

        screen.query_one("#settings-provider-api-key", Input).focus()
        await pilot.pause()
        await pilot.press(_CLEAR_KEY)
        await _source_becomes(pilot, screen, "none")
        assert _verdict(screen).startswith("Not ready"), _verdict(screen)
