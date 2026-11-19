"""Settings ▸ Providers & Models ▸ Connect: the API key row follows an unsaved
Env var edit, because Save does (review of PR #3057, R1 #10; ruling C4).

Mounted with the real application stylesheet at 211x44 and driven by real
keypresses.
"""

from __future__ import annotations

import pytest
from textual.widgets import Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_settings_api_key_row_help import (
    _SAVED_HELP,
    _app,
    _key_row,
    _type_env_var,
)
from Tests.UI.test_settings_connect_rows import _FAKE_KEY, _SIZE, _open_providers
from Tests.UI.test_settings_narrow_layout import _region_rows, _SettingsCssHarness


@pytest.mark.asyncio
@private_profile_test
async def test_an_env_var_typed_over_a_saved_key_says_save_removes_the_key(
    request, monkeypatch
):
    """R1 #10: Save follows a typed Env var and deletes the stored key
    (ADR-012's explicit env-var choice); the row kept saying "saved in
    config" until it was gone."""
    host = _SettingsCssHarness(_app(monkeypatch, {"api_key": _FAKE_KEY}), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        assert _key_row(screen) == ("saved in config", _SAVED_HELP)

        await _type_env_var(screen, pilot, "A1_TYPED_VAR")

        assert screen._provider_current_credential_source("openai") == "environment"
        removes = "s removes the saved key · Env var used"
        assert _key_row(screen) == ("cleared *", removes)
        row = screen.query_one("#settings-provider-api-key-row")
        assert removes in "\n".join(_region_rows(screen, row))

        await _type_env_var(screen, pilot, "")
        assert _key_row(screen) == ("saved in config", _SAVED_HELP)


@pytest.mark.asyncio
@private_profile_test
async def test_the_api_key_row_reads_the_env_var_being_typed(request, monkeypatch):
    """R1 #10: with no stored key the row named the saved variable while
    another was typed. C4: a key pasted into Env var by mistake is not
    repeated in this row's help."""
    monkeypatch.setenv("A1_TYPED_VAR", _FAKE_KEY)
    monkeypatch.delenv("A1_TYPED_VARX", raising=False)
    monkeypatch.delenv("A1_SAVED_VAR", raising=False)
    host = _SettingsCssHarness(
        _app(monkeypatch, {"api_key_env_var": "A1_SAVED_VAR"}), "settings"
    )

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        assert _key_row(screen) == ("missing", "paste one, or set A1_SAVED_VAR")

        await _type_env_var(screen, pilot, "A1_TYPED_VAR")
        assert _key_row(screen) == ("from env var", "A1_TYPED_VAR in your shell")

        await pilot.press("X")
        await pilot.pause()
        assert _key_row(screen) == ("missing", "paste one, or set A1_TYPED_VARX")

        pasted = "sk_live_" + "a1B2c3D4" * 3
        await _type_env_var(screen, pilot, pasted)
        assert _key_row(screen) == ("missing", "paste one, or set that variable")
        shown = "\n".join(str(static.renderable) for static in screen.query(Static))
        assert pasted not in shown
