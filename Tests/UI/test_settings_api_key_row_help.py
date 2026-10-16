"""Settings ▸ Providers & Models ▸ Connect: what the API key row's help says
beside a saved key (review of PR #3057, R1 #4 and #26; rulings C2 and C5).

Mounted with the real application stylesheet at 211x44 and driven by real
keypresses. The row never names a key that does not work in the field and
never calls an empty field "masked". Its helpers are shared with
test_settings_api_key_row_env_var.py.
"""

from __future__ import annotations

import pytest
from textual.widgets import Input

from Tests.private_profile import private_profile_test
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_connect_rows import (
    _FAKE_KEY,
    _SIZE,
    _open_providers,
    _text,
)
from Tests.UI.test_settings_narrow_layout import _region_rows, _SettingsCssHarness

_KEY_WORD = "#settings-provider-key-status"
_KEY_HELP = "#settings-provider-api-key-help"
_ENV_VAR = "#settings-provider-credential-env-var"
_SAVED_HELP = "hidden · ctrl+l clears the saved key"


def _app(monkeypatch, openai: dict[str, object]):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": openai}
    return app


def _key_row(screen) -> tuple[str, str]:
    return _text(screen, _KEY_WORD), _text(screen, _KEY_HELP)


async def _type_env_var(screen, pilot, name: str) -> None:
    """Replace the Env var field's text with ``name``, key by key."""
    field = screen.query_one(_ENV_VAR, Input)
    field.focus()
    await pilot.pause()
    await pilot.press("end", *["backspace"] * len(field.value), *name)
    await pilot.pause()
    assert field.value == name


@pytest.mark.asyncio
@private_profile_test
async def test_a_saved_key_row_says_hidden_and_names_only_the_clear_key(
    request, monkeypatch
):
    """C2, C5: the saved key is not in the field, so nothing there is
    "masked"; and "t" typed in the field is text, so the row does not offer
    it as the test key."""
    host = _SettingsCssHarness(_app(monkeypatch, {"api_key": _FAKE_KEY}), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        api_key = screen.query_one("#settings-provider-api-key", Input)
        row = screen.query_one("#settings-provider-api-key-row")

        assert api_key.value == ""
        assert _key_row(screen) == ("saved in config", _SAVED_HELP)
        assert _SAVED_HELP in "\n".join(_region_rows(screen, row))

        api_key.focus()
        await pilot.pause()
        await pilot.press("t")
        await pilot.pause()
        assert api_key.value == "t"
        assert _key_row(screen) == ("edited *", "masked · s saves it to config")
