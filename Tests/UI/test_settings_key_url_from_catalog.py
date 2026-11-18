"""Settings ▸ Providers & Models: where to get a key comes from the shared
catalog (TASK-34100.6 AC#27, review provider-25; review v3r1 R1-03 /
G6-BV3-02).

The first-run wizard says where to get a key from ``provider_setup_facts``'s
sourced ``key_url`` (or the dashboard fallback for an unconfirmed page).
TASK-33007.2 rebuilt Connect and removed the Settings line that read the same
fact, so only the wizard said it. Connect's rows stay one line each (its AC#7
keeps details in the Inspector's Key block), so the line is restored there,
under the Key heading, read from the same catalog record.

Mounted with the real application stylesheet at 211x44.
"""

from __future__ import annotations

import pytest
from textual.widgets import Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_connect_rows import (
    _SIZE,
    _held,
    _no_provider_key_env,
    _open_providers,
    _text,
    _type_filter,
    _until,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from tldw_chatbook.Chat.provider_setup_facts import key_help_text, provider_setup_facts

_KEY_URL = "#settings-provider-key-url"


@pytest.mark.asyncio
@private_profile_test
async def test_the_inspector_key_block_says_where_to_get_a_key_from_the_catalog(
    request, monkeypatch
):
    _no_provider_key_env(monkeypatch)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {}}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        line = screen.query_one(_KEY_URL, Static)
        assert line.display
        assert _text(screen, _KEY_URL) == key_help_text(
            provider_setup_facts("openai"), "OpenAI"
        )
        assert _text(screen, _KEY_URL) == "New keys: platform.openai.com/api-keys."

        await _type_filter(pilot, screen, "Anthropic")
        await pilot.press("enter")
        await _until(pilot, lambda: _held(screen) == "anthropic")
        assert _text(screen, _KEY_URL) == "New keys: console.anthropic.com."

        await _type_filter(pilot, screen, "Ollama")
        await pilot.press("enter")
        await _until(pilot, lambda: "ollama" in _held(screen))
        assert not line.display, "a provider that needs no key gets no key line"
