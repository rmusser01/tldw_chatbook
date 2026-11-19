"""Settings ▸ Providers & Models: Revert after a provider switch shows the
saved provider's Model defaults rows (review of #3057, R1 #5).

Revert set each row's value by hand and never said which rows the saved
provider has, so the discarded provider's rows stayed shown or hidden.
"""

from __future__ import annotations

import pytest
from textual.widgets import Input

from Tests.private_profile import private_profile_test
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_connect_rows import _FAKE_KEY, _SIZE, _open_providers
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from Tests.UI.test_settings_provider_keyboard_journeys import _revert, _settle
from tldw_chatbook.UI.Screens.settings_screen import (
    PROVIDER_MODEL_PROFILE_FIELD_KEYS,
    SettingsCategoryId,
)

_HIDDEN = "settings-gated-profile-hidden"


def _hidden_rows(screen) -> set[str]:
    hidden = set()
    for draft_key in PROVIDER_MODEL_PROFILE_FIELD_KEYS:
        row = screen.query_one(f"#settings-{draft_key.replace('_', '-')}-row")
        if row.has_class(_HIDDEN):
            hidden.add(draft_key)
    return hidden


@pytest.mark.asyncio
@private_profile_test
async def test_revert_after_a_provider_switch_shows_the_saved_providers_rows(
    request, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {"api_key": _FAKE_KEY},
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "local-model"},
    }
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        saved_rows = _hidden_rows(screen)

        screen.query_one("#settings-provider-search", Input).focus()
        await pilot.pause()
        await pilot.press(*"llama.cpp", "enter")
        await _settle(host, pilot)
        assert screen._provider_setting_values_mapping()["provider"] == "llama_cpp"
        assert _hidden_rows(screen) != saved_rows, "pick providers whose rows differ"

        await _revert(host, pilot, discard=True)

        assert not screen._category_has_unsaved_changes(
            SettingsCategoryId.PROVIDERS_MODELS
        )
        assert screen._provider_setting_values_mapping()["provider"] == "openai"
        assert _hidden_rows(screen) == saved_rows
