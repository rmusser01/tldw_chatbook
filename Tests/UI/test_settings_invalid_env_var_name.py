"""A hand-edited `api_key_env_var` that is no variable name does not raise
(review of #3057, R1 #6).

The readiness record rejects such a name, and get_provider_readiness let
the ValueError out: Settings ▸ Providers & Models died while composing its
API key row. It is now the invalid-settings readiness, like a malformed
provider table.
"""

from __future__ import annotations

import pytest
from textual.widgets import Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_connect_rows import _SIZE, _open_providers
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from Tests.UI.test_settings_provider_keyboard_journeys import _settle
from tldw_chatbook.Chat.provider_readiness import get_provider_readiness

_NOT_A_NAME = "MY KEY-1"


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize(
    "table",
    [
        {"api_key_env_var": _NOT_A_NAME},
        {"api_key_env_var": _NOT_A_NAME, "credential_source": "environment"},
    ],
)
def test_an_env_var_name_no_record_can_carry_is_invalid_settings(table):
    readiness = get_provider_readiness(
        "openai", {"api_settings": {"openai": table}}, environ={}
    )

    assert not readiness.ready
    assert readiness.reason == "Invalid provider settings"
    assert readiness.env_var is None
    assert "api_settings.openai" in (readiness.recovery or "")


@pytest.mark.asyncio
@private_profile_test
async def test_settings_composes_the_card_over_an_invalid_env_var_name(
    request, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {"api_key_env_var": _NOT_A_NAME}}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)

        env_var = screen.query_one("#settings-provider-credential-env-var", Input)
        assert env_var.value == _NOT_A_NAME
        verdict = screen.query_one("#settings-provider-readiness", Static)
        assert str(verdict.renderable).startswith("Not ready")
        # The row refresh every edit runs does not raise either.
        screen._update_provider_dynamic_widgets()
        await pilot.pause()
        assert screen.query_one("#settings-provider-key-status", Static).display
