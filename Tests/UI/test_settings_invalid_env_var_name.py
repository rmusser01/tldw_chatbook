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
from Tests.UI.test_settings_api_key_row_help import _key_row
from Tests.UI.test_settings_connect_rows import _SIZE, _open_providers
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from Tests.UI.test_settings_provider_keyboard_journeys import _settle
from tldw_chatbook.Chat.provider_readiness import get_provider_readiness
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

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


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    ("saved", "typed", "before", "after"),
    [
        # A saved name that is no name, corrected in the field right below:
        # the row said "invalid" until Save, the Key check row already not.
        (
            _NOT_A_NAME,
            "MY_KEY_1",
            ("invalid", "repair it in Advanced Config"),
            ("missing", "paste one, or set MY_KEY_1"),
        ),
        # A typo made in this form: "or set MY KEY-1" named nothing settable.
        (
            "",
            _NOT_A_NAME,
            ("missing", "paste one, or set OPENAI_API_KEY"),
            ("invalid", "Env var is not a variable's name"),
        ),
    ],
)
async def test_the_api_key_row_follows_the_env_var_name_being_typed(
    request, monkeypatch, saved, typed, before, after
):
    """Fix round 1, finding 4: the row read its verdict from the SAVED name
    and printed a typed one without asking whether it is a name."""
    for name in ("OPENAI_API_KEY", "MY_KEY_1"):
        monkeypatch.delenv(name, raising=False)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {"api_key_env_var": saved} if saved else {}
    }
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        assert _key_row(screen) == before

        screen.query_one("#settings-provider-credential-env-var", Input).value = typed
        await pilot.pause()
        await pilot.pause()

        assert _key_row(screen) == after
        verdict = screen.query_one("#settings-provider-readiness", Static)
        assert str(verdict.renderable).startswith("Not ready")


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize(
    ("env_var_invalid", "next_step"),
    [
        (False, "fix this provider's settings in Advanced Config"),
        # Finding 4(b): a typo made in this form is not Advanced Config's.
        (True, "check the Env var field"),
    ],
)
def test_the_test_result_sends_an_env_var_typo_to_its_own_field(
    env_var_invalid, next_step
):
    readiness = get_provider_readiness(
        "openai", {"api_settings": {"openai": {"api_key_env_var": _NOT_A_NAME}}}
    )

    rows = dict(
        SettingsScreen._provider_test_rows(
            readiness,
            display_name="OpenAI",
            model="gpt-4.1",
            endpoint="",
            dirty=frozenset({"credential_env_var"}),
            env_var_invalid=env_var_invalid,
        )
    )

    assert rows["Config"] == (
        f"OpenAI is not ready: Invalid provider settings — {next_step}"
    )


@pytest.mark.asyncio
@private_profile_test
async def test_a_valid_name_typed_over_a_malformed_table_repairs_nothing(
    request, monkeypatch
):
    """Fix round 2, findings 5 and 10: while Env var was edited, the API key
    row dropped "invalid" and the Test result blamed the Env var field,
    whatever made the settings invalid. A second [api_settings.Moonshot]
    table is not the name's fault, and Save leaves it in place."""
    monkeypatch.delenv("MOONSHOT_KEY", raising=False)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "moonshot", "model": "kimi-k3"}
    app.app_config["api_settings"] = {"moonshot": {}, "Moonshot": {}}
    host = _SettingsCssHarness(app, "settings")
    invalid = ("invalid", "repair it in Advanced Config")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        assert _key_row(screen) == invalid

        env_var = screen.query_one("#settings-provider-credential-env-var", Input)
        env_var.value = "MOONSHOT_KEY"
        await pilot.pause()
        await pilot.pause()

        assert _key_row(screen) == invalid
        rows, _toast, _passed = screen._provider_readiness_test_report()
        assert (
            "Invalid provider settings — fix this provider's settings in "
            "Advanced Config"
        ) in rows
