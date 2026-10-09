"""Settings > Providers > Anthropic: sign in with an API key or the Claude subscription.

TASK-34201, owner-approved design (2026-10-03): a "Sign in with" select at the
top of Credentials (since TASK-33007.2, directly above the API key row in
Connect), Anthropic only; choosing the subscription disables, but
keeps visible, the API key and Env var rows; a guidance line, no confirm.
Saves go through the captured atomic writer, never a real config file.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, Select, Static

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
)
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_configuration_hub import _open_settings_category
from Tests.UI.test_settings_qwencloud_api_mode import _capture_atomic_writes
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Settings_Modules.providers_models_card import (
    AUTH_SOURCE_API_KEY_HELP,
    AUTH_SOURCE_SUBSCRIPTION_HELP,
)

# The real Settings screen goes through config-participant admission, which the
# per-test sandbox refuses (RecoveryRequired raw_source_selection_changed, seen in
# CI's UI Fast Lane); keep the collection-time private profile, as
# test_console_fork_fresh_lineage_flow.py does. Saves stay captured below.
pytestmark = pytest.mark.bootstrap_profile

_DRAFT_KEY = "provider_auth_source:anthropic"


def _anthropic_app(*, auth_source: str | None = None):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "Anthropic", "model": "claude-sonnet-4-6"}
    anthropic: dict[str, object] = {
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "model": "claude-sonnet-4-6",
    }
    if auth_source is not None:
        anthropic["auth_source"] = auth_source
    app.app_config["api_settings"] = {"anthropic": anthropic}
    return app


def _choose(screen, value: str) -> None:
    selector = screen.query_one("#settings-provider-auth-source", Select)
    selector.value = value
    screen.handle_provider_auth_source_changed(Select.Changed(selector, value))


def _switch_provider(screen, provider_key: str) -> None:
    provider = screen.query_one("#settings-provider-value", Select)
    provider.value = provider_key
    screen.handle_provider_value_changed(Select.Changed(provider, provider_key))


@pytest.mark.asyncio
async def test_only_anthropic_shows_the_sign_in_select():
    app = _anthropic_app()
    app.app_config["api_settings"]["openai"] = {"api_base_url": "https://api.openai.com/v1"}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        selector = screen.query_one("#settings-provider-auth-source", Select)
        row = screen.query_one("#settings-provider-auth-source-row")

        assert selector.value == "api_key"
        assert selector.disabled is False
        assert row.has_class("settings-gated-profile-hidden") is False
        assert {
            (str(label), str(value))
            for label, value in selector._options
            if value is not Select.NULL
        } == {("API key", "api_key"), ("Claude subscription", "claude_subscription")}
        assert "Sign in with" in str(row.query_one(".settings-input-label", Static).content)

        _switch_provider(screen, "openai")
        await pilot.pause()

        assert selector.disabled is True
        assert row.has_class("settings-gated-profile-hidden") is True


@pytest.mark.asyncio
async def test_subscription_disables_but_keeps_the_key_rows_and_switching_back_restores_them():
    host = DestinationHarness(_anthropic_app(), "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        api_key = screen.query_one("#settings-provider-api-key", Input)
        env_var = screen.query_one("#settings-provider-credential-env-var", Input)
        clear = screen.query_one("#settings-provider-api-key-clear", Button)
        # Rewritten on purpose (TASK-33007 capture fix 7): the guidance is the
        # Sign in with row's one-line help; its long copy is the Inspector's.
        guidance = screen.query_one("#settings-provider-auth-source-guidance", Static)
        assert str(guidance.content) == AUTH_SOURCE_API_KEY_HELP

        _choose(screen, "claude_subscription")
        await pilot.pause()

        assert api_key.disabled is True and env_var.disabled is True and clear.disabled is True
        assert api_key.display and env_var.display  # visible, not hidden
        assert env_var.value == "ANTHROPIC_API_KEY"  # nothing lost
        assert str(guidance.content) == AUTH_SOURCE_SUBSCRIPTION_HELP

        _choose(screen, "api_key")
        await pilot.pause()

        assert api_key.disabled is False and env_var.disabled is False
        assert env_var.value == "ANTHROPIC_API_KEY"
        assert str(guidance.content) == AUTH_SOURCE_API_KEY_HELP


@pytest.mark.asyncio
async def test_saving_the_subscription_writes_only_auth_source(monkeypatch):
    app = _anthropic_app()
    calls = _capture_atomic_writes(monkeypatch)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        _choose(screen, "claude_subscription")
        await pilot.click("#settings-save-category")
        await pilot.pause()

        assert calls == [({"api_settings.anthropic": {"auth_source": "claude_subscription"}}, {})]
        assert app.app_config["api_settings"]["anthropic"]["auth_source"] == "claude_subscription"
        draft = screen._settings_drafts.get(SettingsCategoryId.PROVIDERS_MODELS)
        assert draft is None or _DRAFT_KEY not in draft.values


@pytest.mark.asyncio
async def test_revert_discards_an_unsaved_choice():
    app = _anthropic_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        selector = screen.query_one("#settings-provider-auth-source", Select)
        _choose(screen, "claude_subscription")
        assert screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS].values[_DRAFT_KEY] == (
            "claude_subscription"
        )

        await pilot.click("#settings-revert-category")
        await pilot.pause()
        await pilot.click("#confirm-button")
        await pilot.pause()

        assert selector.value == "api_key"
        assert SettingsCategoryId.PROVIDERS_MODELS not in screen._settings_drafts
        assert "auth_source" not in app.app_config["api_settings"]["anthropic"]


@pytest.mark.asyncio
async def test_an_unsaved_choice_survives_a_provider_switch():
    app = _anthropic_app()
    app.app_config["api_settings"]["openai"] = {"api_base_url": "https://api.openai.com/v1"}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        selector = screen.query_one("#settings-provider-auth-source", Select)

        # Switch before the select's Changed event runs: the switch snapshots it.
        selector.value = "claude_subscription"
        _switch_provider(screen, "openai")
        await pilot.pause()
        assert screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS].values[_DRAFT_KEY] == (
            "claude_subscription"
        )

        _switch_provider(screen, "anthropic")
        await pilot.pause()
        assert selector.value == "claude_subscription"
        assert screen.query_one("#settings-provider-api-key", Input).disabled is True


@pytest.mark.asyncio
async def test_a_saved_subscription_loads_and_readiness_previews_a_draft():
    app = _anthropic_app(auth_source="claude_subscription")
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        selector = screen.query_one("#settings-provider-auth-source", Select)
        assert selector.value == "claude_subscription"
        assert screen.query_one("#settings-provider-api-key", Input).disabled is True

        _choose(screen, "api_key")
        staged = screen._provider_test_staged_config("Anthropic")
        assert staged["api_settings"]["anthropic"]["auth_source"] == "api_key"
        assert app.app_config["api_settings"]["anthropic"]["auth_source"] == "claude_subscription"


@pytest.mark.asyncio
async def test_an_untouched_select_creates_no_draft():
    """With the default left alone, nothing about today's behavior changes."""
    host = DestinationHarness(_anthropic_app(), "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        draft = screen._settings_drafts.get(SettingsCategoryId.PROVIDERS_MODELS)
        assert draft is None or _DRAFT_KEY not in draft.values


def test_the_readiness_overlay_carries_a_draft_sign_in_choice():
    """Pure: the staged-config overlay writes a draft choice and leaves the source alone."""
    from tldw_chatbook.UI.Screens.settings_screen import overlay_provider_draft_config

    saved = {"api_settings": {"anthropic": {"model": "claude-sonnet-4-6"}}}
    staged = overlay_provider_draft_config(
        saved,
        provider_save_key="anthropic",
        endpoint_key="api_base_url",
        draft_endpoint=None,
        draft_env_var=None,
        draft_api_key=None,
        draft_auth_source="claude_subscription",
    )
    assert staged["api_settings"]["anthropic"]["auth_source"] == "claude_subscription"
    assert "auth_source" not in saved["api_settings"]["anthropic"]
    untouched = overlay_provider_draft_config(
        saved,
        provider_save_key="anthropic",
        endpoint_key="api_base_url",
        draft_endpoint=None,
        draft_env_var=None,
        draft_api_key=None,
    )
    assert "auth_source" not in untouched["api_settings"]["anthropic"]


@pytest.mark.asyncio
async def test_switching_a_saved_subscription_back_to_api_key_shows_the_stored_key():
    """Qodo #2990: a saved subscription choice must not hide a stored key."""
    app = _anthropic_app(auth_source="claude_subscription")
    app.app_config["api_settings"]["anthropic"]["api_key"] = "sk-ant-test-0000000000000000"
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        clear = screen.query_one("#settings-provider-api-key-clear", Button)
        assert clear.disabled is True  # subscription chosen

        _choose(screen, "api_key")
        await pilot.pause()

        assert clear.disabled is False
        # TASK-33007.2 folded the credential status line into the API key
        # row: its Source word and help line say where the key comes from.
        source = str(screen.query_one("#settings-provider-key-status", Static).content)
        help_line = str(screen.query_one("#settings-provider-api-key-help", Static).content)
        assert "Claude subscription" not in help_line
        assert source == "saved in config"


@pytest.mark.asyncio
async def test_an_unsaved_subscription_choice_shows_in_the_credential_status():
    """Qodo #2990: the status row follows the choice before Save."""
    host = DestinationHarness(_anthropic_app(), "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        _choose(screen, "claude_subscription")
        await pilot.pause()

        source = str(screen.query_one("#settings-provider-key-status", Static).content)
        help_line = str(screen.query_one("#settings-provider-api-key-help", Static).content)
        assert source == "subscription"
        assert "Claude subscription" in help_line
