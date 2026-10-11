"""Settings ▸ Providers & Models: the first edit after a deep link keeps the
linked provider (review of #3057, R1 #1 = R2 F2).

A deep link to a provider that is not the saved default shows that provider
without a draft. The first edit used to pin the draft to the SAVED default:
the Provider control renamed itself, "Sign in with" stopped answering, and
Save still wrote the linked provider.
"""

from __future__ import annotations

import pytest
from textual.widgets import Input, Select

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import _active_destination_screen
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_category_sweep import _settle_settings
from Tests.UI.test_settings_configuration_hub import (
    _capture_provider_settings_mutations,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from Tests.UI.test_settings_provider_keyboard_journeys import _revert, _settle
from tldw_chatbook.UI.Screens.settings_screen import SettingsCategoryId

_SIZE = (211, 44)
_FAKE_KEY = "sk-proj-abcdefghijklmnop1234"
_TYPED_KEY = "sk-ant-api03-abcdefghijklmnop1234"
PROVIDERS_MODELS = SettingsCategoryId.PROVIDERS_MODELS


async def _deep_link_to_anthropic_key(host, pilot):
    await _settle_settings(pilot)
    screen = _active_destination_screen(host)
    screen.apply_navigation_context(
        {
            "category": PROVIDERS_MODELS.value,
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "field": "api_key",
        }
    )
    await _settle(host, pilot)
    await pilot.pause()
    return screen


def _app():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {"api_key": _FAKE_KEY},
        "anthropic": {"model": "claude-sonnet-4-6"},
    }
    return app


@pytest.mark.asyncio
@private_profile_test
async def test_the_first_edit_after_a_deep_link_keeps_the_linked_provider(
    request, monkeypatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    mutations = _capture_provider_settings_mutations(monkeypatch)
    app = _app()
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _deep_link_to_anthropic_key(host, pilot)
        control = screen.query_one("#settings-provider-search", Input)
        select = screen.query_one("#settings-provider-value", Select)
        api_key = screen.query_one("#settings-provider-api-key", Input)
        assert control.value == "Anthropic"
        assert select.value == "anthropic"
        assert api_key.has_focus
        assert not screen._category_has_unsaved_changes(PROVIDERS_MODELS)

        await pilot.press("x")
        await pilot.pause()

        # The control, the hidden Select and the draft name one provider.
        assert control.value == "Anthropic"
        assert select.value == "anthropic"
        staged = screen._provider_setting_values_mapping()
        assert (staged["provider"], staged["model"]) == (
            "anthropic",
            "claude-sonnet-4-6",
        )

        # "Sign in with" still answers for the linked provider.
        auth = screen.query_one("#settings-provider-auth-source", Select)
        auth.value = "claude_subscription"
        await pilot.pause()
        assert screen._provider_auth_source_value("anthropic") == (
            "claude_subscription"
        )
        auth.value = "api_key"
        await pilot.pause()
        assert screen._provider_auth_source_value("anthropic") == "api_key"

        api_key.focus()
        await pilot.pause()
        await pilot.press("backspace", *_TYPED_KEY, "escape", "s")
        await _settle(host, pilot)

        assert len(mutations) == 1, mutations
        values, _deleted = mutations[0]
        assert values["chat_defaults"]["provider"] == "anthropic"
        assert values["api_settings.anthropic"]["api_key"] == _TYPED_KEY
        assert "api_settings.openai" not in values
        assert control.value == "Anthropic"


def _form_provider(screen) -> tuple[str, object, object]:
    """The provider as the control, the hidden Select and the form say it."""
    return (
        screen.query_one("#settings-provider-search", Input).value,
        screen.query_one("#settings-provider-value", Select).value,
        screen._provider_display_setting_values()["provider"],
    )


_ANTHROPIC = ("Anthropic", "anthropic", "anthropic")


@pytest.mark.asyncio
@private_profile_test
async def test_a_model_chosen_after_a_deep_link_keeps_the_linked_provider(
    request, monkeypatch
):
    """The model handler dropped the link before it staged: same pin, same
    flip. Choosing the linked model again empties the draft, and the form
    still names the linked provider; a saved choice is what the form holds."""
    from Tests.UI.test_settings_default_model_picker import _CatalogScope, _widgets

    mutations = _capture_provider_settings_mutations(monkeypatch)
    app = _app()
    app.app_config["api_settings"]["anthropic"]["api_key"] = _TYPED_KEY
    app.llm_provider_catalog_scope_service = _CatalogScope(
        "anthropic", catalog=("claude-sonnet-4-6", "claude-opus-4-1")
    )
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _deep_link_to_anthropic_key(host, pilot)
        _picker, field, _results, adapter = _widgets(screen)
        assert adapter.value == "claude-sonnet-4-6"

        async def choose(text: str) -> None:
            field.focus()
            await pilot.pause()
            await pilot.press(*text, "down", "enter")
            await _settle(host, pilot)

        await choose("opus")
        assert adapter.value == "claude-opus-4-1"
        staged = screen._provider_setting_values_mapping()
        assert (staged["provider"], staged["model"]) == ("anthropic", "claude-opus-4-1")
        assert _form_provider(screen) == _ANTHROPIC

        await choose("sonnet")
        assert adapter.value == "claude-sonnet-4-6"
        assert not screen._category_has_unsaved_changes(PROVIDERS_MODELS)
        assert _form_provider(screen) == _ANTHROPIC

        await choose("opus")
        await pilot.press("escape", "s")
        await _settle(host, pilot)
        assert len(mutations) == 1, mutations
        assert mutations[0][0]["chat_defaults"] == {
            "provider": "anthropic",
            "model": "claude-opus-4-1",
        }
        assert not screen._category_has_unsaved_changes(PROVIDERS_MODELS)
        assert screen._provider_display_setting_values()["model"] == "claude-opus-4-1"
        assert _form_provider(screen) == _ANTHROPIC


@pytest.mark.asyncio
@private_profile_test
async def test_revert_after_a_deep_link_returns_the_form_to_the_linked_provider(
    request, monkeypatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _deep_link_to_anthropic_key(host, pilot)
        await pilot.press("x")
        await pilot.pause()
        assert screen._category_has_unsaved_changes(PROVIDERS_MODELS)

        await _revert(host, pilot, discard=True)

        assert not screen._category_has_unsaved_changes(PROVIDERS_MODELS)
        assert _form_provider(screen) == _ANTHROPIC
        assert screen.query_one("#settings-model-value", Input).value == (
            "claude-sonnet-4-6"
        )
        assert screen.query_one("#settings-provider-api-key", Input).value == ""


@pytest.mark.asyncio
@private_profile_test
async def test_the_saved_default_can_be_chosen_from_a_deep_linked_form(request):
    """The list compared the choice with the SAVED provider, so choosing the
    saved default from a form linked elsewhere did nothing."""
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _deep_link_to_anthropic_key(host, pilot)
        control = screen.query_one("#settings-provider-search", Input)
        control.focus()
        await pilot.pause()
        await pilot.press(*"openai", "enter")
        await _settle(host, pilot)

        assert _form_provider(screen) == ("OpenAI", "openai", "openai")
        assert screen.query_one("#settings-model-value", Input).value == "gpt-4.1"
        assert not screen._category_has_unsaved_changes(PROVIDERS_MODELS)
