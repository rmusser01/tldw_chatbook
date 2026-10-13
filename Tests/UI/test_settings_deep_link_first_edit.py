"""Settings ▸ Providers & Models: the first edit after a deep link keeps the
linked provider (review of #3057, R1 #1 = R2 F2).

A deep link to a provider that is not the saved default shows that provider
without a draft. The first edit used to pin the draft to the SAVED default:
the Provider control renamed itself, "Sign in with" stopped answering, and
Save still wrote the linked provider.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, Select, Static

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


async def _deep_link_to_anthropic_key(
    host, pilot, provider="anthropic", model="claude-sonnet-4-6"
):
    await _settle_settings(pilot)
    screen = _active_destination_screen(host)
    screen.apply_navigation_context(
        {
            "category": PROVIDERS_MODELS.value,
            "provider": provider,
            "model": model,
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


def _staged_pair(screen) -> tuple[object, object]:
    staged = screen._provider_setting_values_mapping()
    return staged["provider"], staged["model"]


_LINKED_PAIR = ("anthropic", "claude-sonnet-4-6")


@pytest.mark.asyncio
@private_profile_test
async def test_sign_in_with_as_the_first_edit_after_a_deep_link_is_staged(
    request, monkeypatch
):
    """Fix round 1, finding 1: the handler read the SAVED default's provider
    while the link had no draft, returned early and staged nothing; the next
    paint put "API key" back in the Select."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _deep_link_to_anthropic_key(host, pilot)
        # Finding 11(b): before any edit the Inspector's guide is the linked
        # provider's, not the saved default's.
        guide = dict(screen._provider_field_guidance_rows())
        assert guide["Saved as"] == "api_settings.anthropic.api_key", guide
        auth = screen.query_one("#settings-provider-auth-source", Select)

        auth.value = "claude_subscription"
        await pilot.pause()

        assert screen._provider_auth_source_value("anthropic") == (
            "claude_subscription"
        )
        assert screen._category_has_unsaved_changes(PROVIDERS_MODELS)
        assert _staged_pair(screen) == _LINKED_PAIR
        screen._update_provider_dynamic_widgets()  # the next paint keeps it
        await pilot.pause()
        assert auth.value == "claude_subscription"
        assert _form_provider(screen) == _ANTHROPIC


@pytest.mark.asyncio
@private_profile_test
async def test_api_mode_as_the_first_edit_after_a_deep_link_is_staged(request):
    app = _app()
    app.app_config["api_settings"]["qwencloud"] = {"model": "qwen-max"}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _deep_link_to_anthropic_key(host, pilot, "qwencloud", "qwen-max")
        mode = screen.query_one("#settings-provider-api-mode", Select)
        assert mode.value == "responses"

        mode.value = "chat_completions"
        await pilot.pause()

        assert screen._provider_api_mode_value("qwencloud") == "chat_completions"
        assert _staged_pair(screen) == ("qwencloud", "qwen-max")


@pytest.mark.asyncio
@private_profile_test
async def test_reset_as_the_first_edit_after_a_deep_link_resets_the_linked_pair(
    request,
):
    """Finding 11(a): Reset read the default pair's state, which has no
    override, and did nothing."""
    app = _app()
    app.app_config["model_capabilities"] = {
        "models": {"claude-sonnet-4-6": {"context_window": 123_456}}
    }
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _deep_link_to_anthropic_key(host, pilot)

        screen.query_one("#settings-model-context-window-reset", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert screen._provider_draft_value("model_context_window_reset") is True
        assert _staged_pair(screen) == _LINKED_PAIR


@pytest.mark.asyncio
@private_profile_test
async def test_a_value_equal_to_the_saved_default_s_is_staged_in_a_linked_form(
    request, monkeypatch
):
    """Finding 11(c): the equal-to-current guard compared with the default
    provider's Env var, so the same name typed for the linked one was dropped."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = _app()
    app.app_config["api_settings"]["openai"]["api_key_env_var"] = "SHARED_KEY_VAR"
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _deep_link_to_anthropic_key(host, pilot)
        env_var = screen.query_one("#settings-provider-credential-env-var", Input)
        assert env_var.value == ""

        env_var.value = "SHARED_KEY_VAR"
        await pilot.pause()

        assert screen._provider_draft_value("credential_env_var") == "SHARED_KEY_VAR"
        assert _staged_pair(screen) == _LINKED_PAIR


@pytest.mark.asyncio
@private_profile_test
async def test_a_refused_saved_model_default_is_no_edit_in_a_linked_form(request):
    """Finding 11(d): a hand-edited ``seed = -1`` is shown as a new chat reads
    it and its mount echo is no edit; under a link the stager compared the
    echo with the DEFAULT pair's saved value and staged it."""
    app = _app()
    app.app_config["api_settings"]["llama_cpp"] = {
        "api_url": "http://127.0.0.1:9099",
        "model_defaults": {"qwen": {"seed": -1, "temperature": 3.0}},
    }
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _deep_link_to_anthropic_key(host, pilot, "llama_cpp", "qwen")
        seed = screen.query_one("#settings-model-profile-seed", Input)
        assert seed.value == "-1"
        assert _form_provider(screen)[1:] == ("llama_cpp", "llama_cpp")

        # The echo an Input posts when it mounts, delivered again.
        screen._stage_model_profile_input(
            "model_profile_seed", "-1", screen._normalise_model_profile_seed
        )
        await pilot.pause()

        assert not screen._category_has_unsaved_changes(PROVIDERS_MODELS)


@pytest.mark.asyncio
@private_profile_test
async def test_an_edit_typed_and_deleted_in_a_linked_form_leaves_no_draft(
    request, monkeypatch
):
    """Finding 16: the link's mount-echo guard ran with a draft too, so the
    Backspace that restored the saved value was dropped and the draft kept
    the deleted character."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _deep_link_to_anthropic_key(host, pilot)
        for selector in (
            "#settings-provider-credential-env-var",
            "#settings-provider-endpoint-value",
        ):
            screen.query_one(selector, Input).focus()
            await pilot.pause()
            await pilot.press("x")
            await pilot.pause()
            assert screen._category_has_unsaved_changes(PROVIDERS_MODELS), selector

            await pilot.press("backspace")
            await pilot.pause()
            assert not screen._category_has_unsaved_changes(PROVIDERS_MODELS), selector
            assert _form_provider(screen) == _ANTHROPIC


@pytest.mark.asyncio
@private_profile_test
async def test_the_linked_model_s_window_can_be_staged_for_a_model_chosen_after(
    request,
):
    """Finding 16 (2): with a model chosen in the linked form the context
    window guard still compared with the LINK's model, so that model's own
    size, pasted in one event, was not staged."""
    from Tests.UI.test_settings_default_model_picker import _CatalogScope, _widgets

    app = _app()
    app.app_config["api_settings"]["anthropic"]["api_key"] = _TYPED_KEY
    app.app_config["model_capabilities"] = {
        "models": {"claude-sonnet-4-6": {"context_window": 123_456}}
    }
    app.llm_provider_catalog_scope_service = _CatalogScope(
        "anthropic", catalog=("claude-sonnet-4-6", "claude-opus-4-1")
    )
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _deep_link_to_anthropic_key(host, pilot)
        _picker, field, _results, adapter = _widgets(screen)
        field.focus()
        await pilot.pause()
        await pilot.press(*"opus", "down", "enter")
        await _settle(host, pilot)
        assert adapter.value == "claude-opus-4-1"

        screen.query_one("#settings-model-context-window", Input).value = "123456"
        await pilot.pause()

        assert screen._provider_draft_value("model_context_window") == 123_456


@pytest.mark.asyncio
@private_profile_test
async def test_the_key_check_row_is_composed_for_the_linked_provider(
    request, monkeypatch
):
    """Finding 6: recomposed (another category and back), the card built
    every Connect row from the linked provider but asked the SAVED default
    for the Key check verdict while the form's widgets were not there yet:
    "Ready" for OpenAI's key beside Anthropic's "missing", until the
    subscription poll's next tick re-said it."""
    from tldw_chatbook.UI.Settings_Modules import providers_models_card as card

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _deep_link_to_anthropic_key(host, pilot)
        said: list[str] = []
        verdict = card.key_check_verdict
        monkeypatch.setattr(
            card, "key_check_verdict", lambda s: said.append(verdict(s)) or said[-1]
        )

        screen._select_category(SettingsCategoryId.CONSOLE_BEHAVIOR.value)
        await _settle(host, pilot)
        screen._select_category(PROVIDERS_MODELS.value)
        await _settle(host, pilot)
        await pilot.pause()

        assert _form_provider(screen) == _ANTHROPIC
        assert said and all(word.startswith("Not ready") for word in said), said


def _pair_words(screen) -> list[str]:
    """The Provider and Model rows' Source words."""
    return [
        str(screen.query_one(selector, Static).renderable)
        for selector in ("#settings-provider-source", "#settings-model-source")
    ]


@pytest.mark.asyncio
@private_profile_test
async def test_a_linked_pair_never_reads_new_chat_default(request, monkeypatch):
    """Finding 5: the Provider and Model rows read "new-chat default" for a
    linked pair that is not the default, and went on reading it after the
    first edit although Save then writes that pair."""
    from tldw_chatbook.UI.Settings_Modules.providers_models_card import (
        SELECTION_SOURCE_WORDS,
    )

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _deep_link_to_anthropic_key(host, pilot)
        assert _pair_words(screen) == [SELECTION_SOURCE_WORDS["navigation"]] * 2

        await pilot.press("x")
        await pilot.pause()

        assert _pair_words(screen) == ["edited *", "edited *"]
        assert screen._provider_return_dirty_field_names() == (
            "API key",
            "Model",
            "Provider",
        )


@pytest.mark.asyncio
@private_profile_test
async def test_an_edit_save_does_not_make_the_default_leaves_the_pair_unedited(
    request, monkeypatch
):
    """Fix round 2, finding 4: any draft marked the linked pair "edited *"
    and listed it unsaved, but Save writes chat_defaults only for a
    connection edit. Sign in with alone wrote auth_source and left the
    default where it was, and both rows read "not the default" again."""
    from tldw_chatbook.UI.Settings_Modules.providers_models_card import (
        SELECTION_SOURCE_WORDS,
    )

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    mutations = _capture_provider_settings_mutations(monkeypatch)
    host = _SettingsCssHarness(_app(), "settings")
    linked = [SELECTION_SOURCE_WORDS["navigation"]] * 2

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _deep_link_to_anthropic_key(host, pilot)
        screen.query_one(
            "#settings-provider-auth-source", Select
        ).value = "claude_subscription"
        await pilot.pause()

        assert _pair_words(screen) == linked
        assert screen._provider_return_dirty_field_names() == ("Sign in with",)
        assert screen._category_unsaved_count(PROVIDERS_MODELS) == 1

        await pilot.press("escape", "s")
        await _settle(host, pilot)

        assert len(mutations) == 1, mutations
        values, _deleted = mutations[0]
        assert "chat_defaults" not in values
        assert values["api_settings.anthropic"]["auth_source"] == (
            "claude_subscription"
        )
        assert _pair_words(screen) == linked
