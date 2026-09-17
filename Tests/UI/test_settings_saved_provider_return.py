"""Saved provider changes must reach mounted Settings without stealing drafts."""

import copy

import pytest
from textual.screen import Screen
from textual.widgets import Input, OptionList, Select, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_settings_configuration_hub import (
    _capture_provider_settings_mutations,
    _open_settings_category,
)
from Tests.UI.test_settings_provider_keyboard_journeys import (
    CATEGORY,
    ProviderSettingsHarness,
    _revert,
    _settle,
)
from tldw_chatbook.UI.Screens.provider_model_resolution import provider_config_key
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen


def _change_saved_provider(app):
    config = copy.deepcopy(app.app_config)
    config["chat_defaults"].update(provider="llama_cpp", model="saved-model")
    app.app_config = config


def _assert_projection(screen, provider, model, endpoint):
    assert (
        provider_config_key(screen._provider_setting_values()["provider"]) == provider
    )
    assert screen._provider_widget_value() == provider
    assert screen.query_one("#settings-model-value", Input).value == model
    assert (
        screen.query_one("#settings-provider-endpoint-value", Input).value == endpoint
    )
    picker = screen.query_one("#settings-provider-picker", OptionList)
    assert picker.highlighted is not None
    assert picker.get_option_at_index(picker.highlighted).provider_id == provider
    readiness = str(screen.query_one("#settings-provider-readiness", Static).renderable)
    assert screen._provider_display_name(provider) in readiness
    assert model in readiness
    detail, _, _ = screen._provider_readiness_test_report()
    assert screen._provider_display_name(provider).casefold() in detail.casefold()
    assert f"model={model}" in detail


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("return_kind", ["fresh", "restored", "retained"])
async def test_clean_settings_follows_changed_saved_provider(theme, size, return_kind):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "boot-model"}
    app.app_config["api_settings"] = {
        "openai": {"api_base_url": "https://old.invalid/v1"},
        "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
    }
    host = ProviderSettingsHarness(app, "settings")
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        await _settle(host, pilot)
        screen = host.screen
        _assert_projection(screen, "openai", "boot-model", "https://old.invalid/v1")
        assert not screen._category_has_unsaved_changes(CATEGORY)
        previous_model_widget = screen.query_one("#settings-model-value", Input)
        screen._provider_test_result = "Configuration check | old provider checked"
        state = screen.save_state()
        await host.push_screen(Screen())
        _change_saved_provider(app)
        if return_kind == "retained":
            await host.pop_screen()
            assert host.screen is screen
        else:
            replacement = SettingsScreen(app)
            if return_kind == "restored":
                replacement.restore_state(state)
            else:
                replacement.active_category = CATEGORY.value
            await host.push_screen(replacement)
        await _settle(host, pilot)
        _assert_projection(
            host.screen, "llama_cpp", "saved-model", "http://127.0.0.1:9099"
        )
        assert not host.screen._category_has_unsaved_changes(CATEGORY)
        if return_kind == "retained":
            assert (
                host.screen.query_one("#settings-model-value", Input)
                is not previous_model_widget
            )
            assert "re-run" in host.screen._provider_test_result.lower()
        assert "Saved chat defaults" in str(
            host.screen.query_one("#settings-provider-source", Static).renderable
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("return_kind", ["restored", "retained"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model-value", "unsaved-model"),
        ("provider-endpoint-value", "https://draft.invalid/v1"),
        ("provider-credential-env-var", "REVIEW_PROVIDER_KEY"),
        ("model-profile-temperature", "0.4"),
    ],
)
async def test_saved_default_change_does_not_retarget_a_draft(
    return_kind, field, value
):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "boot-model"}
    app.app_config["api_settings"] = {
        "openai": {"api_base_url": "https://old.invalid/v1"},
        "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
    }
    host = ProviderSettingsHarness(app, "settings")
    async with host.run_test(size=(170, 48)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        await _settle(host, pilot)
        screen = host.screen
        screen.query_one(f"#settings-{field}", Input).value = value
        await _settle(host, pilot)
        assert screen._category_has_unsaved_changes(CATEGORY)
        state = screen.save_state()
        await host.push_screen(Screen())
        _change_saved_provider(app)
        if return_kind == "retained":
            await host.pop_screen()
        else:
            replacement = SettingsScreen(app)
            replacement.restore_state(state)
            await host.push_screen(replacement)
        await _settle(host, pilot)
        _assert_projection(
            host.screen,
            "openai",
            value if field == "model-value" else "boot-model",
            value if field == "provider-endpoint-value" else "https://old.invalid/v1",
        )
        assert host.screen.query_one(f"#settings-{field}", Input).value == value
        assert host.screen._category_has_unsaved_changes(CATEGORY)
        assert app.app_config["chat_defaults"] == {
            "provider": "llama_cpp",
            "model": "saved-model",
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("return_kind", ["restored", "retained"])
async def test_api_mode_only_draft_keeps_its_provider_on_return(return_kind):
    from Tests.UI.test_settings_qwencloud_api_mode import _qwencloud_app

    app = _qwencloud_app(api_mode="responses")
    host = ProviderSettingsHarness(app, "settings")
    async with host.run_test(size=(170, 48)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        await _settle(host, pilot)
        screen = host.screen
        screen.query_one(
            "#settings-provider-api-mode", Select
        ).value = "chat_completions"
        await _settle(host, pilot)
        state = screen.save_state()
        await host.push_screen(Screen())
        _change_saved_provider(app)
        if return_kind == "retained":
            await host.pop_screen()
        else:
            replacement = SettingsScreen(app)
            replacement.restore_state(state)
            await host.push_screen(replacement)
        await _settle(host, pilot)
        _assert_projection(
            host.screen,
            "qwencloud",
            "qwen3.8-max",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        assert (
            host.screen.query_one("#settings-provider-api-mode", Select).value
            == "chat_completions"
        )
        assert host.screen._provider_draft().dirty_keys == {
            "provider_api_mode:qwencloud"
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["save", "revert"])
async def test_second_edit_after_default_change_uses_draft_originals(
    finish, monkeypatch
):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "boot-model"}
    app.app_config["api_settings"] = {
        "openai": {"api_base_url": "https://old.invalid/v1"},
        "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
    }
    mutations = _capture_provider_settings_mutations(monkeypatch)
    host = ProviderSettingsHarness(app, "settings")
    async with host.run_test(size=(170, 48)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        await _settle(host, pilot)
        screen = host.screen
        screen.query_one("#settings-model-value", Input).value = "unsaved-model"
        await _settle(host, pilot)
        await host.push_screen(Screen())
        _change_saved_provider(app)
        await host.pop_screen()
        await _settle(host, pilot)
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.value = "https://draft.invalid/v1"
        await _settle(host, pilot)
        assert (
            screen._provider_draft().originals["endpoint"] == "https://old.invalid/v1"
        )
        endpoint.value = "https://old.invalid/v1"
        await _settle(host, pilot)
        assert screen._provider_draft().dirty_keys == {"model"}
        if finish == "save":
            screen.action_settings_save_category(allow_text_entry_focus=True)
            await _settle(host, pilot)
            assert len(mutations) == 1
            assert mutations[0][0]["chat_defaults"] == {
                "provider": "openai",
                "model": "unsaved-model",
            }
            assert "api_settings.llama_cpp" not in mutations[0][0]
            _assert_projection(
                screen, "openai", "unsaved-model", "https://old.invalid/v1"
            )
        else:
            await _revert(host, pilot, discard=True)
            assert mutations == []
            _assert_projection(
                screen, "llama_cpp", "saved-model", "http://127.0.0.1:9099"
            )
        assert not screen._category_has_unsaved_changes(CATEGORY)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["QwenCloud", "ReviewCloud"])
async def test_unchanged_saved_provider_preserves_retained_widgets_and_verdict(
    provider,
):
    from Tests.UI.test_settings_qwencloud_api_mode import _qwencloud_app

    app = _qwencloud_app()
    app.app_config["chat_defaults"]["provider"] = provider
    host = ProviderSettingsHarness(app, "settings")
    async with host.run_test(size=(170, 48)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        await _settle(host, pilot)
        screen = host.screen
        model = screen.query_one("#settings-model-value", Input)
        screen._provider_test_result = "Configuration check | same provider checked"
        await host.push_screen(Screen())
        await host.pop_screen()
        await _settle(host, pilot)
        assert screen.query_one("#settings-model-value", Input) is model
        assert (
            screen._provider_test_result
            == "Configuration check | same provider checked"
        )
        assert not screen._category_has_unsaved_changes(CATEGORY)
