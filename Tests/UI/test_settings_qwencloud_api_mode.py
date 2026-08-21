"""Canonical Settings coverage for QwenCloud's provider-scoped API mode."""

from __future__ import annotations

import hmac
from copy import deepcopy

import pytest
from textual.widgets import Input, Select, Static

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
)
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_configuration_hub import (
    _open_settings_category,
    _settle_settings_mount_storm,
)
from tldw_chatbook.Chat import provider_setup_persistence as provider_persistence_module
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
from tldw_chatbook.config import ConfigMutationResult
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId


def _qwencloud_app(*, api_mode: object = "__absent__"):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "QwenCloud",
        "model": "qwen3.8-max",
    }
    qwencloud = {
        "api_base_url": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
        "api_key_env_var": "DASHSCOPE_API_KEY",
        "model": "qwen3.8-max",
    }
    if api_mode != "__absent__":
        qwencloud["api_mode"] = api_mode
    app.app_config["api_settings"] = {"qwencloud": qwencloud}
    return app


def _switch_to_qwencloud(screen) -> None:
    """Select QwenCloud through the first-class Settings provider inventory."""
    provider = screen.query_one("#settings-provider-value", Select)
    provider.value = "qwencloud"
    screen.handle_provider_value_changed(Select.Changed(provider, "qwencloud"))


def _capture_atomic_writes(
    monkeypatch,
    result: ConfigMutationResult | None = None,
):
    mutation_result = result or ConfigMutationResult(True, True, None)
    calls: list[tuple[dict[str, dict[str, object]], dict[str, tuple[str, ...]]]] = []

    def write(section_values, *, delete_keys=None):
        calls.append(
            (
                {
                    section: deepcopy(dict(values))
                    for section, values in section_values.items()
                },
                {section: tuple(keys) for section, keys in (delete_keys or {}).items()},
            )
        )
        return mutation_result

    monkeypatch.setattr(
        provider_persistence_module,
        "apply_settings_mutation_to_cli_config",
        write,
    )
    return calls


def _expected_qwen_sections(
    *,
    endpoint: str,
    mode: str,
    provider_section: str = "qwencloud",
    env_var: str = "DASHSCOPE_API_KEY",
) -> dict[str, dict[str, object]]:
    return {
        f"api_settings.{provider_section}": {
            "model": "qwen3.8-max",
            "api_base_url": endpoint,
            "api_key_env_var": env_var,
            "credential_source": "environment",
            **({"api_mode": mode} if provider_section == "qwencloud" else {}),
        },
        "chat_defaults": {"provider": "qwencloud", "model": "qwen3.8-max"},
        "provider_setup.confirmed": {"qwencloud": True},
    }


@pytest.mark.asyncio
async def test_qwencloud_api_mode_selector_visibility_options_and_default():
    app = _qwencloud_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        selector = screen.query_one("#settings-provider-api-mode", Select)
        row = screen.query_one("#settings-provider-api-mode-row")
        provider = screen.query_one("#settings-provider-value", Select)

        assert selector.value == "responses"
        assert selector.disabled is False
        assert selector.can_focus is True
        assert row.has_class("settings-gated-profile-hidden") is False
        assert {
            (str(label), str(value))
            for label, value in selector._options
            if value is not Select.NULL
        } == {
            ("Responses", "responses"),
            ("Chat Completions", "chat_completions"),
        }
        assert "API mode" in str(row.query_one(".settings-input-label", Static).content)
        assert provider.value == "qwencloud"

        provider.value = "openai"
        screen.handle_provider_value_changed(Select.Changed(provider, "openai"))
        await pilot.pause()

        assert selector.disabled is True
        assert row.has_class("settings-gated-profile-hidden") is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    ("API mode", "api_settings.<provider>.api_mode", "mode"),
)
async def test_qwencloud_api_mode_field_search_enter_focuses_selector(query):
    app = _qwencloud_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings_mount_storm(pilot)
        await pilot.click("#settings-category-search")
        await pilot.press(*tuple(query))
        await pilot.press("enter")
        for _ in range(8):
            await pilot.pause()

        screen = _active_destination_screen(host)
        assert screen.active_category == SettingsCategoryId.PROVIDERS_MODELS.value
        assert host.focused is not None
        assert host.focused.id == "settings-provider-api-mode"
        assert screen.query_one("#settings-provider-api-mode", Select).disabled is False


@pytest.mark.asyncio
async def test_qwencloud_api_mode_loads_saved_chat_completions():
    app = _qwencloud_app(api_mode="chat_completions")
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)

        selector = screen.query_one("#settings-provider-api-mode", Select)
        assert selector.value == "chat_completions"
        assert selector.disabled is False


@pytest.mark.asyncio
async def test_normalized_saved_qwencloud_mode_does_not_create_mount_draft():
    app = _qwencloud_app(api_mode="  Chat_Completions  ")
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)

        assert (
            screen.query_one("#settings-provider-api-mode", Select).value
            == "chat_completions"
        )
        assert SettingsCategoryId.PROVIDERS_MODELS not in screen._settings_drafts


@pytest.mark.asyncio
async def test_returning_qwencloud_mode_to_original_removes_only_its_namespace():
    app = _qwencloud_app(api_mode="responses")
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)
        screen._stage_provider_value("model", "qwen3.8-max-draft")

        mode.value = "chat_completions"
        screen.handle_provider_api_mode_changed(
            Select.Changed(mode, "chat_completions")
        )
        mode.value = "responses"
        screen.handle_provider_api_mode_changed(Select.Changed(mode, "responses"))

        draft = screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS]
        assert draft.values["model"] == "qwen3.8-max-draft"
        assert "provider_api_mode:qwencloud" not in draft.values
        assert "provider_api_mode:qwencloud" not in draft.originals


@pytest.mark.asyncio
async def test_immediate_qwencloud_save_snapshots_visible_mode(monkeypatch):
    app = _qwencloud_app(api_mode="responses")
    calls = _capture_atomic_writes(monkeypatch)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)

        mode.value = "chat_completions"
        screen.action_settings_save_category(allow_text_entry_focus=True)

        assert calls == [
            (
                {"api_settings.qwencloud": {"api_mode": "chat_completions"}},
                {},
            )
        ]
        assert app.app_config["chat_defaults"] == {
            "provider": "QwenCloud",
            "model": "qwen3.8-max",
        }
        assert app.app_config["api_settings"]["qwencloud"]["api_base_url"] == (
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        assert app.app_config["api_settings"]["qwencloud"]["api_key_env_var"] == (
            "DASHSCOPE_API_KEY"
        )
        assert app.app_config["api_settings"]["qwencloud"]["api_mode"] == (
            "chat_completions"
        )


@pytest.mark.asyncio
async def test_qwencloud_api_mode_accepts_normalized_provider_identity():
    app = _qwencloud_app(api_mode="responses")
    app.app_config["chat_defaults"]["provider"] = "  QWENCLOUD  "
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)

        assert screen.query_one("#settings-provider-value", Select).value == "qwencloud"
        mode = screen.query_one("#settings-provider-api-mode", Select)
        assert mode.disabled is False
        assert mode.value == "responses"


@pytest.mark.asyncio
async def test_qwencloud_readiness_overlays_draft_mode_without_persisting():
    app = _qwencloud_app(api_mode="responses")
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)

        mode.value = "chat_completions"
        screen.handle_provider_api_mode_changed(
            Select.Changed(mode, "chat_completions")
        )
        staged = screen._provider_test_staged_config("QWENCLOUD")

        assert staged["api_settings"]["qwencloud"]["api_mode"] == ("chat_completions")
        assert app.app_config["api_settings"]["qwencloud"]["api_mode"] == "responses"


@pytest.mark.asyncio
async def test_qwencloud_save_failure_preserves_mode_input_and_draft(monkeypatch):
    app = _qwencloud_app(api_mode="responses")
    calls = _capture_atomic_writes(
        monkeypatch, ConfigMutationResult(False, False, "before_replace")
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)

        mode.value = "chat_completions"
        screen.handle_provider_api_mode_changed(
            Select.Changed(mode, "chat_completions")
        )
        screen.action_settings_save_category(allow_text_entry_focus=True)

        assert mode.value == "chat_completions"
        draft = screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS]
        assert draft.values["provider_api_mode:qwencloud"] == "chat_completions"
        assert app.app_config["api_settings"]["qwencloud"]["api_mode"] == "responses"
        assert len(calls) == 1
        assert (
            "not fully applied"
            in str(
                screen.query_one("#settings-provider-save-result", Static).content
            ).lower()
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        ConfigMutationResult(False, False, "before_replace"),
        ConfigMutationResult(True, False, "cache_reload"),
    ],
    ids=("before-replace", "cache-reload"),
)
async def test_qwencloud_canonical_multifield_save_failure_is_atomic(
    monkeypatch,
    result,
):
    app = _qwencloud_app(api_mode="responses")
    original_api_settings = deepcopy(app.app_config["api_settings"])
    disk_api_settings = deepcopy(original_api_settings)
    atomic_calls = _capture_atomic_writes(monkeypatch, result)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        mode = screen.query_one("#settings-provider-api-mode", Select)
        endpoint.value = "https://new.example.test/compatible-mode/v1"
        mode.value = "chat_completions"
        screen.handle_provider_api_mode_changed(
            Select.Changed(mode, "chat_completions")
        )
        await pilot.pause()

        await pilot.click("#settings-save-category")

        assert atomic_calls == [
            (
                _expected_qwen_sections(
                    endpoint="https://new.example.test/compatible-mode/v1",
                    mode="chat_completions",
                ),
                {"api_settings.qwencloud": ("api_key",)},
            )
        ]
        assert disk_api_settings == original_api_settings
        assert app.app_config["api_settings"] == original_api_settings
        assert endpoint.value == "https://new.example.test/compatible-mode/v1"
        assert mode.value == "chat_completions"
        draft = screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS]
        assert draft.values["endpoint"] == endpoint.value
        assert draft.values["provider_api_mode:qwencloud"] == "chat_completions"
        assert (
            "not fully applied"
            in str(
                screen.query_one("#settings-provider-save-result", Static).content
            ).lower()
        )


@pytest.mark.asyncio
async def test_qwencloud_canonical_multifield_save_uses_one_atomic_mutation(
    monkeypatch,
):
    app = _qwencloud_app(api_mode="responses")
    atomic_calls = _capture_atomic_writes(monkeypatch)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        mode = screen.query_one("#settings-provider-api-mode", Select)
        endpoint.value = "https://new.example.test/compatible-mode/v1"
        mode.value = "chat_completions"
        screen.handle_provider_api_mode_changed(
            Select.Changed(mode, "chat_completions")
        )
        await pilot.pause()

        await pilot.click("#settings-save-category")

        assert atomic_calls == [
            (
                _expected_qwen_sections(
                    endpoint="https://new.example.test/compatible-mode/v1",
                    mode="chat_completions",
                ),
                {"api_settings.qwencloud": ("api_key",)},
            )
        ]
        assert app.app_config["api_settings"]["qwencloud"]["api_base_url"] == (
            "https://new.example.test/compatible-mode/v1"
        )
        assert app.app_config["api_settings"]["qwencloud"]["api_mode"] == (
            "chat_completions"
        )
        assert SettingsCategoryId.PROVIDERS_MODELS not in screen._settings_drafts


@pytest.mark.asyncio
async def test_qwencloud_atomic_save_consumes_deferred_api_key_clear_once(
    monkeypatch,
):
    app = _qwencloud_app(api_mode="responses")

    atomic_calls = _capture_atomic_writes(monkeypatch)
    host = DestinationHarness(app, "settings")
    credential = "local-test-credential"

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        api_key = screen.query_one("#settings-provider-api-key", Input)
        mode = screen.query_one("#settings-provider-api-mode", Select)

        endpoint.value = "https://new.example.test/compatible-mode/v1"
        api_key.focus()
        await pilot.pause()
        await pilot.press(*tuple(credential))
        mode.value = "chat_completions"
        await pilot.pause()

        assert endpoint.value == "https://new.example.test/compatible-mode/v1"
        assert len(api_key.value) == len(credential)
        assert mode.value == "chat_completions"

        await pilot.click("#settings-save-category")
        for _ in range(4):
            await pilot.pause()

        assert len(atomic_calls) == 1
        sections, deletes = atomic_calls[0]
        assert sections["api_settings.qwencloud"]["api_key"] == credential
        assert deletes["api_settings.qwencloud"] == ("api_key_env_var",)
        assert api_key.value == ""
        assert SettingsCategoryId.PROVIDERS_MODELS not in screen._settings_drafts
        stored_provider = app.app_config["api_settings"]["qwencloud"]
        assert stored_provider["api_base_url"] == (
            "https://new.example.test/compatible-mode/v1"
        )
        assert stored_provider["api_mode"] == "chat_completions"
        stored_key = stored_provider["api_key"]
        assert isinstance(stored_key, str)
        assert hmac.compare_digest(stored_key, credential)

        await pilot.click("#settings-save-category")
        for _ in range(2):
            await pilot.pause()

        assert len(atomic_calls) == 1
        assert SettingsCategoryId.PROVIDERS_MODELS not in screen._settings_drafts
        stored_key = app.app_config["api_settings"]["qwencloud"]["api_key"]
        assert isinstance(stored_key, str)
        assert hmac.compare_digest(stored_key, credential)


@pytest.mark.asyncio
async def test_provider_switch_clear_is_suppressed_but_user_key_clear_is_staged():
    app = _qwencloud_app(api_mode="responses")
    app.app_config["api_settings"]["openai"] = {
        "api_base_url": "https://api.openai.com/v1",
        "api_key": "saved-test-credential",
        "model": "gpt-4.1",
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        api_key = screen.query_one("#settings-provider-api-key", Input)
        api_key.focus()
        await pilot.pause()
        await pilot.press("x")

        provider = screen.query_one("#settings-provider-value", Select)
        provider.value = "openai"
        screen.handle_provider_value_changed(Select.Changed(provider, "openai"))
        for _ in range(3):
            await pilot.pause()

        draft = screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS]
        assert api_key.value == ""
        assert "api_key" not in draft.values

        api_key.focus()
        await pilot.pause()
        await pilot.press("x")
        await pilot.press("backspace")
        await pilot.pause()

        draft = screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS]
        assert draft.values["api_key"] == ""


@pytest.mark.asyncio
async def test_qwencloud_api_mode_draft_survives_provider_switch():
    app = _qwencloud_app(api_mode="responses")
    app.app_config["api_settings"]["openai"] = {
        "api_base_url": "https://api.openai.com/v1",
        "model": "gpt-4.1",
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)
        provider = screen.query_one("#settings-provider-value", Select)

        # Switch immediately after changing the widget. The provider transition
        # must snapshot QwenCloud even if its queued Changed event has not run.
        mode.value = "chat_completions"
        provider.value = "openai"
        screen.handle_provider_value_changed(Select.Changed(provider, "openai"))
        await pilot.pause()

        draft = screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS]
        assert draft.values["provider_api_mode:qwencloud"] == "chat_completions"
        assert mode.disabled is True

        _switch_to_qwencloud(screen)
        await pilot.pause()

        assert mode.value == "chat_completions"
        assert mode.disabled is False


@pytest.mark.asyncio
async def test_saving_other_provider_never_mutates_qwencloud_mode(monkeypatch):
    app = _qwencloud_app(api_mode="responses")
    app.app_config["api_settings"]["openai"] = {
        "api_base_url": "https://api.openai.com/v1",
        "model": "gpt-4.1",
    }
    calls = _capture_atomic_writes(monkeypatch)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)
        provider = screen.query_one("#settings-provider-value", Select)

        mode.value = "chat_completions"
        screen.handle_provider_api_mode_changed(
            Select.Changed(mode, "chat_completions")
        )
        provider.value = "openai"
        screen.handle_provider_value_changed(Select.Changed(provider, "openai"))
        await pilot.pause()
        await pilot.click("#settings-save-category")

        assert app.app_config["api_settings"]["qwencloud"]["api_mode"] == "responses"
        assert len(calls) == 1
        sections, deletes = calls[0]
        assert "api_settings.qwencloud" not in sections
        assert "api_settings.qwencloud" not in deletes
        draft = screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS]
        assert draft.values["provider_api_mode:qwencloud"] == "chat_completions"

        _switch_to_qwencloud(screen)
        await pilot.pause()
        assert mode.value == "chat_completions"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("saved_mode", "selected_mode"),
    [
        ("responses", "chat_completions"),
        ("chat_completions", "responses"),
    ],
)
async def test_qwencloud_api_mode_save_and_revert_exact_values(
    monkeypatch,
    saved_mode: str,
    selected_mode: str,
):
    app = _qwencloud_app(api_mode=saved_mode)
    calls = _capture_atomic_writes(monkeypatch)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)

        mode.value = selected_mode
        screen.handle_provider_api_mode_changed(Select.Changed(mode, selected_mode))
        await pilot.click("#settings-save-category")

        assert calls == [
            (
                {"api_settings.qwencloud": {"api_mode": selected_mode}},
                {},
            )
        ]
        assert app.app_config["chat_defaults"] == {
            "provider": "QwenCloud",
            "model": "qwen3.8-max",
        }
        assert app.app_config["api_settings"]["qwencloud"]["api_mode"] == selected_mode
        draft = screen._settings_drafts.get(SettingsCategoryId.PROVIDERS_MODELS)
        assert draft is None or "provider_api_mode:qwencloud" not in draft.values

        mode.value = saved_mode
        screen.handle_provider_api_mode_changed(Select.Changed(mode, saved_mode))
        await pilot.click("#settings-revert-category")
        await pilot.pause()
        await pilot.click("#confirm-button")
        await pilot.pause()

        assert mode.value == selected_mode
        assert SettingsCategoryId.PROVIDERS_MODELS not in screen._settings_drafts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_mode",
    ["response", "", "   ", 42, False, [], {}],
    ids=("unknown", "empty", "whitespace", "integer", "boolean", "list", "mapping"),
)
async def test_invalid_persisted_qwencloud_mode_blocks_save_and_send(
    monkeypatch,
    invalid_mode: object,
):
    app = _qwencloud_app(api_mode=invalid_mode)
    app.app_config["api_settings"]["qwencloud"]["api_key"] = "KEY-CANARY"
    calls = _capture_atomic_writes(monkeypatch)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)

        assert mode.value is Select.NULL
        assert mode.has_class("settings-invalid-input")
        assert SettingsCategoryId.PROVIDERS_MODELS not in screen._settings_drafts
        screen.action_settings_save_category(allow_text_entry_focus=True)
        await pilot.pause()

        assert calls == []
        assert "QwenCloud API mode is invalid" in str(
            screen.query_one("#settings-provider-save-result", Static).content
        )
        assert app.app_config["api_settings"]["qwencloud"]["api_mode"] == invalid_mode

    gateway = ConsoleProviderGateway(
        config_provider=lambda: app.app_config,
        environ={},
        chat_api_call_fn=lambda **_kwargs: pytest.fail(
            "invalid QwenCloud mode must block before dispatch"
        ),
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="QwenCloud")
    )

    assert resolution.ready is False
    assert "invalid API mode setting" in resolution.visible_copy
    assert "KEY-CANARY" not in resolution.visible_copy


@pytest.mark.asyncio
async def test_invalid_qwencloud_mode_keyboard_selection_repairs_to_responses(
    monkeypatch,
):
    app = _qwencloud_app(api_mode="invalid")
    calls = _capture_atomic_writes(monkeypatch)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)

        assert mode.value is Select.NULL
        assert SettingsCategoryId.PROVIDERS_MODELS not in screen._settings_drafts
        assert "Open API mode and choose Responses or Chat Completions" in str(
            screen.query_one("#settings-provider-api-mode-guidance", Static).content
        )

        mode.focus()
        await pilot.pause()
        await pilot.press("enter")
        assert mode.expanded is True
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()

        assert mode.expanded is False
        assert mode.value == "responses"
        draft = screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS]
        assert draft.values["provider_api_mode:qwencloud"] == "responses"
        assert mode.has_class("settings-invalid-input") is False

        await pilot.click("#settings-save-category")

        assert calls == [
            (
                {"api_settings.qwencloud": {"api_mode": "responses"}},
                {},
            )
        ]
        assert app.app_config["api_settings"]["qwencloud"]["api_mode"] == "responses"


@pytest.mark.asyncio
async def test_qwencloud_api_mode_field_guide_describes_mode_contract():
    app = _qwencloud_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)
        mode.focus()
        await pilot.pause()

        assert screen._active_settings_field_id == "settings-provider-api-mode"
        guide = " | ".join(
            f"{label}: {value}"
            for label, value in screen._provider_field_guidance_rows()
        )
        assert "Focused setting: API mode" in guide
        assert "api_settings.qwencloud.api_mode" in guide
        assert "Responses is stateless with store=false" in guide
        assert "Chat Completions disables thinking replay" in guide
        assert "existing function tools work in both" in guide
        assert "QwenCloud built-in tools are excluded" in guide

        ownership = screen._ownership_record(SettingsCategoryId.PROVIDERS_MODELS)
        assert "api_settings.<provider>.api_mode" in ownership.owns_config_sections


@pytest.mark.asyncio
async def test_qwencloud_alias_mode_save_migrates_only_mode_to_canonical_owner(
    monkeypatch,
):
    app = _qwencloud_app(api_mode="responses")
    alias_fields = {
        "api_mode": "responses",
        "api_base_url": "https://proxy.example.com/compatible-mode/v1",
        "api_key_env_var": "DASHSCOPE_API_KEY",
        "model": "qwen3.8-max",
    }
    other_provider = {"api_base_url": "https://api.openai.com/v1"}
    app.app_config["api_settings"] = {
        "QwenCloud": dict(alias_fields),
        "openai": dict(other_provider),
    }
    atomic_mutations = _capture_atomic_writes(monkeypatch)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)

        mode.value = "chat_completions"
        screen.handle_provider_api_mode_changed(
            Select.Changed(mode, "chat_completions")
        )
        await pilot.click("#settings-save-category")

    assert atomic_mutations == [
        (
            {"api_settings.qwencloud": {"api_mode": "chat_completions"}},
            {"api_settings.QwenCloud": ("api_mode",)},
        )
    ]
    assert app.app_config["api_settings"]["qwencloud"] == {
        "api_mode": "chat_completions"
    }
    assert app.app_config["api_settings"]["QwenCloud"] == {
        key: value for key, value in alias_fields.items() if key != "api_mode"
    }
    assert app.app_config["api_settings"]["openai"] == other_provider

    gateway = ConsoleProviderGateway(
        config_provider=lambda: app.app_config,
        environ={"DASHSCOPE_API_KEY": "TEST-KEY"},
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="QwenCloud")
    )
    assert resolution.ready is True
    assert resolution.api_mode == "chat_completions"


@pytest.mark.asyncio
@pytest.mark.parametrize("canonical_first", [False, True])
async def test_qwencloud_malformed_canonical_table_uses_canonical_recovery_without_alias_credentials(
    canonical_first,
):
    app = _qwencloud_app()
    alias = {
        "api_mode": "chat_completions",
        "api_key": "ALIAS-SECRET-CANARY",
        "api_key_env_var": "ALIAS_ENV_CANARY",
        "model": "alias-model",
    }
    entries = (
        [("qwencloud", []), ("QwenCloud", alias)]
        if canonical_first
        else [("QwenCloud", alias), ("qwencloud", [])]
    )
    app.app_config["api_settings"] = dict(entries)
    original_api_settings = deepcopy(app.app_config["api_settings"])
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        selector = screen.query_one("#settings-provider-api-mode", Select)
        guidance = screen.query_one("#settings-provider-api-mode-guidance", Static)
        env_var = screen.query_one("#settings-provider-credential-env-var", Input)

        assert screen._provider_config_entry("QwenCloud")[0] == "qwencloud"
        assert selector.value is Select.NULL
        recovery = str(guidance.content)
        assert "api_settings.qwencloud is not a table" in recovery
        assert "Advanced Config" in recovery
        assert "config.toml" in recovery
        assert env_var.value == ""
        assert "Provider settings invalid" in screen._provider_key_status("QwenCloud")
        assert "ALIAS-SECRET-CANARY" not in screen._provider_key_status("QwenCloud")
        assert "ALIAS_ENV_CANARY" not in screen._provider_key_status("QwenCloud")
        assert app.app_config["api_settings"] == original_api_settings


@pytest.mark.asyncio
async def test_malformed_qwencloud_table_selection_test_and_save_stay_blocked(
    monkeypatch,
):
    app = _qwencloud_app()
    app.app_config["api_settings"] = {"qwencloud": ["not-a-table"]}
    original_api_settings = deepcopy(app.app_config["api_settings"])
    atomic_writes = _capture_atomic_writes(monkeypatch)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)
        guidance = screen.query_one("#settings-provider-api-mode-guidance", Static)

        assert mode.value is Select.NULL
        assert mode.has_class("settings-invalid-input")
        assert "Advanced Config" in str(guidance.content)

        mode.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()

        assert mode.value == "responses"
        assert mode.has_class("settings-invalid-input")
        assert "api_settings.qwencloud is not a table" in str(guidance.content)
        draft = screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS]
        assert draft.values["provider_api_mode:qwencloud"] == "responses"

        await pilot.click("#settings-test-provider")
        assert "Invalid provider settings" in str(
            screen.query_one("#settings-provider-test-result", Static).content
        )

        await pilot.click("#settings-save-category")
        await pilot.pause()

        assert atomic_writes == []
        assert app.app_config["api_settings"] == original_api_settings
        assert "Advanced Config" in str(
            screen.query_one("#settings-provider-save-result", Static).content
        )
        draft = screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS]
        assert draft.values["provider_api_mode:qwencloud"] == "responses"


@pytest.mark.asyncio
@pytest.mark.parametrize("canonical_first", [False, True])
async def test_qwencloud_malformed_alias_never_becomes_save_owner_when_canonical_is_valid(
    canonical_first,
    monkeypatch,
):
    app = _qwencloud_app(api_mode="responses")
    canonical = {
        "api_mode": "responses",
        "api_key_env_var": "CANONICAL_ENV",
        "model": "qwen3.8-max",
    }
    entries = (
        [("qwencloud", canonical), ("QwenCloud", [])]
        if canonical_first
        else [("QwenCloud", []), ("qwencloud", canonical)]
    )
    app.app_config["api_settings"] = dict(entries)
    calls = _capture_atomic_writes(monkeypatch)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)

        assert screen._provider_config_entry("QwenCloud")[0] == "qwencloud"
        assert screen.query_one("#settings-provider-api-mode", Select).value == (
            "responses"
        )
        assert (
            screen.query_one("#settings-provider-credential-env-var", Input).value
            == "CANONICAL_ENV"
        )
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.value = "https://new.example.test/compatible-mode/v1"
        await pilot.pause()
        await pilot.click("#settings-save-category")

        assert calls == [
            (
                {
                    "api_settings.qwencloud": {
                        "model": "qwen3.8-max",
                        "api_base_url": "https://new.example.test/compatible-mode/v1",
                        "api_key_env_var": "CANONICAL_ENV",
                        "credential_source": "environment",
                    },
                    "chat_defaults": {
                        "provider": "qwencloud",
                        "model": "qwen3.8-max",
                    },
                    "provider_setup.confirmed": {"qwencloud": True},
                },
                {"api_settings.qwencloud": ("api_key",)},
            )
        ]
        assert app.app_config["api_settings"]["QwenCloud"] == []
        assert app.app_config["api_settings"]["qwencloud"]["api_base_url"] == (
            "https://new.example.test/compatible-mode/v1"
        )


@pytest.mark.asyncio
async def test_qwencloud_alias_only_malformed_table_uses_canonical_recovery_owner():
    app = _qwencloud_app()
    app.app_config["api_settings"] = {"QwenCloud": ["SECRET-CANARY"]}
    original_api_settings = deepcopy(app.app_config["api_settings"])
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)

        assert screen._provider_config_entry("QwenCloud")[0] == "qwencloud"
        assert screen.query_one("#settings-provider-api-mode", Select).value is (
            Select.NULL
        )
        assert (
            screen.query_one("#settings-provider-credential-env-var", Input).value == ""
        )
        assert "SECRET-CANARY" not in screen._provider_key_status("QwenCloud")
        assert app.app_config["api_settings"] == original_api_settings


@pytest.mark.asyncio
async def test_qwencloud_alias_mode_migration_waits_for_other_provider_writes(
    monkeypatch,
):
    app = _qwencloud_app(api_mode="responses")
    app.app_config["api_settings"] = {
        "QwenCloud": {
            "api_mode": "responses",
            "api_base_url": "https://old.example.test/compatible-mode/v1",
            "api_key_env_var": "DASHSCOPE_API_KEY",
            "model": "qwen3.8-max",
        }
    }
    atomic_mutations = _capture_atomic_writes(
        monkeypatch, ConfigMutationResult(False, False, "before_replace")
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.value = "https://new.example.test/compatible-mode/v1"
        screen._stage_provider_value("endpoint", endpoint.value)
        mode = screen.query_one("#settings-provider-api-mode", Select)
        mode.value = "chat_completions"
        screen.handle_provider_api_mode_changed(
            Select.Changed(mode, "chat_completions")
        )

        await pilot.click("#settings-save-category")

        assert len(atomic_mutations) == 1
        sections, deletes = atomic_mutations[0]
        assert sections["api_settings.QwenCloud"]["api_base_url"] == (
            "https://new.example.test/compatible-mode/v1"
        )
        assert sections["api_settings.qwencloud"] == {"api_mode": "chat_completions"}
        assert "api_mode" in deletes["api_settings.QwenCloud"]
        assert "qwencloud" not in app.app_config["api_settings"]
        assert app.app_config["api_settings"]["QwenCloud"]["api_mode"] == "responses"
        draft = screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS]
        assert draft.values["provider_api_mode:qwencloud"] == "chat_completions"
        assert endpoint.value == "https://new.example.test/compatible-mode/v1"


@pytest.mark.asyncio
async def test_qwencloud_alias_mode_migration_waits_for_profile_write(
    monkeypatch,
):
    app = _qwencloud_app(api_mode="responses")
    app.app_config["api_settings"] = {
        "QwenCloud": {
            "api_mode": "responses",
            "api_key_env_var": "DASHSCOPE_API_KEY",
            "model": "qwen3.8-max",
        }
    }
    atomic_mutations = _capture_atomic_writes(
        monkeypatch, ConfigMutationResult(False, False, "before_replace")
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        profile = screen.query_one("#settings-model-profile-temperature", Input)
        profile.value = "0.25"
        await pilot.pause()
        mode = screen.query_one("#settings-provider-api-mode", Select)
        mode.value = "chat_completions"
        screen.handle_provider_api_mode_changed(
            Select.Changed(mode, "chat_completions")
        )

        await pilot.click("#settings-save-category")

        assert len(atomic_mutations) == 1
        sections, deletes = atomic_mutations[0]
        assert (
            sections["api_settings.QwenCloud"]["model_defaults"]["qwen3.8-max"][
                "temperature"
            ]
            == 0.25
        )
        assert sections["api_settings.qwencloud"] == {"api_mode": "chat_completions"}
        assert "api_mode" in deletes["api_settings.QwenCloud"]
        assert "qwencloud" not in app.app_config["api_settings"]
        assert app.app_config["api_settings"]["QwenCloud"]["api_mode"] == "responses"
        assert mode.value == "chat_completions"
        assert profile.value == "0.25"
        draft = screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS]
        assert draft.values["provider_api_mode:qwencloud"] == "chat_completions"


@pytest.mark.asyncio
async def test_qwencloud_alias_mode_migration_waits_for_context_write(
    monkeypatch,
):
    app = _qwencloud_app(api_mode="responses")
    app.app_config["api_settings"] = {
        "QwenCloud": {
            "api_mode": "responses",
            "api_key_env_var": "DASHSCOPE_API_KEY",
            "model": "qwen3.8-max",
        }
    }
    app.app_config["model_capabilities"] = {}
    atomic_mutations = _capture_atomic_writes(
        monkeypatch, ConfigMutationResult(False, False, "before_replace")
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        context = screen.query_one("#settings-model-context-window", Input)
        context.value = "32768"
        await pilot.pause()
        mode = screen.query_one("#settings-provider-api-mode", Select)
        mode.value = "chat_completions"
        screen.handle_provider_api_mode_changed(
            Select.Changed(mode, "chat_completions")
        )

        await pilot.click("#settings-save-category")

        assert len(atomic_mutations) == 1
        sections, deletes = atomic_mutations[0]
        assert (
            sections["model_capabilities.models"]["qwen3.8-max"]["context_window"]
            == 32768
        )
        assert sections["api_settings.qwencloud"] == {"api_mode": "chat_completions"}
        assert "api_mode" in deletes["api_settings.QwenCloud"]
        assert "qwencloud" not in app.app_config["api_settings"]
        assert app.app_config["api_settings"]["QwenCloud"]["api_mode"] == "responses"
        assert mode.value == "chat_completions"
        assert context.value == "32768"
        draft = screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS]
        assert draft.values["provider_api_mode:qwencloud"] == "chat_completions"


@pytest.mark.asyncio
async def test_qwencloud_canonical_mode_wins_over_alias_for_settings_and_readiness():
    app = _qwencloud_app(api_mode="responses")
    app.app_config["api_settings"] = {
        "QwenCloud": {
            "api_mode": "chat_completions",
            "api_base_url": "https://proxy.example.com/compatible-mode/v1",
            "api_key_env_var": "DASHSCOPE_API_KEY",
            "model": "qwen3.8-max",
        },
        "qwencloud": {"api_mode": "responses"},
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)

        assert mode.value == "responses"
        assert SettingsCategoryId.PROVIDERS_MODELS not in screen._settings_drafts

    gateway = ConsoleProviderGateway(
        config_provider=lambda: app.app_config,
        environ={"DASHSCOPE_API_KEY": "TEST-KEY"},
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="QwenCloud")
    )
    assert resolution.ready is True
    assert resolution.api_mode == "responses"


@pytest.mark.asyncio
async def test_qwencloud_alias_config_readiness_overlay_uses_draft_canonical_mode():
    app = _qwencloud_app(api_mode="responses")
    app.app_config["api_settings"] = {
        "QwenCloud": {
            "api_mode": "responses",
            "api_key_env_var": "DASHSCOPE_API_KEY",
            "model": "qwen3.8-max",
        },
        "qwencloud": {"api_mode": "responses"},
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)
        mode.value = "chat_completions"
        screen.handle_provider_api_mode_changed(
            Select.Changed(mode, "chat_completions")
        )

        staged = screen._provider_test_staged_config("QwenCloud")

    assert staged["api_settings"]["qwencloud"]["api_mode"] == "chat_completions"
    assert app.app_config["api_settings"]["qwencloud"]["api_mode"] == "responses"
    gateway = ConsoleProviderGateway(
        config_provider=lambda: staged,
        environ={"DASHSCOPE_API_KEY": "TEST-KEY"},
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="QwenCloud")
    )
    assert resolution.ready is True
    assert resolution.api_mode == "chat_completions"
