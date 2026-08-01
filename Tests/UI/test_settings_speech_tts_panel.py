from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Select, Static

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _visible_text,
    _wait_for_selector,
)
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
import tldw_chatbook.UI.Screens.settings_screen as settings_screen_module
from tldw_chatbook.UI.Screens.settings_speech_tts import (
    BUILT_IN_TTS_PROVIDER_ORDER,
    CredentialIntent,
    GLOBAL_TTS_PROVIDER_FIELD_IDS,
    load_global_speech_tts_state,
)
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
    STTSSettingsSaveResult,
)
from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
    SpeechTTSSettingsPanel,
)


async def _settle(pilot) -> None:
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


async def _open_speech_tts(host, pilot):
    await _settle(pilot)
    screen = _active_destination_screen(host)
    button = screen.query_one("#settings-category-speech-tts", Button)
    screen.query_one("#settings-category-list").scroll_to_widget(
        button, animate=False, immediate=True
    )
    await pilot.pause()
    await pilot.click("#settings-category-speech-tts")
    await _wait_for_selector(screen, pilot, "#settings-speech-tts-panel", timeout=8.0)
    return screen


class _PanelHarness(App[None]):
    def __init__(
        self,
        *,
        configure_provider: str = "audio_cpp",
        state=None,
    ) -> None:
        super().__init__()
        self.configure_provider = configure_provider
        self.state = state or load_global_speech_tts_state({})
        self.events: list[STTSSettingsSaveEvent] = []

    def compose(self) -> ComposeResult:
        yield SpeechTTSSettingsPanel(
            state=self.state,
            configure_provider=self.configure_provider,
            id="panel",
        )

    @on(STTSSettingsSaveEvent)
    def record_save(self, event: STTSSettingsSaveEvent) -> None:
        self.events.append(event)
        event.stop()


_BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "css"
    / "tldw_cli_modular.tcss"
)


class _StyledDestinationHarness(DestinationHarness):
    CSS_PATH = _BUNDLE


class _StyledPanelHarness(_PanelHarness):
    CSS_PATH = _BUNDLE


def test_speech_tts_is_a_first_class_core_settings_category() -> None:
    screen = SettingsScreen(_build_test_app())

    summary = next(
        item
        for item in screen._category_summaries()
        if item.category is SettingsCategoryId.SPEECH_TTS
    )

    assert summary.title == "Speech & TTS"
    assert "application-wide" in summary.description.lower()
    assert SettingsCategoryId.SPEECH_TTS in dict(screen._category_groups())["Core"]


@pytest.mark.parametrize(
    "query",
    (
        "speech",
        "TTS",
        "voice",
        "audio.cpp",
        "audio_cpp",
        "OpenAI",
        "ElevenLabs",
        "Kokoro",
        "Chatterbox",
        "Higgs",
        "AllTalk",
    ),
)
def test_settings_search_indexes_required_speech_vocabulary(query: str) -> None:
    screen = SettingsScreen(_build_test_app())
    summary = next(
        item
        for item in screen._category_summaries()
        if item.category is SettingsCategoryId.SPEECH_TTS
    )

    assert screen._category_matches_search(summary, query)


@pytest.mark.asyncio
async def test_provider_search_opens_speech_tts_with_named_provider_selected() -> None:
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        await _settle(pilot)
        screen = _active_destination_screen(host)
        search = screen.query_one("#settings-category-search", Input)
        search.value = "Kokoro"
        search.focus()
        await pilot.press("enter")

        await _wait_for_selector(
            screen,
            pilot,
            "#settings-speech-provider-kokoro",
            timeout=8.0,
        )
        assert screen.active_category == SettingsCategoryId.SPEECH_TTS.value
        assert (
            screen.query_one("#settings-speech-configure-provider", Select).value
            == "kokoro"
        )


@pytest.mark.asyncio
async def test_global_panel_states_scope_and_mounts_only_selected_provider() -> None:
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = await _open_speech_tts(host, pilot)
        text = _visible_text(screen)

        assert "application-wide Speech & TTS defaults" in text
        assert "Studio preferences" in text
        assert screen.query_one("#settings-speech-default-provider", Select)
        assert screen.query_one("#settings-speech-configure-provider", Select)
        assert screen.query_one("#settings-speech-open-lab", Button)
        assert len(screen.query(".settings-speech-provider-form")) == 1

        configure = screen.query_one("#settings-speech-configure-provider", Select)
        assert configure.value in BUILT_IN_TTS_PROVIDER_ORDER
        assert screen.query_one(f"#settings-speech-provider-{configure.value}")


@pytest.mark.asyncio
async def test_real_stylesheet_keeps_fields_usable_and_outer_detail_scrollable() -> (
    None
):
    host = _StyledDestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = await _open_speech_tts(host, pilot)
        panel = screen.query_one(
            "#settings-speech-tts-panel",
            SpeechTTSSettingsPanel,
        )

        assert (
            screen.query_one(
                "#settings-speech-default-provider",
                Select,
            ).region.width
            >= 20
        )
        assert screen.query_one("#settings-speech-speed", Input).region.width >= 20

        configure = screen.query_one("#settings-speech-configure-provider", Select)
        configure.value = "kokoro"
        await _wait_for_selector(
            screen,
            pilot,
            "#settings-speech-provider-kokoro",
            timeout=4.0,
        )
        path_input = screen.query_one(
            "#settings-speech-kokoro-onnx-model-path",
            Input,
        )
        browse = screen.query_one(
            "#settings-speech-kokoro-onnx-model-path-browse",
            Button,
        )
        path_control = path_input.parent
        assert path_control is not None
        assert path_input.region.width >= 20
        assert browse.region.y == path_input.region.y
        assert browse.region.x + browse.region.width <= (
            path_control.region.x + path_control.region.width
        )

        detail = screen.query_one("#settings-detail-pane-body")
        assert panel.max_scroll_y == 0
        assert detail.max_scroll_y > 0
        save = screen.query_one("#settings-speech-save", Button)
        save.scroll_visible(animate=False)
        await pilot.pause()
        assert 0 <= save.region.y < pilot.app.size.height


@pytest.mark.asyncio
async def test_category_reloads_the_latest_runtime_config_snapshot(monkeypatch) -> None:
    monkeypatch.setattr(
        settings_screen_module,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(
            values={
                "COMPREHENSIVE_CONFIG_RAW": {
                    "app_tts": {
                        "default_provider": "openai",
                        "OPENAI_ORG_ID": "org-from-latest-snapshot",
                    }
                }
            }
        ),
    )
    app = _build_test_app()
    app.app_config = {
        "COMPREHENSIVE_CONFIG_RAW": {
            "app_tts": {"OPENAI_ORG_ID": "org-stale-app-startup"}
        }
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        screen = await _open_speech_tts(host, pilot)

        assert (
            screen.query_one(
                "#settings-speech-openai-organization-id",
                Input,
            ).value
            == "org-from-latest-snapshot"
        )


@pytest.mark.asyncio
async def test_each_provider_form_mounts_its_complete_bounded_inventory() -> None:
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(220, 80)) as pilot:
        screen = await _open_speech_tts(host, pilot)
        for provider_id in BUILT_IN_TTS_PROVIDER_ORDER:
            configure = screen.query_one("#settings-speech-configure-provider", Select)
            configure.value = provider_id
            await _wait_for_selector(
                screen,
                pilot,
                f"#settings-speech-provider-{provider_id}",
                timeout=4.0,
            )
            assert len(screen.query(".settings-speech-provider-form")) == 1
            for field_id in GLOBAL_TTS_PROVIDER_FIELD_IDS[provider_id]:
                assert screen.query(
                    f"#settings-speech-{provider_id}-{field_id.replace('_', '-')}"
                ), f"{provider_id} is missing {field_id}"


@pytest.mark.asyncio
async def test_panel_exposes_path_pickers_and_no_managed_audio_cpp_controls() -> None:
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(220, 80)) as pilot:
        screen = await _open_speech_tts(host, pilot)
        for provider_id, picker_ids in {
            "kokoro": (
                "settings-speech-kokoro-onnx-model-path-browse",
                "settings-speech-kokoro-voices-json-path-browse",
            ),
            "chatterbox": (
                "settings-speech-chatterbox-voice-resource-directory-browse",
            ),
            "higgs": (
                "settings-speech-higgs-model-path-browse",
                "settings-speech-higgs-voice-resource-directory-browse",
            ),
        }.items():
            configure = screen.query_one("#settings-speech-configure-provider", Select)
            configure.value = provider_id
            await _wait_for_selector(
                screen,
                pilot,
                f"#settings-speech-provider-{provider_id}",
                timeout=4.0,
            )
            for picker_id in picker_ids:
                assert screen.query_one(f"#{picker_id}", Button)

        configure = screen.query_one("#settings-speech-configure-provider", Select)
        configure.value = "audio_cpp"
        await _wait_for_selector(
            screen, pilot, "#settings-speech-provider-audio_cpp", timeout=4.0
        )
        audio_text = " ".join(
            node.renderable.plain
            if hasattr(node.renderable, "plain")
            else str(node.renderable)
            for node in screen.query("#settings-speech-provider-audio_cpp Static")
        ).lower()
        for forbidden in (
            "binary path",
            "server.json",
            "bind address",
            "launch server",
            "restart server",
            "supervise",
            "stop server",
        ):
            assert forbidden not in audio_text


@pytest.mark.asyncio
async def test_normal_panel_actions_do_not_contact_or_initialize_tts(
    monkeypatch,
) -> None:
    calls: list[str] = []

    async def forbidden_service(*_args, **_kwargs):
        calls.append("service")
        raise AssertionError("normal Settings action contacted TTS runtime")

    monkeypatch.setattr(
        "tldw_chatbook.TTS.TTS_Generation.get_tts_service",
        forbidden_service,
    )
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = await _open_speech_tts(host, pilot)
        screen.query_one("#settings-speech-model-value", Input).value = "draft-model"
        await pilot.click("#settings-speech-restore-defaults")
        await pilot.pause()
        await pilot.click("#settings-speech-revert")
        await pilot.pause()

    assert calls == []


@pytest.mark.asyncio
async def test_save_posts_validated_non_secret_atomic_proposal() -> None:
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 60)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        app.query_one(
            "#settings-speech-audio_cpp-synthesis-timeout-seconds", Input
        ).value = "321"
        await pilot.click("#settings-speech-save")
        await pilot.pause()

        assert len(app.events) == 1
        event = app.events[0]
        assert event.settings["audio_cpp"]["synthesis_timeout_seconds"] == 321.0
        assert "openai_api_key" not in event.settings
        assert "elevenlabs_api_key" not in event.settings
        assert event.reply_to is panel
        assert event.request_id == 1

        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=1,
                persisted=True,
                provider_statuses={"audio_cpp": "applied"},
            )
        )
        assert (
            "Saved locally"
            in app.query_one("#settings-speech-save-result", Static).renderable
        )
        assert (
            "audio.cpp: applied"
            in app.query_one("#settings-speech-save-result", Static).renderable
        )


@pytest.mark.asyncio
async def test_cache_reload_failure_keeps_persistence_and_runtime_results_distinct() -> (
    None
):
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 60)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        app.query_one(
            "#settings-speech-audio_cpp-synthesis-timeout-seconds",
            Input,
        ).value = "321"
        await pilot.click("#settings-speech-save")
        await pilot.pause()

        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=1,
                persisted=True,
                provider_statuses={},
                failure_phase="cache_reload",
            )
        )

        result = str(app.query_one("#settings-speech-save-result", Static).renderable)
        assert "Saved locally" in result
        assert "cache reload failed" in result
        assert panel.has_unsaved_changes() is False


@pytest.mark.asyncio
async def test_invalid_save_is_field_specific_and_posts_no_event() -> None:
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 60)) as pilot:
        invalid = "https://example.invalid/not-an-origin"
        app.query_one("#settings-speech-audio_cpp-base-url", Input).value = invalid
        await pilot.click("#settings-speech-save")
        await pilot.pause()

        assert app.events == []
        error = app.query_one(
            "#settings-speech-audio_cpp-base-url-error", Static
        ).renderable
        assert error
        assert invalid not in str(error)


@pytest.mark.asyncio
async def test_invalid_global_default_renders_an_adjacent_field_error() -> None:
    app = _PanelHarness(configure_provider="openai")
    async with app.run_test(size=(150, 60)) as pilot:
        app.query_one("#settings-speech-speed", Input).value = "not-a-speed"
        await pilot.click("#settings-speech-save")
        await pilot.pause()

        assert app.events == []
        error = app.query_one("#settings-speech-speed-error", Static).renderable
        assert error
        assert "not-a-speed" not in str(error)


@pytest.mark.asyncio
async def test_credential_operations_are_separate_from_ordinary_save() -> None:
    app = _PanelHarness(configure_provider="openai")
    async with app.run_test(size=(150, 60)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)

        panel.submit_credential_mutation(
            "openai",
            CredentialIntent.SET,
            "synthetic-test-credential",
        )
        await pilot.pause()

        assert len(app.events) == 1
        event = app.events[0]
        assert event.settings == {"openai_api_key": "synthetic-test-credential"}
        assert event.delete_setting_keys == ()
        assert event.preferences is None

        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=1,
                persisted=True,
                provider_statuses={"openai": "unchanged"},
            )
        )
        panel.submit_credential_mutation(
            "openai",
            CredentialIntent.CLEAR,
            None,
        )
        await pilot.pause()

        assert app.events[1].settings == {}
        assert app.events[1].delete_setting_keys == ("openai_api_key",)


@pytest.mark.asyncio
async def test_overlapping_save_is_blocked_without_losing_the_pending_baseline() -> (
    None
):
    app = _PanelHarness(configure_provider="openai")
    async with app.run_test(size=(150, 80)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        organization = app.query_one(
            "#settings-speech-openai-organization-id",
            Input,
        )
        organization.value = "org-first"
        panel.request_save()
        await pilot.pause()
        assert len(app.events) == 1
        assert app.query_one("#settings-speech-save", Button).disabled is True

        organization.value = "org-second"
        panel.request_save()
        await pilot.pause()

        assert len(app.events) == 1
        assert (
            "already in progress"
            in str(
                app.query_one("#settings-speech-save-result", Static).renderable
            ).lower()
        )

        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=1,
                persisted=True,
                provider_statuses={"openai": "applied"},
            )
        )
        assert app.query_one("#settings-speech-save", Button).disabled is False
        assert panel.has_unsaved_changes() is True

        await panel.revert_to_saved()
        await pilot.pause()
        assert (
            app.query_one(
                "#settings-speech-openai-organization-id",
                Input,
            ).value
            == "org-first"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ("kokoro", "higgs"))
async def test_existing_mps_device_is_a_valid_visible_selection(
    provider_id: str,
) -> None:
    raw: dict[str, object] = {
        "app_tts": {"KOKORO_DEVICE_DEFAULT": "mps"},
        "HiggsSettings": {"device": "mps"},
    }
    state = load_global_speech_tts_state(
        {"COMPREHENSIVE_CONFIG_RAW": raw},
        environment={},
    )
    app = _PanelHarness(configure_provider=provider_id, state=state)
    async with app.run_test(size=(150, 80)) as pilot:
        await pilot.pause()
        assert (
            app.query_one(
                f"#settings-speech-{provider_id}-device",
                Select,
            ).value
            == "mps"
        )


@pytest.mark.asyncio
async def test_successful_credential_mutation_refreshes_visible_actions() -> None:
    app = _PanelHarness(configure_provider="openai")
    async with app.run_test(size=(150, 120)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        panel.submit_credential_mutation(
            "openai",
            CredentialIntent.SET,
            "synthetic-test-credential",
        )
        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=1,
                persisted=True,
                provider_statuses={"openai": "unchanged"},
            )
        )
        await pilot.pause()

        assert (
            str(
                app.query_one(
                    "#settings-speech-openai-credential-edit",
                    Button,
                ).label
            )
            == "Replace credential"
        )
        assert app.query_one(
            "#settings-speech-openai-credential-clear",
            Button,
        )


@pytest.mark.asyncio
async def test_environment_credential_is_read_only_and_editor_starts_empty() -> None:
    state = load_global_speech_tts_state(
        {
            "COMPREHENSIVE_CONFIG_RAW": {
                "API": {"openai_api_key": "synthetic-local-fallback"}
            }
        },
        environment={"OPENAI_API_KEY": "synthetic-environment-value"},
    )
    app = _PanelHarness(configure_provider="openai", state=state)
    async with app.run_test(size=(150, 120)) as pilot:
        credential = app.query_one("#settings-speech-openai-credential")
        rendered = " ".join(str(node.render()) for node in credential.query(Static))
        assert "Environment" in rendered
        assert "read-only" in rendered
        assert "shadowed" in rendered
        assert "synthetic-local-fallback" not in rendered
        assert "synthetic-environment-value" not in rendered

        edit = app.query_one("#settings-speech-openai-credential-edit", Button)
        edit.scroll_visible(animate=False)
        await pilot.pause()
        await pilot.click("#settings-speech-openai-credential-edit")
        await pilot.pause()

        editor = app.screen.query_one(
            "#settings-speech-credential-new-value",
            Input,
        )
        assert editor.value == ""
        assert editor.password is True
        await pilot.press("escape")


@pytest.mark.asyncio
async def test_credential_editor_is_a_bounded_modal_with_real_styles() -> None:
    app = _StyledPanelHarness(configure_provider="openai")
    async with app.run_test(size=(120, 40)) as pilot:
        app.query_one(
            "#settings-speech-openai-credential-edit",
            Button,
        ).press()
        await pilot.pause()

        card = app.screen.query_one(".settings-speech-credential-modal")
        assert 40 <= card.region.width <= 80
        assert card.region.height < pilot.app.size.height
        assert card.region.x > 0
        assert card.region.y > 0


@pytest.mark.asyncio
async def test_panel_tracks_dirty_state_and_revert_restores_the_saved_snapshot() -> (
    None
):
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 60)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        assert panel.has_unsaved_changes() is False

        app.query_one(
            "#settings-speech-audio_cpp-synthesis-timeout-seconds",
            Input,
        ).value = "321"
        await pilot.pause()
        assert panel.has_unsaved_changes() is True

        await panel.revert_to_saved()
        assert panel.has_unsaved_changes() is False


@pytest.mark.asyncio
async def test_revert_clears_local_validation_errors() -> None:
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 60)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        app.query_one(
            "#settings-speech-audio_cpp-base-url",
            Input,
        ).value = "https://example.invalid/not-an-origin"
        panel.request_save()
        assert app.query_one(
            "#settings-speech-audio_cpp-base-url-error",
            Static,
        ).renderable

        await panel.revert_to_saved()
        await pilot.pause()

        assert not app.query_one(
            "#settings-speech-audio_cpp-base-url-error",
            Static,
        ).renderable


@pytest.mark.asyncio
async def test_restore_defaults_clears_replaced_validation_errors() -> None:
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 60)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        app.query_one(
            "#settings-speech-audio_cpp-base-url",
            Input,
        ).value = "https://example.invalid/not-an-origin"
        panel.request_save()
        assert app.query_one(
            "#settings-speech-audio_cpp-base-url-error",
            Static,
        ).renderable

        await pilot.click("#settings-speech-restore-defaults")
        await pilot.pause()

        assert not app.query_one(
            "#settings-speech-audio_cpp-base-url-error",
            Static,
        ).renderable


@pytest.mark.asyncio
async def test_successful_save_marks_only_the_published_provider_snapshot_saved() -> (
    None
):
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 60)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        audio_timeout = app.query_one(
            "#settings-speech-audio_cpp-synthesis-timeout-seconds",
            Input,
        )
        audio_timeout.value = "321"

        configure = app.query_one("#settings-speech-configure-provider", Select)
        configure.value = "openai"
        await pilot.pause()
        app.query_one(
            "#settings-speech-openai-organization-id", Input
        ).value = "org-new"
        await pilot.click("#settings-speech-save")
        await pilot.pause()

        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=1,
                persisted=True,
                provider_statuses={"openai": "applied"},
            )
        )
        configure = app.query_one("#settings-speech-configure-provider", Select)
        configure.value = "audio_cpp"
        await pilot.pause()

        assert panel.has_unsaved_changes() is True
        assert (
            app.query_one(
                "#settings-speech-audio_cpp-synthesis-timeout-seconds",
                Input,
            ).value
            == "321"
        )


@pytest.mark.asyncio
async def test_settings_generic_save_and_revert_actions_route_to_speech_panel() -> None:
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = await _open_speech_tts(host, pilot)
        panel = screen.query_one(
            "#settings-speech-tts-panel",
            SpeechTTSSettingsPanel,
        )
        model_value = screen.query_one("#settings-speech-model-value", Input)
        model_value.value = f"{model_value.value}-draft"
        await pilot.pause()
        request_save = Mock()
        revert_to_saved = AsyncMock()
        panel.request_save = request_save
        panel.revert_to_saved = revert_to_saved

        screen.action_settings_save_category(allow_text_entry_focus=True)
        screen.action_settings_revert_category(allow_text_entry_focus=True)
        await _settle(pilot)

        request_save.assert_called_once_with()
        revert_to_saved.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_unsaved_speech_draft_survives_settings_category_recompose() -> None:
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = await _open_speech_tts(host, pilot)
        model = screen.query_one("#settings-speech-model-value", Input)
        model.value = "unsaved-exact-model"
        await pilot.pause()

        await pilot.click("#settings-category-overview")
        await pilot.pause()
        await pilot.click("#settings-category-speech-tts")
        await _wait_for_selector(
            screen,
            pilot,
            "#settings-speech-model-value",
            timeout=8.0,
        )

        panel = screen.query_one(
            "#settings-speech-tts-panel",
            SpeechTTSSettingsPanel,
        )
        assert screen.query_one("#settings-speech-model-value", Input).value == (
            "unsaved-exact-model"
        )
        assert panel.has_unsaved_changes() is True
