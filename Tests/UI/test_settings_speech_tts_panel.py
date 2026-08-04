from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Select, Static, Switch

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _visible_text,
    _wait_for_selector,
)
from tldw_chatbook.Chat.console_voice_input import (
    DEFAULT_REALTIME_IDLE_TIMEOUT_MINUTES,
    DEFAULT_REALTIME_MODEL,
    DEFAULT_REALTIME_PROVIDER,
)
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
    STTSSettingsSaveResult,
)
from tldw_chatbook import config as config_module
from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    TTSModelInfo,
    TTSNativeCapabilityObservation,
    TTSNativeCapabilitySnapshot,
    TTSProviderCatalog,
    TTSVoiceDiscoveryResult,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Lab_Modules import lab_speech_status as lab_speech_status_module
from tldw_chatbook.UI.Screens import settings_screen as settings_screen_module
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.UI.Screens.settings_speech_tts import (
    BUILT_IN_TTS_PROVIDER_ORDER,
    GLOBAL_TTS_PROVIDER_FIELD_IDS,
    CredentialIntent,
    GlobalSpeechTTSEffectiveSource,
    load_global_speech_tts_state,
)
from tldw_chatbook.UI.Speech.speech_settings_contracts import (
    SpeechTTSNavigationIntent,
    SpeechTTSNavigationTarget,
    SpeechTTSRuntimeState,
    SpeechTTSRuntimeStatus,
    SpeechTTSStatusFreshness,
)
from tldw_chatbook.UI.Speech.speech_runtime_status import (
    SpeechLocalDependencyAvailability,
    SpeechTTSRuntimeStatusStore,
)
from tldw_chatbook.Widgets.Settings_Widgets import (
    speech_tts_settings_panel as speech_tts_settings_panel_module,
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
        observation: TTSNativeCapabilityObservation | None = None,
        current_configuration_revision: int | None = None,
        runtime_status_store: SpeechTTSRuntimeStatusStore | None = None,
    ) -> None:
        super().__init__()
        self.configure_provider = configure_provider
        self.state = state or load_global_speech_tts_state({})
        self.observation = observation
        self.current_configuration_revision = current_configuration_revision
        self.runtime_status_store = runtime_status_store
        self.events: list[STTSSettingsSaveEvent] = []
        self.navigation: list[NavigateToScreen] = []

    def compose(self) -> ComposeResult:
        yield SpeechTTSSettingsPanel(
            state=self.state,
            configure_provider=self.configure_provider,
            audio_cpp_observation=self.observation,
            audio_cpp_configuration_revision=self.current_configuration_revision,
            runtime_status_store=self.runtime_status_store,
            id="panel",
        )

    @on(STTSSettingsSaveEvent)
    def record_save(self, event: STTSSettingsSaveEvent) -> None:
        self.events.append(event)
        event.stop()

    @on(NavigateToScreen)
    def record_navigation(self, event: NavigateToScreen) -> None:
        self.navigation.append(event)
        event.stop()


def _audio_cpp_observation() -> TTSNativeCapabilityObservation:
    catalog = TTSProviderCatalog(
        provider_id="audio_cpp",
        revision=7,
        health=ProviderHealth(state="available", fresh=True),
        models=(
            TTSModelInfo(
                model_id="model-a",
                display_name="Model A",
                family="fake",
                upstream_mode="tts",
                formats=("wav",),
                voices=(),
                supports_speed=False,
                omit_voice_uses_server_default=True,
            ),
            TTSModelInfo(
                model_id="model-b",
                display_name="Model B",
                family="fake",
                upstream_mode="tts",
                formats=("wav",),
                voices=(),
                supports_speed=False,
                omit_voice_uses_server_default=True,
            ),
        ),
    )
    return TTSNativeCapabilityObservation(
        snapshot=TTSNativeCapabilitySnapshot(
            provider_id="audio_cpp",
            configuration_revision=4,
            state="unverified",
            catalog=catalog,
            voice_results={
                "model-a": TTSVoiceDiscoveryResult(
                    provider_id="audio_cpp",
                    model_id="model-a",
                    catalog_revision=7,
                    voices=("voice-a", "voice-b"),
                    state="complete",
                )
            },
        ),
        observed_at=datetime(2026, 8, 1, 12, 30, tzinfo=timezone.utc),
    )


def _audio_cpp_state(
    *,
    model_mode: str = "first_available",
    model_id: str | None = None,
    voice_mode: str = "server_default",
    voice_id: str | None = None,
    saved_provider: bool = False,
) -> object:
    state = load_global_speech_tts_state({})
    state.defaults.provider_id = "audio_cpp"
    state.defaults.model_mode = model_mode
    state.defaults.model_id = model_id
    state.defaults.voice_mode = voice_mode
    state.defaults.voice_id = voice_id
    state.defaults.response_format = "wav"
    state.defaults.speed = 1.0
    if saved_provider:
        state.provider_sources["audio_cpp"] = GlobalSpeechTTSEffectiveSource.SAVED_LOCAL
    return state


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


@pytest.mark.asyncio
async def test_production_settings_actions_cross_the_pushed_screen_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings save and Lab actions must reach the production App."""

    exact_settings = {
        "COMPREHENSIVE_CONFIG_RAW": {
            "app_tts": {
                "default_provider": "audio_cpp",
                "default_model_mode": "exact",
                "default_model": "supertonic-3",
                "default_voice_mode": "exact",
                "default_voice": "F1",
                "default_format": "wav",
                "default_speed": 1.0,
            }
        }
    }
    monkeypatch.setattr(
        settings_screen_module,
        "get_runtime_config_snapshot",
        lambda: config_module.RuntimeConfigSnapshot(0, exact_settings),
    )
    persisted: list[tuple[object, object]] = []

    def apply_settings(sets, *, delete_keys):
        persisted.append((sets, delete_keys))
        return config_module.ConfigMutationResult(True, True, None)

    monkeypatch.setattr(
        config_module,
        "apply_settings_mutation_to_cli_config",
        apply_settings,
    )
    app = _build_test_app(configured_default="settings")

    async with app.run_test(size=(190, 55)) as pilot:
        for _ in range(200):
            if isinstance(app.screen, SettingsScreen):
                break
            await pilot.pause(0.01)
        else:
            raise AssertionError("production app did not mount Settings")
        screen = app.screen
        await _wait_for_selector(
            screen,
            pilot,
            "#settings-category-speech-tts",
        )
        screen.query_one("#settings-category-speech-tts", Button).press()
        await _wait_for_selector(
            screen,
            pilot,
            "#settings-speech-tts-panel",
            timeout=8.0,
        )
        panel = screen.query_one(
            "#settings-speech-tts-panel",
            SpeechTTSSettingsPanel,
        )
        screen.query_one(
            "#settings-speech-model-policy", Select
        ).value = "first_available"
        screen.query_one(
            "#settings-speech-voice-policy", Select
        ).value = "server_default"
        await pilot.pause()

        screen.query_one("#settings-save-category", Button).press()
        for _ in range(300):
            if persisted and panel._latest_request_id is None:
                break
            await pilot.pause(0.01)

        assert persisted
        assert panel._latest_request_id is None
        sets, deletes = persisted[0]
        assert sets["app_tts"]["default_model_mode"] == "first_available"
        assert sets["app_tts"]["default_voice_mode"] == "server_default"
        assert set(deletes["app_tts"]) == {"default_model", "default_voice"}

        screen.query_one("#settings-speech-open-lab-bottom", Button).press()
        for _ in range(300):
            if getattr(app.screen, "screen_name", None) == "stts":
                break
            await pilot.pause(0.01)

        assert getattr(app.screen, "screen_name", None) == "stts"
        for _ in range(300):
            navigating = [
                worker
                for worker in app.workers
                if worker.group == "screen-navigation" and not worker.is_finished
            ]
            if not navigating:
                break
            await pilot.pause(0.01)


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
async def test_bounded_speech_settings_deep_link_restores_provider_without_action() -> (
    None
):
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        screen.apply_navigation_context(
            {
                "category": "speech-tts",
                "provider": "audio_cpp",
                "intent": "configure",
            }
        )
        await _wait_for_selector(
            screen,
            pilot,
            "#settings-speech-audio_cpp-base-url",
            timeout=8.0,
        )
        await _settle(pilot)

        panel = screen.query_one(
            "#settings-speech-tts-panel",
            SpeechTTSSettingsPanel,
        )
        assert panel.configure_provider == "audio_cpp"
        assert (
            screen.query_one("#settings-speech-configure-provider", Select).value
            == "audio_cpp"
        )
        assert getattr(host.focused, "id", None) == (
            "settings-speech-audio_cpp-base-url"
        )
        assert panel._latest_request_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ("deep-link", "search"))
async def test_dirty_same_category_provider_target_cancel_survives_recompose(
    trigger: str,
) -> None:
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = await _open_speech_tts(host, pilot)
        configure = screen.query_one("#settings-speech-configure-provider", Select)
        if configure.value != "openai":
            configure.value = "openai"
            await _wait_for_selector(
                screen,
                pilot,
                "#settings-speech-openai-organization-id",
            )
        panel = screen.query_one(
            "#settings-speech-tts-panel",
            SpeechTTSSettingsPanel,
        )
        organization = screen.query_one(
            "#settings-speech-openai-organization-id",
            Input,
        )
        organization.value = "unsaved-organization"
        organization.focus()
        await pilot.pause()
        panel._ask_leave_choice = AsyncMock(return_value="cancel")

        if trigger == "deep-link":
            screen.apply_navigation_context(
                {
                    "category": "speech-tts",
                    "provider": "kokoro",
                    "intent": "configure",
                }
            )
        else:
            screen._submit_category_search("Kokoro")
        await _settle(pilot)

        assert panel.configure_provider == "openai"
        assert screen._speech_tts_configure_provider == "openai"
        assert organization.value == "unsaved-organization"

        await screen.recompose()
        await _wait_for_selector(
            screen,
            pilot,
            "#settings-speech-openai-organization-id",
        )

        remounted = screen.query_one(
            "#settings-speech-tts-panel",
            SpeechTTSSettingsPanel,
        )
        assert remounted.configure_provider == "openai"
        assert (
            screen.query_one(
                "#settings-speech-openai-organization-id",
                Input,
            ).value
            == "unsaved-organization"
        )


@pytest.mark.parametrize(
    "extra",
    (
        {"api_key": "private"},
        {"text": "private synthesis text"},
        {"url": "https://user:secret@example.invalid"},
    ),
)
def test_speech_settings_deep_link_rejects_extra_payload(
    extra: dict[str, str],
) -> None:
    screen = SettingsScreen(_build_test_app())
    context: dict[str, object] = {
        "category": "speech-tts",
        "provider": "audio_cpp",
        "intent": "configure",
        **extra,
    }

    screen.apply_navigation_context(context)

    assert screen.active_category == SettingsCategoryId.OVERVIEW.value
    assert screen._speech_tts_configure_provider is None


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
async def test_audio_cpp_first_run_offers_only_dynamic_default_policies() -> None:
    app = _PanelHarness(state=_audio_cpp_state())

    async with app.run_test(size=(150, 60)):
        model_policy = app.query_one("#settings-speech-model-policy", Select)
        voice_policy = app.query_one("#settings-speech-voice-policy", Select)

        assert model_policy._legal_values == {"first_available"}
        assert voice_policy._legal_values == {"server_default"}
        assert app.query_one("#settings-speech-model-value", Select).disabled is True
        assert app.query_one("#settings-speech-voice-value", Select).disabled is True
        status = str(
            app.query_one("#settings-speech-audio-cpp-choice-status", Static).renderable
        )
        assert "Model: Not observed" in status
        assert "Voice: Not observed" in status


@pytest.mark.asyncio
async def test_first_switch_to_audio_cpp_does_not_pin_another_providers_exact_ids() -> (
    None
):
    state = load_global_speech_tts_state({})
    assert state.defaults.provider_id == "openai"
    assert state.defaults.model_mode == "exact"
    assert state.defaults.voice_mode == "exact"
    app = _PanelHarness(state=state)

    async with app.run_test(size=(150, 60)) as pilot:
        app.query_one("#settings-speech-default-provider", Select).value = "audio_cpp"
        await pilot.pause()

        model_policy = app.query_one("#settings-speech-model-policy", Select)
        voice_policy = app.query_one("#settings-speech-voice-policy", Select)
        assert model_policy.value == "first_available"
        assert model_policy._legal_values == {"first_available"}
        assert voice_policy.value == "server_default"
        assert voice_policy._legal_values == {"server_default"}
        assert app.query_one("#settings-speech-model-value", Select).disabled is True
        assert app.query_one("#settings-speech-voice-value", Select).disabled is True


@pytest.mark.asyncio
async def test_returning_to_audio_cpp_restores_its_persisted_exact_ids() -> None:
    app = _PanelHarness(
        state=_audio_cpp_state(
            model_mode="exact",
            model_id="saved-model",
            voice_mode="exact",
            voice_id="saved-voice",
        )
    )

    async with app.run_test(size=(150, 60)) as pilot:
        provider = app.query_one("#settings-speech-default-provider", Select)
        provider.value = "openai"
        await pilot.pause()
        app.query_one("#settings-speech-default-provider", Select).value = "audio_cpp"
        await pilot.pause()

        assert app.query_one("#settings-speech-model-policy", Select).value == "exact"
        assert app.query_one("#settings-speech-model-value", Select).value == (
            "saved-model"
        )
        assert app.query_one("#settings-speech-voice-policy", Select).value == "exact"
        assert app.query_one("#settings-speech-voice-value", Select).value == (
            "saved-voice"
        )


@pytest.mark.asyncio
async def test_audio_cpp_cached_choices_are_revisioned_and_model_scoped() -> None:
    app = _PanelHarness(
        state=_audio_cpp_state(
            model_mode="exact",
            model_id="model-a",
            voice_mode="exact",
            voice_id="voice-a",
        ),
        observation=_audio_cpp_observation(),
        current_configuration_revision=4,
    )

    async with app.run_test(size=(150, 60)):
        model = app.query_one("#settings-speech-model-value", Select)
        voice = app.query_one("#settings-speech-voice-value", Select)

        assert model.value == "model-a"
        assert {value for _label, value in model._options} >= {
            "model-a",
            "model-b",
        }
        assert voice.value == "voice-a"
        assert {value for _label, value in voice._options} >= {
            "voice-a",
            "voice-b",
        }
        status = str(
            app.query_one("#settings-speech-audio-cpp-choice-status", Static).renderable
        )
        assert "Model: Fresh" in status
        assert "Voice: Fresh" in status
        provenance = str(
            app.query_one(
                "#settings-speech-audio-cpp-observation-provenance", Static
            ).renderable
        )
        assert "configuration revision 4" in provenance
        assert "catalog revision 7" in provenance
        output_format = app.query_one("#settings-speech-output-format", Select)
        speed = app.query_one("#settings-speech-speed", Input)
        assert output_format.value == "wav"
        assert output_format.disabled is True
        assert speed.value == "1.0"
        assert speed.disabled is True
        assert "requires WAV output and speed 1.0" in str(
            app.query_one("#settings-speech-default-constraints", Static).renderable
        )


@pytest.mark.asyncio
async def test_audio_cpp_persisted_exact_values_remain_pinned_unverified() -> None:
    app = _PanelHarness(
        state=_audio_cpp_state(
            model_mode="exact",
            model_id="saved-model",
            voice_mode="exact",
            voice_id="saved-voice",
        )
    )

    async with app.run_test(size=(150, 60)):
        assert app.query_one("#settings-speech-model-value", Select).value == (
            "saved-model"
        )
        assert app.query_one("#settings-speech-voice-value", Select).value == (
            "saved-voice"
        )
        status = str(
            app.query_one("#settings-speech-audio-cpp-choice-status", Static).renderable
        )
        assert "Model: Unverified" in status
        assert "Voice: Unverified" in status


@pytest.mark.asyncio
async def test_audio_cpp_stale_observation_is_visibly_distinct() -> None:
    app = _PanelHarness(
        state=_audio_cpp_state(
            model_mode="exact",
            model_id="model-a",
            voice_mode="exact",
            voice_id="voice-a",
        ),
        observation=_audio_cpp_observation(),
        current_configuration_revision=5,
    )

    async with app.run_test(size=(150, 60)):
        status = str(
            app.query_one("#settings-speech-audio-cpp-choice-status", Static).renderable
        )
        assert "Model: Stale" in status
        assert "Voice: Stale" in status


@pytest.mark.asyncio
async def test_audio_cpp_authoritative_missing_model_stays_visible() -> None:
    app = _PanelHarness(
        state=_audio_cpp_state(
            model_mode="exact",
            model_id="missing-model",
            voice_mode="exact",
            voice_id="saved-voice",
        ),
        observation=_audio_cpp_observation(),
        current_configuration_revision=4,
    )

    async with app.run_test(size=(150, 60)):
        model = app.query_one("#settings-speech-model-value", Select)
        assert model.value == "missing-model"
        assert "Missing" in str(model._options[-1][0])
        status = str(
            app.query_one("#settings-speech-audio-cpp-choice-status", Static).renderable
        )
        assert "Model: Missing" in status
        assert "Voice: Unverified" in status


@pytest.mark.asyncio
async def test_audio_cpp_remote_http_warning_and_dirty_draft_attribution_update() -> (
    None
):
    app = _PanelHarness(state=_audio_cpp_state())

    async with app.run_test(size=(150, 70)) as pilot:
        base_url = app.query_one("#settings-speech-audio_cpp-base-url", Input)
        base_url.value = "http://remote.example.test:8080"
        await pilot.pause()

        warning = str(
            app.query_one(
                "#settings-speech-audio-cpp-transport-warning", Static
            ).renderable
        )
        attribution = str(
            app.query_one(
                "#settings-speech-audio-cpp-draft-attribution", Static
            ).renderable
        )
        assert "not transport-encrypted" in warning
        assert "submitted text" in warning
        assert "saved server configuration" in attribution
        assert "unsaved Server URL draft" in attribution

        base_url.value = "http://127.0.0.1:8080"
        await pilot.pause()
        assert not str(
            app.query_one(
                "#settings-speech-audio-cpp-transport-warning", Static
            ).renderable
        )
        assert "unsaved Server URL draft" not in str(
            app.query_one(
                "#settings-speech-audio-cpp-draft-attribution", Static
            ).renderable
        )


@pytest.mark.asyncio
async def test_audio_cpp_recovery_opens_speech_lab_playground_without_provider_work() -> (
    None
):
    app = _PanelHarness(state=_audio_cpp_state())

    async with app.run_test(size=(150, 60)) as pilot:
        app.query_one("#settings-speech-audio-cpp-open-lab", Button).press()
        await pilot.pause()

    assert len(app.navigation) == 1
    assert app.navigation[0].screen_name == "stts"
    assert app.navigation[0].screen_context == {
        "view": "playground",
        "provider": "audio_cpp",
        "intent": "refresh-models",
    }


@pytest.mark.asyncio
async def test_settings_inspector_keeps_audio_cpp_ready_when_local_deps_missing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        speech_tts_settings_panel_module,
        "speech_local_dependency_availability",
        Mock(
            return_value=SpeechLocalDependencyAvailability(
                stt=False,
                kokoro=False,
                chatterbox=False,
                higgs=False,
            )
        ),
    )
    app = _PanelHarness(
        state=_audio_cpp_state(saved_provider=True),
        observation=_audio_cpp_observation(),
        current_configuration_revision=4,
    )

    async with app.run_test(size=(150, 80)):
        assert "Saved" in str(
            app.query_one("#settings-speech-status-provider-configuration").render()
        )
        assert "Ready" in str(
            app.query_one("#settings-speech-status-provider-runtime").render()
        )
        assert "Ready" in str(
            app.query_one("#settings-speech-status-catalog-freshness").render()
        )
        for row_id in (
            "stt-dependency",
            "kokoro-dependency",
            "chatterbox-dependency",
            "higgs-dependency",
        ):
            assert "Unavailable" in str(
                app.query_one(f"#settings-speech-status-{row_id}").render()
            )


@pytest.mark.asyncio
async def test_first_settings_mount_probes_local_speech_dependencies_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed_modules: list[str] = []

    def find_spec(module_name: str):
        probed_modules.append(module_name)
        if module_name == "nemo_toolkit":
            return None
        return SimpleNamespace(name=module_name)

    monkeypatch.setattr(
        lab_speech_status_module,
        "find_spec",
        find_spec,
    )
    import_probe = Mock(side_effect=AssertionError("must not import local runtimes"))
    monkeypatch.setattr(lab_speech_status_module, "check_tts_deps", import_probe)
    monkeypatch.setattr(lab_speech_status_module, "check_stt_deps", import_probe)
    app = _PanelHarness(state=_audio_cpp_state(saved_provider=True))

    async with app.run_test(size=(150, 80)):
        for row_id in (
            "stt-dependency",
            "kokoro-dependency",
            "chatterbox-dependency",
            "higgs-dependency",
        ):
            assert "Ready" in str(
                app.query_one(f"#settings-speech-status-{row_id}").render()
            )

    assert probed_modules == [
        "nemo_toolkit",
        "faster_whisper",
        "kokoro_onnx",
        "chatterbox",
        "boson_multimodal",
    ]
    import_probe.assert_not_called()


@pytest.mark.asyncio
async def test_dirty_global_link_cancel_preserves_draft_focus_and_navigation() -> None:
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 80)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        field = app.query_one(
            "#settings-speech-audio_cpp-synthesis-timeout-seconds",
            Input,
        )
        field.value = "321"
        field.focus()
        await pilot.pause()
        panel._ask_leave_choice = AsyncMock(return_value="cancel")

        allowed = await panel.confirm_leave()
        if allowed:
            await panel._open_lab(
                SpeechTTSNavigationTarget(
                    "audio_cpp",
                    SpeechTTSNavigationIntent.TEST,
                )
            )
        await pilot.pause()

        assert app.navigation == []
        assert field.value == "321"
        assert app.focused is field


@pytest.mark.asyncio
async def test_dirty_global_link_failed_save_keeps_owner_draft_and_focus() -> None:
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 80)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        field = app.query_one(
            "#settings-speech-audio_cpp-synthesis-timeout-seconds",
            Input,
        )
        field.value = "321"
        field.focus()
        panel._ask_leave_choice = AsyncMock(return_value="save")

        leave_task = asyncio.create_task(panel.confirm_leave())
        for _ in range(100):
            if app.events:
                break
            await pilot.pause(0.01)
        assert app.events
        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=app.events[0].request_id or 0,
                persisted=False,
                provider_statuses={},
                failure_phase="before_replace",
            )
        )
        allowed = await leave_task
        if allowed:
            await panel._open_lab(
                SpeechTTSNavigationTarget(
                    "audio_cpp",
                    SpeechTTSNavigationIntent.TEST,
                )
            )
        await pilot.pause()

        assert app.navigation == []
        assert field.value == "321"
        assert app.focused is field
        assert panel.has_unsaved_changes() is True


@pytest.mark.asyncio
async def test_dirty_global_link_successful_save_continues_with_bounded_intent() -> (
    None
):
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 80)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        app.query_one(
            "#settings-speech-audio_cpp-synthesis-timeout-seconds",
            Input,
        ).value = "321"
        panel._ask_leave_choice = AsyncMock(return_value="save")

        leave_task = asyncio.create_task(panel.confirm_leave())
        for _ in range(100):
            if app.events:
                break
            await pilot.pause(0.01)
        assert app.events
        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=app.events[0].request_id or 0,
                persisted=True,
                provider_statuses={"audio_cpp": "applied"},
                provider_configuration_revisions={"audio_cpp": 5},
                provider_runtime_revisions={"audio_cpp": 9},
            )
        )
        allowed = await leave_task
        if allowed:
            await panel._open_lab(
                SpeechTTSNavigationTarget(
                    "audio_cpp",
                    SpeechTTSNavigationIntent.TEST,
                )
            )
        await pilot.pause()

        assert panel.has_unsaved_changes() is False
        assert len(app.navigation) == 1
        assert app.navigation[0].screen_context == {
            "view": "playground",
            "provider": "audio_cpp",
            "intent": "test",
        }


@pytest.mark.asyncio
async def test_every_leave_guard_waits_for_an_existing_save_instead_of_discarding() -> (
    None
):
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 80)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        field = app.query_one(
            "#settings-speech-audio_cpp-synthesis-timeout-seconds",
            Input,
        )
        field.value = "321"
        field.focus()
        request_id = panel.request_save()
        assert request_id is not None
        assert panel._latest_request_id == request_id
        panel._ask_leave_choice = AsyncMock(return_value="discard")

        leave_task = asyncio.create_task(panel.confirm_leave())
        await pilot.pause()

        assert leave_task.done() is False
        panel._ask_leave_choice.assert_not_awaited()
        assert field.value == "321"

        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=request_id,
                persisted=True,
                provider_statuses={"audio_cpp": "applied"},
                provider_configuration_revisions={"audio_cpp": 1},
                provider_runtime_revisions={"audio_cpp": 1},
            )
        )

        assert await leave_task is True
        panel._ask_leave_choice.assert_not_awaited()
        assert panel.has_unsaved_changes() is False


@pytest.mark.asyncio
async def test_dirty_provider_switch_discard_removes_hidden_draft() -> None:
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 80)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        original_timeout = panel.original_state.providers["audio_cpp"][
            "synthesis_timeout_seconds"
        ]
        app.query_one(
            "#settings-speech-audio_cpp-synthesis-timeout-seconds",
            Input,
        ).value = "321"
        panel._ask_leave_choice = AsyncMock(return_value="discard")

        app.query_one("#settings-speech-configure-provider", Select).value = "openai"
        await _wait_for_selector(
            app.screen,
            pilot,
            "#settings-speech-openai-organization-id",
        )

        assert panel.configure_provider == "openai"
        app.query_one("#settings-speech-configure-provider", Select).value = "audio_cpp"
        await _wait_for_selector(
            app.screen,
            pilot,
            "#settings-speech-audio_cpp-synthesis-timeout-seconds",
        )
        assert app.query_one(
            "#settings-speech-audio_cpp-synthesis-timeout-seconds",
            Input,
        ).value == str(original_timeout)


@pytest.mark.asyncio
async def test_settings_dismissal_cancel_preserves_dirty_speech_owner() -> None:
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = await _open_speech_tts(host, pilot)
        panel = screen.query_one(
            "#settings-speech-tts-panel",
            SpeechTTSSettingsPanel,
        )
        field = screen.query_one("#settings-speech-model-value", Input)
        field.value = "dismissal-draft"
        field.focus()
        panel._ask_leave_choice = AsyncMock(return_value="cancel")
        await pilot.pause()

        assert await screen.flush_pending_work() is False

        assert screen.active_category == SettingsCategoryId.SPEECH_TTS.value
        assert field.value == "dismissal-draft"
        assert host.focused is field


@pytest.mark.asyncio
async def test_saved_but_unavailable_and_reconfiguring_are_not_rendered_ready() -> None:
    app = _PanelHarness(
        configure_provider="audio_cpp",
        state=_audio_cpp_state(),
        current_configuration_revision=4,
    )
    async with app.run_test(size=(150, 80)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        timeout = app.query_one(
            "#settings-speech-audio_cpp-synthesis-timeout-seconds",
            Input,
        )
        timeout.value = "321"
        panel.request_save()
        await pilot.pause()
        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=1,
                persisted=True,
                provider_statuses={"audio_cpp": "pending"},
                provider_configuration_revisions={"audio_cpp": 5},
                provider_runtime_revisions={"audio_cpp": 9},
            )
        )
        await pilot.pause()

        assert "Saved" in str(
            app.query_one("#settings-speech-status-provider-configuration").render()
        )
        assert "Reconfiguring" in str(
            app.query_one("#settings-speech-status-provider-runtime").render()
        )

        panel.receive_stts_settings_runtime_result(
            STTSSettingsSaveResult(
                request_id=1,
                persisted=True,
                provider_statuses={"audio_cpp": "unavailable"},
                provider_configuration_revisions={"audio_cpp": 5},
                provider_runtime_revisions={"audio_cpp": 9},
            )
        )
        await pilot.pause()
        assert "Unavailable" in str(
            app.query_one("#settings-speech-status-provider-runtime").render()
        )

        timeout = app.query_one(
            "#settings-speech-audio_cpp-synthesis-timeout-seconds",
            Input,
        )
        timeout.value = "322"
        panel.request_save()
        await pilot.pause()
        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=2,
                persisted=True,
                provider_statuses={"audio_cpp": "unavailable"},
                provider_configuration_revisions={"audio_cpp": 6},
                provider_runtime_revisions={"audio_cpp": 10},
            )
        )
        await pilot.pause()

        assert "Saved" in str(
            app.query_one("#settings-speech-status-provider-configuration").render()
        )
        runtime = str(
            app.query_one("#settings-speech-status-provider-runtime").render()
        )
        assert "Unavailable" in runtime
        assert "Ready" not in runtime

        panel.receive_stts_settings_runtime_result(
            STTSSettingsSaveResult(
                request_id=1,
                persisted=True,
                provider_statuses={"audio_cpp": "applied"},
                provider_configuration_revisions={"audio_cpp": 5},
                provider_runtime_revisions={"audio_cpp": 9},
            )
        )
        assert "Unavailable" in str(
            app.query_one("#settings-speech-status-provider-runtime").render()
        )


@pytest.mark.asyncio
async def test_delayed_save_completion_cannot_replace_newer_lab_runtime_evidence() -> (
    None
):
    store = SpeechTTSRuntimeStatusStore()
    app = _PanelHarness(
        configure_provider="audio_cpp",
        state=_audio_cpp_state(saved_provider=True),
        current_configuration_revision=4,
        runtime_status_store=store,
    )
    async with app.run_test(size=(150, 80)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        app.query_one(
            "#settings-speech-audio_cpp-synthesis-timeout-seconds",
            Input,
        ).value = "321"
        panel.request_save()
        await pilot.pause()
        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=1,
                persisted=True,
                provider_statuses={"audio_cpp": "pending"},
                provider_configuration_revisions={"audio_cpp": 5},
                provider_runtime_revisions={"audio_cpp": 9},
            )
        )
        pending = panel._runtime_statuses["audio_cpp"]
        store.publish_runtime(
            SpeechTTSRuntimeStatus(
                provider_id="audio_cpp",
                saved_configuration_revision=5,
                runtime_revision=9,
                catalog_revision=None,
                model_scope=None,
                runtime_state=SpeechTTSRuntimeState.READY,
                observed_at=pending.observed_at + timedelta(seconds=1),
                freshness=SpeechTTSStatusFreshness.FRESH,
            )
        )

        panel.receive_stts_settings_runtime_result(
            STTSSettingsSaveResult(
                request_id=1,
                persisted=True,
                provider_statuses={"audio_cpp": "unavailable"},
                provider_configuration_revisions={"audio_cpp": 5},
                provider_runtime_revisions={"audio_cpp": 9},
            )
        )
        await pilot.pause()

        runtime = str(
            app.query_one("#settings-speech-status-provider-runtime").render()
        )
        assert "Ready" in runtime
        assert "Unavailable" not in runtime


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
async def test_settings_reads_only_existing_cached_audio_cpp_observation(
    monkeypatch,
) -> None:
    class CachedObservationOnlyService:
        def __init__(self) -> None:
            self.reads: list[tuple[str, str]] = []

        def latest_native_capability_observation(self, provider_id: str):
            self.reads.append(("observation", provider_id))
            return _audio_cpp_observation()

        def configuration_revision(self, provider_id: str) -> int:
            self.reads.append(("revision", provider_id))
            return 4

        async def get_catalog(self, *_args, **_kwargs):
            raise AssertionError("Settings must not discover a catalog")

        async def observe_voices(self, *_args, **_kwargs):
            raise AssertionError("Settings must not discover voices")

        async def ensure_ready(self, *_args, **_kwargs):
            raise AssertionError("Settings must not test a provider")

    monkeypatch.setattr(
        settings_screen_module,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(
            values={
                "app_tts": {
                    "default_provider": "audio_cpp",
                    "default_model_mode": "exact",
                    "default_model": "model-a",
                    "default_voice_mode": "exact",
                    "default_voice": "voice-a",
                    "default_format": "wav",
                    "default_speed": 1.0,
                }
            }
        ),
    )
    app = _build_test_app()
    service = CachedObservationOnlyService()
    app.tts_service = service
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        screen = await _open_speech_tts(host, pilot)

        assert screen.query_one("#settings-speech-model-value", Select).value == (
            "model-a"
        )
        assert screen.query_one("#settings-speech-voice-value", Select).value == (
            "voice-a"
        )

        assert service.reads
        assert set(service.reads) == {
            ("observation", "audio_cpp"),
            *(("revision", provider_id) for provider_id in BUILT_IN_TTS_PROVIDER_ORDER),
        }


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
                provider_configuration_revisions={"audio_cpp": 1},
                provider_runtime_revisions={"audio_cpp": 1},
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
                provider_configuration_revisions={"openai": 1},
                provider_runtime_revisions={"openai": 1},
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
                provider_configuration_revisions={"openai": 1},
                provider_runtime_revisions={"openai": 1},
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
                provider_configuration_revisions={"openai": 1},
                provider_runtime_revisions={"openai": 1},
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
        assert panel.state.provider_sources["openai"] is (
            GlobalSpeechTTSEffectiveSource.SAVED_LOCAL
        )
        assert panel.original_state.provider_sources["openai"] is (
            GlobalSpeechTTSEffectiveSource.SAVED_LOCAL
        )
        assert "Saved" in str(
            app.query_one("#settings-speech-status-provider-configuration").render()
        )
        assert "Saved local config" in str(
            app.query_one("#settings-speech-provider-source", Static).renderable
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
    # Unstyled `_PanelHarness` sections default to Textual's built-in `1fr`
    # Vertical sizing (no app stylesheet to declare `height: auto`), so each
    # top-level `.settings-focus-card` competes for a share of the viewport.
    # Task 6 added one more such section (Realtime engine); 120 rows is no
    # longer enough headroom for that fr-share split to keep this row's
    # rendered position matching its cached click region -- 150 restores
    # the margin `_StyledPanelHarness` (real CSS, `height: auto`) doesn't
    # need at all.
    async with app.run_test(size=(150, 150)) as pilot:
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


def test_environment_projected_credential_is_not_a_saved_local_fallback() -> None:
    """Normalized environment aliases must not be treated as persisted secrets."""

    state = load_global_speech_tts_state(
        {
            "COMPREHENSIVE_CONFIG_RAW": {
                "api_settings": {
                    "openai": {"api_key_env_var": "OPENAI_API_KEY"},
                }
            },
            # ``config.load_settings`` publishes this compatibility projection
            # after resolving the environment. It is effective runtime state,
            # not evidence that a local credential exists on disk.
            "openai_api": {"api_key": "synthetic-environment-value"},
        },
        environment={"OPENAI_API_KEY": "synthetic-environment-value"},
    )

    credential = state.credentials["openai"]
    assert credential.source.value == "Environment"
    assert credential.local_saved is False
    assert credential.local_shadowed is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_id", "field_id", "environment_variable"),
    (
        ("kokoro", "onnx-model-path", "KOKORO_MODEL_PATH"),
        ("kokoro", "voices-json-path", "KOKORO_VOICES_PATH"),
        ("higgs", "model-path", "HIGGS_MODEL_PATH"),
    ),
)
async def test_environment_owned_legacy_paths_are_labeled_and_read_only(
    provider_id: str,
    field_id: str,
    environment_variable: str,
) -> None:
    state = load_global_speech_tts_state(
        {},
        environment={environment_variable: "/environment/runtime-path"},
    )
    app = _PanelHarness(configure_provider=provider_id, state=state)

    async with app.run_test(size=(150, 120)):
        field = app.query_one(
            f"#settings-speech-{provider_id}-{field_id}",
            Input,
        )
        browse = app.query_one(
            f"#settings-speech-{provider_id}-{field_id}-browse",
            Button,
        )
        rendered = " ".join(
            str(node.render()) for node in app.query(".settings-detail-row")
        )

        assert field.disabled is True
        assert browse.disabled is True
        assert environment_variable in rendered
        assert "read-only" in rendered
        assert "/environment/runtime-path" not in rendered


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
async def test_dirty_provider_switch_cancel_preserves_visible_owner_and_draft() -> None:
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 60)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        audio_timeout = app.query_one(
            "#settings-speech-audio_cpp-synthesis-timeout-seconds",
            Input,
        )
        audio_timeout.value = "321"
        panel._ask_leave_choice = AsyncMock(return_value="cancel")

        configure = app.query_one("#settings-speech-configure-provider", Select)
        configure.value = "openai"
        await pilot.pause()
        await app.workers.wait_for_complete()

        assert panel.configure_provider == "audio_cpp"
        assert configure.value == "audio_cpp"
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
async def test_dirty_speech_category_cancel_preserves_owner_draft_and_focus() -> None:
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = await _open_speech_tts(host, pilot)
        panel = screen.query_one(
            "#settings-speech-tts-panel",
            SpeechTTSSettingsPanel,
        )
        model = screen.query_one("#settings-speech-model-value", Input)
        model.value = "unsaved-exact-model"
        model.focus()
        panel._ask_leave_choice = AsyncMock(return_value="cancel")
        await pilot.pause()

        screen.query_one("#settings-category-overview", Button).press()
        await screen.workers.wait_for_complete()
        await pilot.pause()

        assert screen.active_category == SettingsCategoryId.SPEECH_TTS.value
        assert screen.query_one("#settings-speech-model-value", Input).value == (
            "unsaved-exact-model"
        )
        assert host.focused is model
        assert panel.has_unsaved_changes() is True


# --- Realtime engine block (task 6) ----------------------------------------


@pytest.mark.asyncio
async def test_realtime_block_renders_with_config_defaults() -> None:
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 60)) as pilot:
        await _settle(pilot)
        enabled = app.query_one("#settings-speech-realtime-enabled", Switch)
        provider = app.query_one("#settings-speech-realtime-provider", Select)
        model = app.query_one("#settings-speech-realtime-model", Input)
        voice = app.query_one("#settings-speech-realtime-voice", Input)
        idle_timeout = app.query_one(
            "#settings-speech-realtime-idle-timeout-minutes", Input
        )
        engine = app.query_one("#settings-speech-realtime-handsfree-engine", Select)

        assert enabled.value is False
        assert provider.value == DEFAULT_REALTIME_PROVIDER
        assert [value for _label, value in provider._options] == [
            DEFAULT_REALTIME_PROVIDER
        ]
        assert model.value == DEFAULT_REALTIME_MODEL
        assert voice.value == ""
        assert idle_timeout.value == str(DEFAULT_REALTIME_IDLE_TIMEOUT_MINUTES)
        assert engine.value == "auto"


@pytest.mark.asyncio
async def test_realtime_toggle_and_save_writes_exact_keys_through_shared_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple, dict]] = []

    def _fake_save(*args, **kwargs) -> bool:
        calls.append((args, kwargs))
        return True

    monkeypatch.setattr(
        speech_tts_settings_panel_module,
        "save_settings_to_cli_config",
        _fake_save,
    )
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 60)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        await _settle(pilot)
        app.query_one("#settings-speech-realtime-enabled", Switch).value = True
        app.query_one("#settings-speech-realtime-model", Input).value = "gpt-realtime-mini"
        app.query_one("#settings-speech-realtime-voice", Input).value = "marin"
        app.query_one(
            "#settings-speech-realtime-idle-timeout-minutes", Input
        ).value = "8"
        app.query_one(
            "#settings-speech-realtime-handsfree-engine", Select
        ).value = "realtime"
        await pilot.pause()
        assert panel.has_unsaved_changes() is True

        await pilot.click("#settings-speech-save")
        await pilot.pause()

        # Only the realtime/dictation block changed -- no TTS provider event.
        assert app.events == []
        assert len(calls) == 1
        (section_values,), kwargs = calls[0]
        assert section_values == {
            "realtime": {
                "enabled": True,
                "provider": "openai",
                "model": "gpt-realtime-mini",
                "voice": "marin",
                "idle_timeout_minutes": 8.0,
            },
            "dictation": {"handsfree_engine": "realtime"},
        }
        assert kwargs["delete_keys"] == {}
        assert "Saved" in str(
            app.query_one("#settings-speech-save-result", Static).renderable
        )
        assert panel.has_unsaved_changes() is False


@pytest.mark.asyncio
async def test_realtime_blank_voice_deletes_key_instead_of_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        speech_tts_settings_panel_module,
        "save_settings_to_cli_config",
        lambda *a, **k: (calls.append((a, k)), True)[1],
    )
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 60)) as pilot:
        await _settle(pilot)
        app.query_one("#settings-speech-realtime-enabled", Switch).value = True
        await pilot.pause()

        await pilot.click("#settings-speech-save")
        await pilot.pause()

        assert len(calls) == 1
        (section_values,), kwargs = calls[0]
        assert "voice" not in section_values["realtime"]
        assert kwargs["delete_keys"] == {"realtime": ("voice",)}


@pytest.mark.asyncio
async def test_realtime_invalid_idle_timeout_refuses_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        speech_tts_settings_panel_module,
        "save_settings_to_cli_config",
        lambda *a, **k: (calls.append((a, k)), True)[1],
    )
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 60)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        await _settle(pilot)
        app.query_one(
            "#settings-speech-realtime-idle-timeout-minutes", Input
        ).value = "not-a-number"
        await pilot.pause()

        await pilot.click("#settings-speech-save")
        await pilot.pause()

        assert calls == []
        assert app.events == []
        error = app.query_one(
            "#settings-speech-realtime-idle-timeout-minutes-error", Static
        ).renderable
        assert error
        assert "not-a-number" not in str(error)
        assert panel.has_unsaved_changes() is True


@pytest.mark.asyncio
async def test_realtime_negative_idle_timeout_refuses_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        speech_tts_settings_panel_module,
        "save_settings_to_cli_config",
        lambda *a, **k: (calls.append((a, k)), True)[1],
    )
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 60)) as pilot:
        await _settle(pilot)
        app.query_one(
            "#settings-speech-realtime-idle-timeout-minutes", Input
        ).value = "-1"
        await pilot.pause()

        await pilot.click("#settings-speech-save")
        await pilot.pause()

        assert calls == []
        assert app.events == []
        error = app.query_one(
            "#settings-speech-realtime-idle-timeout-minutes-error", Static
        ).renderable
        assert error


@pytest.mark.asyncio
async def test_realtime_save_failure_surfaces_error_and_keeps_draft_dirty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        speech_tts_settings_panel_module,
        "save_settings_to_cli_config",
        lambda *a, **k: False,
    )
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 60)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        await _settle(pilot)
        app.query_one("#settings-speech-realtime-enabled", Switch).value = True
        await pilot.pause()

        await pilot.click("#settings-speech-save")
        await pilot.pause()

        assert panel.has_unsaved_changes() is True
        result = str(
            app.query_one("#settings-speech-save-result", Static).renderable
        )
        assert "not saved" in result.lower()


@pytest.mark.asyncio
async def test_realtime_block_dirty_state_and_revert() -> None:
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 60)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        await _settle(pilot)
        assert panel.has_unsaved_changes() is False

        app.query_one("#settings-speech-realtime-enabled", Switch).value = True
        await pilot.pause()
        assert panel.has_unsaved_changes() is True

        await panel.revert_to_saved()
        await pilot.pause()

        assert (
            app.query_one("#settings-speech-realtime-enabled", Switch).value is False
        )
        assert panel.has_unsaved_changes() is False


@pytest.mark.asyncio
async def test_realtime_and_tts_changes_save_together_in_one_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        speech_tts_settings_panel_module,
        "save_settings_to_cli_config",
        lambda *a, **k: (calls.append((a, k)), True)[1],
    )
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 60)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        await _settle(pilot)
        app.query_one("#settings-speech-realtime-enabled", Switch).value = True
        app.query_one(
            "#settings-speech-audio_cpp-synthesis-timeout-seconds", Input
        ).value = "321"
        await pilot.pause()

        await pilot.click("#settings-speech-save")
        await pilot.pause()

        assert len(calls) == 1
        assert len(app.events) == 1
        event = app.events[0]
        assert event.settings["audio_cpp"]["synthesis_timeout_seconds"] == 321.0

        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=1,
                persisted=True,
                provider_statuses={"audio_cpp": "applied"},
                provider_configuration_revisions={"audio_cpp": 1},
                provider_runtime_revisions={"audio_cpp": 1},
            )
        )
        assert panel.has_unsaved_changes() is False
