from __future__ import annotations

import asyncio
import builtins
import importlib
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, Collapsible, Input, Select, Static, Switch, TextArea

from Tests.UI.consolidated_css import app_css_text
from Tests.UI.speech_playground_fixtures import FakeTTSService, _resolved
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _visible_text,
    _wait_for_selector,
)
from tldw_chatbook import config as config_module
from tldw_chatbook.Chat.console_voice_input import (
    DEFAULT_REALTIME_IDLE_TIMEOUT_MINUTES,
    DEFAULT_REALTIME_MODEL,
    DEFAULT_REALTIME_PROVIDER,
)
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
    STTSSettingsSaveResult,
)
from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    TTSModelInfo,
    TTSNativeCapabilityObservation,
    TTSNativeCapabilitySnapshot,
    TTSProviderCatalog,
    TTSVoiceDiscoveryResult,
)
from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig
from tldw_chatbook.TTS.audio_cpp_supervisor import AudioCppSupervisor
from tldw_chatbook.TTS.openai_compatible_config import (
    normalize_openai_compatible_endpoint,
    openai_destination_fingerprint,
)
from tldw_chatbook.UI.Lab_Modules import lab_speech_status as lab_speech_status_module
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens import settings_screen as settings_screen_module
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.UI.Screens.settings_speech_tts import (
    BUILT_IN_TTS_PROVIDER_ORDER,
    GLOBAL_TTS_PROVIDER_FIELD_IDS,
    CredentialIntent,
    GlobalSpeechTTSEffectiveSource,
    GlobalSpeechTTSValidationError,
    ProcessProviderTestEvidenceStore,
    build_provider_test_fingerprint,
    load_global_speech_tts_state,
)
from tldw_chatbook.UI.Speech.speech_runtime_status import (
    SpeechLocalDependencyAvailability,
    SpeechTTSRuntimeStatusStore,
)
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.UI.Speech.speech_settings_contracts import (
    SpeechTTSConnectionState,
    SpeechTTSNavigationIntent,
    SpeechTTSNavigationTarget,
    SpeechTTSRuntimeState,
    SpeechTTSRuntimeStatus,
    SpeechTTSStatusFreshness,
)
from tldw_chatbook.Widgets.Settings_Widgets import (
    speech_tts_settings_panel as speech_tts_settings_panel_module,
)
from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
    SpeechTTSSettingsPanel,
)


_LOCAL_SPEECH_RUNTIME_IMPORT_ROOTS = frozenset(
    {
        "nemo",
        "faster_whisper",
        "lightning_whisper_mlx",
        "parakeet_mlx",
        "kokoro_onnx",
        "chatterbox",
        "boson_multimodal",
    }
)


def _install_local_runtime_import_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    import_attempts: list[str] = []
    real_import = builtins.__import__
    real_import_module = importlib.import_module

    def reject_local_runtime_import(name: str) -> None:
        if name.split(".", 1)[0] in _LOCAL_SPEECH_RUNTIME_IMPORT_ROOTS:
            import_attempts.append(name)
            raise AssertionError(f"must not import local runtime {name}")

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        reject_local_runtime_import(name)
        return real_import(name, globals, locals, fromlist, level)

    def guarded_import_module(name: str, package: str | None = None):
        reject_local_runtime_import(name)
        return real_import_module(name, package)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(importlib, "import_module", guarded_import_module)
    return import_attempts


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


async def _click_visible_button(screen, pilot, selector: str) -> None:
    """Scroll a panel action into the Pilot viewport before clicking it."""
    button = screen.query_one(selector, Button)
    button.scroll_visible(animate=False, immediate=True)
    await pilot.pause()
    await pilot.click(selector)


# Small fixed default so every pre-existing test (none of which cares about
# the default-voice-profile picker) keeps seeing a normal, populated Select
# instead of the store-unavailable state. Includes the UUID the default-
# profile tests below select, so `_PanelHarness(configure_provider="openai")`
# with no explicit `profiles=` keeps that id selectable out of the box.
_DEFAULT_TEST_PROFILES: tuple[tuple[str, str], ...] = (
    ("Test Voice", "3f2504e0-4f89-11d3-9a0c-0305e82c3301"),
)
# Distinguishes "caller didn't pass profiles" (-> the small fixed default
# above) from an explicit `profiles=None` (-> simulate an unavailable store).
_PROFILES_UNSET = object()


class _PanelHarness(App[None]):
    def __init__(
        self,
        *,
        configure_provider: str = "audio_cpp",
        state=None,
        profiles: object = _PROFILES_UNSET,
        profiles_unavailable: bool = False,
        observation: TTSNativeCapabilityObservation | None = None,
        current_configuration_revision: int | None = None,
        runtime_status_store: SpeechTTSRuntimeStatusStore | None = None,
        provider_test_evidence: ProcessProviderTestEvidenceStore | None = None,
    ) -> None:
        super().__init__()
        self.configure_provider = configure_provider
        self.state = state or load_global_speech_tts_state({})
        self.profiles = (
            list(_DEFAULT_TEST_PROFILES) if profiles is _PROFILES_UNSET else profiles
        )
        self.profiles_unavailable = profiles_unavailable
        self.observation = observation
        self.current_configuration_revision = current_configuration_revision
        self.runtime_status_store = runtime_status_store
        self.provider_test_evidence = provider_test_evidence
        self.events: list[STTSSettingsSaveEvent] = []
        self.navigation: list[NavigateToScreen] = []

    def compose(self) -> ComposeResult:
        yield SpeechTTSSettingsPanel(
            state=self.state,
            configure_provider=self.configure_provider,
            profiles=self.profiles,
            profiles_unavailable=self.profiles_unavailable,
            audio_cpp_observation=self.observation,
            audio_cpp_configuration_revision=self.current_configuration_revision,
            runtime_status_store=self.runtime_status_store,
            provider_test_evidence=self.provider_test_evidence,
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
_SETTINGS_SHEET = _BUNDLE.with_name("screen_agentic_settings.tcss")
_SETTINGS_CSS_PATH = [str(_BUNDLE), str(_SETTINGS_SHEET)]


class _StyledDestinationHarness(DestinationHarness):
    CSS_PATH = _SETTINGS_CSS_PATH


class _StyledPanelHarness(_PanelHarness):
    CSS_PATH = _SETTINGS_CSS_PATH


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
    monkeypatch.setattr(
        "tldw_chatbook.app.get_cli_setting",
        lambda section, key=None, default=None: (
            False if (section, key) == ("splash_screen", "enabled") else default
        ),
    )
    tts_service = FakeTTSService()
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(tts_service),
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
async def test_scope_banner_points_to_the_two_profile_surfaces() -> None:
    """The pointer card names where profiles actually live (task 1, slice 4).

    Voice profiles are a Speech Lab concept and per-character assignment
    lives in the Roleplay character editor (ADR-039 scope separation) --
    neither is managed from this panel. The card is a static note, not a
    control: it reuses the existing "Open Speech Lab" button rather than
    adding a second, competing affordance.
    """
    app = _PanelHarness()
    async with app.run_test(size=(150, 60)):
        note = app.query_one("#settings-speech-profile-surfaces-note", Static)
        assert str(note.renderable) == (
            "Voice profiles are managed in Lab > Speech > Voice Profiles — "
            "open Speech Lab, above, to get there. Per-character voices are "
            "assigned in the Roleplay character editor's Voice & Speech "
            "section, not here."
        )


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
        "intent": "test",
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
async def test_settings_connection_uses_sample_over_unsupported_catalog() -> None:
    state = _audio_cpp_state(saved_provider=True)
    fingerprint = build_provider_test_fingerprint(
        state,
        provider_id="audio_cpp",
        saved_revision=4,
    )
    evidence = ProcessProviderTestEvidenceStore()
    evidence.record_catalog(fingerprint, SpeechTTSConnectionState.UNSUPPORTED)
    pcm = struct.pack("<h", 100) * 32
    fmt = struct.pack("<HHIIHH", 1, 1, 16_000, 32_000, 2, 16)
    wave_body = (
        b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(pcm))
        + pcm
    )
    wav = b"RIFF" + struct.pack("<I", len(wave_body)) + wave_body
    assert evidence.record_successful_sample(
        fingerprint,
        status_code=200,
        response_format="wav",
        content_type="audio/wav",
        body=wav,
    )
    app = _PanelHarness(
        state=state,
        current_configuration_revision=4,
        provider_test_evidence=evidence,
    )

    async with app.run_test(size=(150, 80)):
        assert "Saved" in str(
            app.query_one("#settings-speech-status-provider-configuration").render()
        )
        connection = str(
            app.query_one("#settings-speech-status-provider-connection").render()
        )
        assert "reachable" in connection
        assert "catalog unsupported" in connection
        assert "sample reachable" in connection


@pytest.mark.asyncio
async def test_settings_connection_invalidates_evidence_at_changed_revision() -> None:
    state = _audio_cpp_state(saved_provider=True)
    fingerprint = build_provider_test_fingerprint(
        state,
        provider_id="audio_cpp",
        saved_revision=4,
    )
    evidence = ProcessProviderTestEvidenceStore()
    evidence.record_catalog(fingerprint, SpeechTTSConnectionState.REACHABLE)
    app = _PanelHarness(
        state=state,
        current_configuration_revision=5,
        provider_test_evidence=evidence,
    )

    async with app.run_test(size=(150, 80)):
        connection = str(
            app.query_one("#settings-speech-status-provider-connection").render()
        )
        assert "not_tested" in connection
        assert "catalog not_tested" in connection


def test_local_runtime_import_guards_intercept_static_and_dynamic_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import_attempts = _install_local_runtime_import_guards(monkeypatch)

    def import_nemo_asr():
        import nemo.collections.asr

        return nemo.collections.asr

    with pytest.raises(AssertionError, match="nemo.collections.asr"):
        import_nemo_asr()
    with pytest.raises(AssertionError, match="faster_whisper"):
        importlib.import_module("faster_whisper")

    assert import_attempts == ["nemo.collections.asr", "faster_whisper"]


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
    imported_local_runtimes = _install_local_runtime_import_guards(monkeypatch)
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
    assert imported_local_runtimes == []


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
async def test_saved_but_unavailable_and_transient_reconfiguration_are_not_ready() -> (
    None
):
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
        pending_copy = str(
            app.query_one("#settings-speech-save-result", Static).render()
        )
        assert "Runtime reconfiguration" in pending_copy
        assert "audio.cpp: pending" in pending_copy

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
async def test_staged_managed_save_is_pending_apply_not_reconfiguring() -> None:
    state = _audio_cpp_state()
    state.providers["audio_cpp"].update(
        AudioCppConfig(
            mode="managed",
            managed_binary_path="/private/test/audiocpp_server",
            managed_server_json_path="/private/test/server.json",
        ).to_mapping()
    )
    app = _PanelHarness(
        configure_provider="audio_cpp",
        state=state,
        current_configuration_revision=4,
    )

    async with app.run_test(size=(150, 80)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        app.query_one("#settings-speech-audio_cpp-mode", Select).value = "external"
        await pilot.pause()
        panel.request_save()
        await pilot.pause()

        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=1,
                persisted=True,
                provider_statuses={"audio_cpp": "pending"},
                provider_configuration_revisions={"audio_cpp": 5},
                provider_runtime_revisions={"audio_cpp": 9},
                staged_provider_ids=frozenset({"audio_cpp"}),
            )
        )
        await pilot.pause()

        runtime = str(
            app.query_one("#settings-speech-status-provider-runtime").render()
        )
        assert "Not checked" in runtime
        assert "Reconfiguring" not in runtime
        pending_copy = str(
            app.query_one("#settings-speech-save-result", Static).render()
        )
        assert "active audio.cpp configuration remains unchanged" in pending_copy
        assert "Open Speech Lab" in pending_copy
        assert "apply External mode" in pending_copy
        assert ("audio_cpp", 1) not in panel._provider_runtime_request_observed_at


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
@pytest.mark.parametrize("terminal_size", ((120, 40), (80, 24)))
async def test_real_stylesheet_keeps_managed_setup_reachable_at_supported_widths(
    terminal_size: tuple[int, int],
) -> None:
    host = _StyledDestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=terminal_size) as pilot:
        screen = await _open_speech_tts(host, pilot)
        configure = screen.query_one("#settings-speech-configure-provider", Select)
        configure.value = "audio_cpp"
        await _wait_for_selector(
            screen,
            pilot,
            "#settings-speech-provider-audio_cpp",
            timeout=4.0,
        )
        mode = screen.query_one("#settings-speech-audio_cpp-mode", Select)
        mode.value = "managed"
        await pilot.pause()

        binary = screen.query_one(
            "#settings-speech-audio_cpp-guided-binary-path", Input
        )
        browse = screen.query_one(
            "#settings-speech-audio_cpp-guided-binary-path-browse", Button
        )
        binary.scroll_visible(animate=False)
        await pilot.pause()
        assert binary.region.width >= 16
        assert browse.region.y == binary.region.y
        assert browse.region.x + browse.region.width <= pilot.app.size.width

        advanced = screen.query_one("#settings-speech-audio-cpp-advanced", Collapsible)
        advanced.collapsed = False
        await pilot.pause(0.6)
        grace = screen.query_one(
            "#settings-speech-audio_cpp-managed-termination-grace-seconds", Input
        )
        grace.scroll_visible(animate=False)
        await pilot.pause()
        assert grace.region.width >= 16
        assert 0 <= grace.region.y < pilot.app.size.height

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
async def test_panel_exposes_mode_specific_managed_audio_cpp_setup_controls() -> None:
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
        mode = screen.query_one("#settings-speech-audio_cpp-mode", Select)
        external_fields = screen.query_one("#settings-speech-audio-cpp-external-fields")
        managed_fields = screen.query_one("#settings-speech-audio-cpp-managed-fields")
        lifecycle_fields = screen.query_one(
            "#settings-speech-audio-cpp-managed-lifecycle-fields"
        )
        assert mode.value == "external"
        assert external_fields.display is True
        assert managed_fields.display is False
        assert lifecycle_fields.display is False

        mode.focus()
        mode.value = "managed"
        await pilot.pause()

        assert mode.has_focus is True
        assert external_fields.display is False
        assert managed_fields.display is True
        assert lifecycle_fields.display is True
        assert screen.query_one(
            "#settings-speech-audio_cpp-managed-binary-path-browse", Button
        )
        assert screen.query_one(
            "#settings-speech-audio_cpp-managed-server-json-path-browse", Button
        )
        assert screen.query_one("#settings-speech-audio-cpp-use-detected", Button)
        assert (
            screen.query_one(
                "#settings-speech-audio_cpp-managed-startup-timeout-seconds",
                Input,
            ).value
            == "30.0"
        )
        assert (
            screen.query_one(
                "#settings-speech-audio_cpp-managed-health-check-interval-seconds",
                Input,
            ).value
            == "10.0"
        )
        assert (
            screen.query_one(
                "#settings-speech-audio_cpp-managed-termination-grace-seconds",
                Input,
            ).value
            == "5.0"
        )
        audio_text = " ".join(
            node.renderable.plain
            if hasattr(node.renderable, "plain")
            else str(node.renderable)
            for node in screen.query("#settings-speech-provider-audio_cpp Static")
        ).lower()
        for required in (
            "managed local server",
            "execute the selected file",
            "server.json",
            "127.0.0.1",
            "working directory",
            "relative paths",
        ):
            assert required in audio_text
        for forbidden_id in (
            "settings-speech-audio-cpp-start-test",
            "settings-speech-audio-cpp-restart",
            "settings-speech-audio-cpp-shutdown",
            "settings-speech-audio-cpp-process-status",
            "settings-speech-audio-cpp-diagnostics",
        ):
            assert not screen.query(f"#{forbidden_id}")


@pytest.mark.asyncio
async def test_first_managed_switch_defaults_to_guided_and_preserves_manual_draft() -> (
    None
):
    app = _PanelHarness(configure_provider="audio_cpp")

    async with app.run_test(size=(150, 70)) as pilot:
        mode = app.query_one("#settings-speech-audio_cpp-mode", Select)
        mode.value = "managed"
        await pilot.pause()

        setup_source = app.query_one(
            "#settings-speech-audio_cpp-managed-setup-source",
            Select,
        )
        guided = app.query_one("#settings-speech-audio-cpp-guided-fields")
        manual = app.query_one("#settings-speech-audio-cpp-manual-json-fields")
        assert setup_source.value == "guided"
        assert guided.display is True
        assert manual.display is False
        assert app.query_one(
            "#settings-speech-audio_cpp-guided-binary-path-browse", Button
        )
        assert app.query_one("#settings-speech-audio-cpp-guided-add-package", Button)
        assert app.query_one(
            "#settings-speech-audio_cpp-guided-default-model-id", Select
        )
        assert app.query_one(
            "#settings-speech-audio_cpp-guided-backend-preference", Select
        )

        manual_binary = app.query_one(
            "#settings-speech-audio_cpp-managed-binary-path", Input
        )
        manual_binary.value = "/manual/audiocpp_server"
        setup_source.value = "user_json"
        await pilot.pause()

        assert guided.display is False
        assert manual.display is True
        assert manual_binary.value == "/manual/audiocpp_server"

        setup_source.value = "guided"
        await pilot.pause()
        assert guided.display is True
        assert manual_binary.value == "/manual/audiocpp_server"


@pytest.mark.asyncio
async def test_guided_scan_review_and_save_remain_inert_and_handoff_to_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "audiocpp_server"
    binary.write_bytes(b"synthetic-binary")
    binary.chmod(0o700)
    package_root = tmp_path / "Supertonic-3-GGUF"
    package_root.mkdir()
    (package_root / "supertonic-3-orig.gguf").write_bytes(
        b"GGUF" + struct.pack("<I", 3)
    )
    lifecycle_calls: list[str] = []

    async def forbidden_start(self, *_args, **_kwargs):
        del self
        lifecycle_calls.append("start")
        raise AssertionError("Settings started managed audio.cpp")

    monkeypatch.setattr(AudioCppSupervisor, "ensure_running", forbidden_start)
    app = _PanelHarness(configure_provider="audio_cpp")

    async with app.run_test(size=(170, 80)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        app.query_one("#settings-speech-audio_cpp-mode", Select).value = "managed"
        await pilot.pause()
        app.query_one(
            "#settings-speech-audio_cpp-guided-binary-path", Input
        ).value = str(binary)

        panel._audio_cpp_package_picker_result(package_root)
        await _settle(pilot)

        package_text = " ".join(
            str(node.renderable)
            for node in app.query(".settings-speech-audio-cpp-package-copy")
        )
        assert "Supertonic 3 Original-Dtype GGUF" in package_text
        assert "supertonic / supertonic_3_orig" in package_text
        assert "Text-to-speech" in package_text
        assert "Exact reviewed recipe" in package_text
        assert "audio.cpp 0.5.1" in package_text
        assert "supertonic-3-orig" in package_text
        assert "Loads lazily" in package_text
        assert "remain in memory until Shutdown" in package_text
        assert str(tmp_path) not in package_text
        assert (
            app.query_one(
                "#settings-speech-audio_cpp-guided-default-model-id",
                Select,
            ).value
            == "supertonic-3-orig"
        )
        unsaved_handoff = app.query_one(
            "#settings-speech-audio-cpp-open-lab",
            Button,
        )
        assert str(unsaved_handoff.label) == "Save Settings before opening Speech Lab"
        assert unsaved_handoff.disabled is True
        assert "saved configuration" in str(unsaved_handoff.tooltip)

        await pilot.click("#settings-speech-save")
        await _settle(pilot)

        assert lifecycle_calls == []
        assert len(app.events) == 1
        saved = app.events[0].settings["audio_cpp"]
        assert saved["mode"] == "managed"
        assert saved["managed_setup_source"] == "guided"
        assert saved["guided_binary_path"] == str(binary)
        assert saved["guided_default_model_id"] == "supertonic-3-orig"
        assert len(saved["guided_packages"]) == 1

        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=app.events[0].request_id or 0,
                persisted=True,
                provider_statuses={"audio_cpp": "pending"},
                provider_configuration_revisions={"audio_cpp": 2},
                provider_runtime_revisions={"audio_cpp": 1},
                staged_provider_ids=frozenset({"audio_cpp"}),
            )
        )
        await pilot.pause()

        result = str(app.query_one("#settings-speech-save-result", Static).renderable)
        assert "Configuration saved — ready to test" in result
        handoff = app.query_one("#settings-speech-audio-cpp-open-lab", Button)
        assert str(handoff.label) == "Open Speech Lab & Hear a Sample"
        handoff.press()
        await _settle(pilot)
        assert app.navigation[-1].screen_name == "stts"
        assert app.navigation[-1].screen_context == {
            "view": "playground",
            "provider": "audio_cpp",
            "intent": SpeechTTSNavigationIntent.TEST.value,
        }


@pytest.mark.asyncio
async def test_guided_reference_required_default_does_not_promise_sample(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "audiocpp_server"
    binary.write_bytes(b"synthetic-binary")
    binary.chmod(0o700)
    package_root = tmp_path / "reviewed-models"
    package_root.mkdir()
    (package_root / "supertonic-3-orig.gguf").write_bytes(
        b"GGUF" + struct.pack("<I", 3)
    )
    (package_root / "pocket-tts-english-bf16.gguf").write_bytes(
        b"GGUF" + struct.pack("<I", 3)
    )
    app = _PanelHarness(configure_provider="audio_cpp")

    async with app.run_test(size=(170, 80)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        app.query_one("#settings-speech-audio_cpp-mode", Select).value = "managed"
        await pilot.pause()
        app.query_one(
            "#settings-speech-audio_cpp-guided-binary-path", Input
        ).value = str(binary)
        panel._audio_cpp_package_picker_result(package_root)
        await _settle(pilot)

        package_text = " ".join(
            str(node.renderable)
            for node in app.query(".settings-speech-audio-cpp-package-copy")
        )
        assert "pocket_tts / pocket_tts_english_bf16" in package_text
        assert "Reference: Required" in package_text
        assert "voice setup" in package_text

        default_model = app.query_one(
            "#settings-speech-audio_cpp-guided-default-model-id", Select
        )
        default_model.value = "pocket-tts-english-bf16"
        await pilot.pause()
        handoff = app.query_one("#settings-speech-audio-cpp-open-lab", Button)
        assert str(handoff.label) == "Save Settings before opening Speech Lab"
        assert handoff.disabled is True

        panel.request_save()
        await _settle(pilot)
        assert len(app.events) == 1
        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=app.events[0].request_id or 0,
                persisted=True,
                provider_statuses={"audio_cpp": "pending"},
                provider_configuration_revisions={"audio_cpp": 2},
                provider_runtime_revisions={"audio_cpp": 1},
                staged_provider_ids=frozenset({"audio_cpp"}),
            )
        )
        await pilot.pause()

        result = str(app.query_one("#settings-speech-save-result", Static).renderable)
        assert "Configuration saved — ready to test" in result
        assert "test the saved Guided settings" in result
        assert "Hear a Sample" not in result
        handoff = app.query_one("#settings-speech-audio-cpp-open-lab", Button)
        assert str(handoff.label) == "Open Speech Lab to Test Connection"
        assert handoff.disabled is False


@pytest.mark.asyncio
async def test_guided_save_rechecks_accepted_package_before_posting(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "audiocpp_server"
    binary.write_bytes(b"synthetic-binary")
    binary.chmod(0o700)
    package_root = tmp_path / "Supertonic-3-GGUF"
    package_root.mkdir()
    model = package_root / "supertonic-3-orig.gguf"
    model.write_bytes(b"GGUF" + struct.pack("<I", 3))
    app = _PanelHarness(configure_provider="audio_cpp")

    async with app.run_test(size=(170, 80)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        app.query_one("#settings-speech-audio_cpp-mode", Select).value = "managed"
        await pilot.pause()
        app.query_one(
            "#settings-speech-audio_cpp-guided-binary-path", Input
        ).value = str(binary)
        panel._audio_cpp_package_picker_result(package_root)
        await _settle(pilot)

        model.write_bytes(b"GGUF" + struct.pack("<I", 3) + b"changed")
        await pilot.click("#settings-speech-save")
        await _settle(pilot)

        assert app.events == []
        result = str(app.query_one("#settings-speech-save-result", Static).renderable)
        assert "changed" in result.lower()
        assert "scan" in result.lower()
        assert app.query_one(
            "#settings-speech-audio_cpp-guided-packages-error",
            Static,
        ).renderable


@pytest.mark.asyncio
async def test_guided_scan_late_cancelled_root_cannot_mutate_newer_draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    older_root = tmp_path / "PocketTTS"
    older_root.mkdir()
    (older_root / "pocket-tts-english-q8_0.gguf").write_bytes(
        b"GGUF" + struct.pack("<I", 3)
    )
    newer_root = tmp_path / "Supertonic"
    newer_root.mkdir()
    (newer_root / "supertonic-3-orig.gguf").write_bytes(b"GGUF" + struct.pack("<I", 3))
    real_scan = speech_tts_settings_panel_module.scan_audio_cpp_package_root_async
    older_started = asyncio.Event()
    release_older = asyncio.Event()

    async def reordered_scan(path: Path, *, request_revision: int):
        if path == older_root:
            older_started.set()
            try:
                await release_older.wait()
            except asyncio.CancelledError:
                # Model an uncooperative filesystem call that completes after its
                # worker was cancelled; the revision fence must still reject it.
                pass
        return await real_scan(path, request_revision=request_revision)

    monkeypatch.setattr(
        speech_tts_settings_panel_module,
        "scan_audio_cpp_package_root_async",
        reordered_scan,
    )
    app = _PanelHarness(configure_provider="audio_cpp")

    async with app.run_test(size=(170, 80)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        app.query_one("#settings-speech-audio_cpp-mode", Select).value = "managed"
        await pilot.pause()

        panel._audio_cpp_package_picker_result(older_root)
        await asyncio.wait_for(older_started.wait(), timeout=1.0)
        panel._audio_cpp_package_picker_result(newer_root)
        await pilot.pause()
        release_older.set()
        await _settle(pilot)

        model_ids = {
            package.public_model_id for package in panel._audio_cpp_guided_packages()
        }
        assert model_ids == {"supertonic-3-orig"}


@pytest.mark.asyncio
async def test_guided_source_round_trip_fences_an_inflight_package_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "Supertonic"
    package_root.mkdir()
    (package_root / "supertonic-3-orig.gguf").write_bytes(
        b"GGUF" + struct.pack("<I", 3)
    )
    real_scan = speech_tts_settings_panel_module.scan_audio_cpp_package_root_async
    scan_started = asyncio.Event()
    release_scan = asyncio.Event()

    async def delayed_scan(path: Path, *, request_revision: int):
        scan_started.set()
        try:
            await release_scan.wait()
        except asyncio.CancelledError:
            # A blocking filesystem call may still return after cancellation.
            pass
        return await real_scan(path, request_revision=request_revision)

    monkeypatch.setattr(
        speech_tts_settings_panel_module,
        "scan_audio_cpp_package_root_async",
        delayed_scan,
    )
    app = _PanelHarness(configure_provider="audio_cpp")

    async with app.run_test(size=(170, 80)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        app.query_one("#settings-speech-audio_cpp-mode", Select).value = "managed"
        await pilot.pause()

        panel._audio_cpp_package_picker_result(package_root)
        await asyncio.wait_for(scan_started.wait(), timeout=1.0)
        source = app.query_one(
            "#settings-speech-audio_cpp-managed-setup-source",
            Select,
        )
        source.value = "user_json"
        await pilot.pause()
        source.value = "guided"
        await pilot.pause()
        release_scan.set()
        await _settle(pilot)

        assert panel._audio_cpp_guided_packages() == ()


@pytest.mark.asyncio
async def test_panel_does_not_offer_unqualified_managed_mode_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        speech_tts_settings_panel_module,
        "_AUDIO_CPP_MANAGED_UI_SUPPORTED",
        False,
    )
    app = _PanelHarness(configure_provider="audio_cpp")

    async with app.run_test(size=(150, 60)):
        mode = app.query_one("#settings-speech-audio_cpp-mode", Select)

        assert [value for _label, value in mode._options] == ["external"]
        assert mode.value == "external"
        assert (
            app.query_one("#settings-speech-audio-cpp-managed-fields").display is False
        )
        notice = str(
            app.query_one(
                "#settings-speech-audio-cpp-managed-platform-notice",
                Static,
            ).render()
        ).lower()
        assert "windows" in notice
        assert "external" in notice


@pytest.mark.asyncio
async def test_use_detected_updates_only_the_unsaved_managed_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detected = "/opt/homebrew/bin/audiocpp_server"
    monkeypatch.setattr(
        speech_tts_settings_panel_module,
        "detect_audio_cpp_server_binary",
        lambda: detected,
    )
    app = _PanelHarness(configure_provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        app.query_one("#settings-speech-audio_cpp-mode", Select).value = "managed"
        await pilot.pause()
        app.query_one("#settings-speech-audio-cpp-use-detected", Button).press()
        await pilot.pause()

        assert (
            app.query_one("#settings-speech-audio_cpp-guided-binary-path", Input).value
            == detected
        )
        assert app.events == []
        result = str(app.query_one("#settings-speech-save-result", Static).renderable)
        assert "draft" in result.lower()
        assert "not started" in result.lower()
        assert detected not in result


@pytest.mark.asyncio
async def test_failed_detection_preserves_the_existing_binary_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        speech_tts_settings_panel_module,
        "detect_audio_cpp_server_binary",
        lambda: None,
    )
    app = _PanelHarness(configure_provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        app.query_one("#settings-speech-audio_cpp-mode", Select).value = "managed"
        await pilot.pause()
        binary = app.query_one("#settings-speech-audio_cpp-guided-binary-path", Input)
        binary.value = "/keep/existing/draft"
        app.query_one("#settings-speech-audio-cpp-use-detected", Button).press()
        await pilot.pause()

        assert binary.value == "/keep/existing/draft"
        assert app.events == []
        result = str(app.query_one("#settings-speech-save-result", Static).renderable)
        assert "not found" in result.lower()
        assert "browse" in result.lower()


@pytest.mark.asyncio
async def test_managed_save_validates_files_without_starting_or_contacting_tts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "audiocpp_server"
    binary.write_bytes(b"synthetic-binary")
    binary.chmod(0o700)
    server_json = tmp_path / "server.json"
    server_json.write_text('{"host":"127.0.0.1","port":19004}', encoding="utf-8")
    lifecycle_calls: list[str] = []

    async def forbidden_start(self, *_args, **_kwargs):
        del self
        lifecycle_calls.append("start")
        raise AssertionError("Settings started managed audio.cpp")

    monkeypatch.setattr(AudioCppSupervisor, "ensure_running", forbidden_start)
    app = _PanelHarness(configure_provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        app.query_one("#settings-speech-audio_cpp-mode", Select).value = "managed"
        await pilot.pause()
        app.query_one(
            "#settings-speech-audio_cpp-managed-setup-source", Select
        ).value = "user_json"
        await pilot.pause()
        app.query_one(
            "#settings-speech-audio_cpp-managed-binary-path", Input
        ).value = str(binary)
        app.query_one(
            "#settings-speech-audio_cpp-managed-server-json-path", Input
        ).value = str(server_json)
        await pilot.click("#settings-speech-save")
        await pilot.pause()

        assert lifecycle_calls == []
        assert len(app.events) == 1
        saved = app.events[0].settings["audio_cpp"]
        assert saved["mode"] == "managed"
        assert saved["base_url"] == AudioCppConfig().base_url
        assert saved["managed_binary_path"] == str(binary)
        assert saved["managed_server_json_path"] == str(server_json)
        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=app.events[0].request_id or 0,
                persisted=True,
                provider_statuses={"audio_cpp": "pending"},
                provider_configuration_revisions={"audio_cpp": 2},
                provider_runtime_revisions={"audio_cpp": 1},
                staged_provider_ids=frozenset({"audio_cpp"}),
            )
        )
        await pilot.pause()
        pending_copy = str(
            app.query_one("#settings-speech-save-result", Static).render()
        )
        assert "active audio.cpp configuration remains unchanged" in pending_copy
        assert "apply the saved Managed settings" in pending_copy
        assert "restart and apply" not in pending_copy


@pytest.mark.asyncio
async def test_invalid_managed_binary_is_adjacent_safe_and_posts_no_save(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "private-missing-audiocpp_server"
    server_json = tmp_path / "server.json"
    server_json.write_text('{"host":"127.0.0.1","port":19005}', encoding="utf-8")
    app = _PanelHarness(configure_provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        app.query_one("#settings-speech-audio_cpp-mode", Select).value = "managed"
        await pilot.pause()
        app.query_one(
            "#settings-speech-audio_cpp-managed-setup-source", Select
        ).value = "user_json"
        await pilot.pause()
        app.query_one(
            "#settings-speech-audio_cpp-managed-binary-path", Input
        ).value = str(missing)
        app.query_one(
            "#settings-speech-audio_cpp-managed-server-json-path", Input
        ).value = str(server_json)
        await pilot.click("#settings-speech-save")
        await pilot.pause()

        assert app.events == []
        error = str(
            app.query_one(
                "#settings-speech-audio_cpp-managed-binary-path-error", Static
            ).renderable
        )
        assert "executable" in error.lower()
        assert str(missing) not in error


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
        await _click_visible_button(
            screen, pilot, "#settings-speech-restore-defaults"
        )
        await pilot.pause()
        await _click_visible_button(screen, pilot, "#settings-speech-revert")
        await pilot.pause()

    assert calls == []


@pytest.mark.asyncio
async def test_mounting_external_audio_cpp_settings_never_starts_managed_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launches: list[str] = []

    async def forbidden_start(self, *_args, **_kwargs):
        del self
        launches.append("ensure_running")
        raise AssertionError("Settings launched managed audio.cpp")

    monkeypatch.setattr(AudioCppSupervisor, "ensure_running", forbidden_start)
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = await _open_speech_tts(host, pilot)
        configure = screen.query_one("#settings-speech-configure-provider", Select)
        configure.value = "audio_cpp"
        await _wait_for_selector(
            screen,
            pilot,
            "#settings-speech-provider-audio_cpp",
            timeout=4.0,
        )

    assert launches == []


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
async def test_default_rollback_failure_keeps_retry_draft_and_warns_about_restart() -> (
    None
):
    app = _PanelHarness(configure_provider="audio_cpp", state=_audio_cpp_state())
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
                provider_statuses={"audio_cpp": "applied"},
                provider_configuration_revisions={"audio_cpp": 7},
                provider_runtime_revisions={"audio_cpp": 41},
                defaults_activated=False,
                defaults_activation_status="rollback_failed",
            )
        )

        result = str(app.query_one("#settings-speech-save-result", Static).renderable)
        assert "rollback failed" in result
        assert "restart may use the new default" in result
        assert "previous default remains active" not in result
        assert panel.has_unsaved_changes() is True


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
async def test_openai_authentication_control_and_official_preset_are_explicit() -> None:
    state = load_global_speech_tts_state(
        {
            "COMPREHENSIVE_CONFIG_RAW": {
                "app_tts": {
                    "OPENAI_BASE_URL": "http://127.0.0.1:8765/v1/audio/speech",
                    "OPENAI_AUTH_MODE": "none",
                }
            }
        },
        environment={},
    )
    app = _PanelHarness(configure_provider="openai", state=state)

    async with app.run_test(size=(150, 60)) as pilot:
        authentication = app.query_one(
            "#settings-speech-openai-authentication-mode", Select
        )
        assert authentication.value == "none"
        assert [(str(label), value) for label, value in authentication._options] == [
            ("API key", "api_key"),
            ("None", "none"),
        ]

        preset = app.query_one("#settings-speech-openai-official-preset", Button)
        preset.scroll_visible(animate=False)
        await pilot.pause()
        preset.press()
        await pilot.pause()

        assert authentication.value == "api_key"
        assert (
            app.query_one("#settings-speech-openai-base-url", Input).value
            == "https://api.openai.com/v1/audio/speech"
        )


@pytest.mark.asyncio
async def test_plaintext_none_save_requires_confirmation_before_posting() -> None:
    app = _PanelHarness(configure_provider="openai")
    endpoint = "http://voice.example.test:8765/v1/audio/speech"

    async with app.run_test(size=(150, 60)) as pilot:
        app.query_one("#settings-speech-openai-base-url", Input).value = endpoint
        app.query_one(
            "#settings-speech-openai-authentication-mode", Select
        ).value = "none"

        await pilot.click("#settings-speech-save")
        await pilot.pause()

        assert app.events == []
        assert app.screen.query_one("#settings-speech-openai-none-http-confirm")

        await pilot.click("#settings-speech-openai-none-http-confirm")
        await pilot.pause()

        assert len(app.events) == 1
        fingerprint = openai_destination_fingerprint(
            "openai", normalize_openai_compatible_endpoint(endpoint)
        )
        assert app.events[0].settings["OPENAI_AUTH_MODE"] == "none"
        assert app.events[0].settings["OPENAI_NONE_HTTP_CONFIRMATION"] == fingerprint
        assert app.events[0].settings["OPENAI_NONE_HTTP_CONFIRMATION"] != endpoint


@pytest.mark.asyncio
async def test_switching_back_to_api_key_clears_confirmation_and_saved_baseline() -> (
    None
):
    endpoint = normalize_openai_compatible_endpoint(
        "http://voice.example.test:8765/v1/audio/speech"
    )
    fingerprint = openai_destination_fingerprint("openai", endpoint)
    state = load_global_speech_tts_state(
        {
            "COMPREHENSIVE_CONFIG_RAW": {
                "app_tts": {
                    "OPENAI_BASE_URL": endpoint.speech_url,
                    "OPENAI_AUTH_MODE": "none",
                    "OPENAI_NONE_HTTP_CONFIRMATION": fingerprint,
                }
            }
        },
        environment={},
    )
    app = _PanelHarness(configure_provider="openai", state=state)

    async with app.run_test(size=(150, 60)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        app.query_one(
            "#settings-speech-openai-authentication-mode", Select
        ).value = "api_key"

        await pilot.click("#settings-speech-save")
        await pilot.pause()

        assert app.events[0].delete_setting_keys == ("OPENAI_NONE_HTTP_CONFIRMATION",)
        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=app.events[0].request_id or 0,
                persisted=True,
                provider_statuses={"openai": "applied"},
                provider_configuration_revisions={"openai": 2},
                provider_runtime_revisions={"openai": 2},
            )
        )
        await pilot.pause()

        assert panel.state.openai_plaintext_confirmation is None
        assert panel.original_state.openai_plaintext_confirmation is None
        assert panel.has_unsaved_changes() is False


@pytest.mark.asyncio
async def test_invalid_persisted_confirmation_cleanup_settles_saved_baseline() -> None:
    endpoint = normalize_openai_compatible_endpoint(
        "http://voice.example.test:8765/v1/audio/speech"
    )
    fingerprint = openai_destination_fingerprint("openai", endpoint)
    state = load_global_speech_tts_state(
        {
            "COMPREHENSIVE_CONFIG_RAW": {
                "app_tts": {
                    "OPENAI_BASE_URL": endpoint.speech_url,
                    "OPENAI_AUTH_MODE": "api_key",
                    "OPENAI_NONE_HTTP_CONFIRMATION": fingerprint,
                }
            }
        },
        environment={},
    )
    app = _PanelHarness(configure_provider="openai", state=state)

    async with app.run_test(size=(150, 60)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)

        await pilot.click("#settings-speech-save")
        await pilot.pause()

        assert app.events[0].delete_setting_keys == ("OPENAI_NONE_HTTP_CONFIRMATION",)
        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=app.events[0].request_id or 0,
                persisted=True,
                provider_statuses={},
            )
        )
        await pilot.pause()

        assert panel.state.openai_plaintext_confirmation_cleanup_needed is False
        assert (
            panel.original_state.openai_plaintext_confirmation_cleanup_needed is False
        )
        assert panel.request_save() is None
        assert len(app.events) == 1


@pytest.mark.asyncio
async def test_cross_provider_save_cleans_stale_confirmation_after_success() -> None:
    endpoint = normalize_openai_compatible_endpoint(
        "http://voice.example.test:8765/v1/audio/speech"
    )
    fingerprint = openai_destination_fingerprint("openai", endpoint)
    state = load_global_speech_tts_state(
        {
            "COMPREHENSIVE_CONFIG_RAW": {
                "app_tts": {
                    "OPENAI_BASE_URL": endpoint.speech_url,
                    "OPENAI_AUTH_MODE": "api_key",
                    "OPENAI_NONE_HTTP_CONFIRMATION": fingerprint,
                }
            }
        },
        environment={},
    )
    app = _PanelHarness(configure_provider="elevenlabs", state=state)

    async with app.run_test(size=(150, 60)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        assert panel.has_unsaved_changes() is True
        assert app.events == []
        original_source = panel.state.provider_sources["elevenlabs"]

        await pilot.click("#settings-speech-save")
        await pilot.pause()

        event = app.events[0]
        assert event.settings == {}
        assert event.delete_setting_keys == ("OPENAI_NONE_HTTP_CONFIRMATION",)
        assert panel.state.openai_plaintext_confirmation_cleanup_needed is True

        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=event.request_id or 0,
                persisted=True,
                provider_statuses={},
            )
        )
        await pilot.pause()

        assert panel.state.openai_plaintext_confirmation_cleanup_needed is False
        assert (
            panel.original_state.openai_plaintext_confirmation_cleanup_needed is False
        )
        assert panel.state.provider_sources["elevenlabs"] is original_source
        assert panel.original_state.provider_sources["elevenlabs"] is original_source
        assert panel.has_unsaved_changes() is False


@pytest.mark.asyncio
async def test_cross_provider_failed_save_keeps_stale_confirmation_dirty() -> None:
    endpoint = normalize_openai_compatible_endpoint(
        "http://voice.example.test:8765/v1/audio/speech"
    )
    fingerprint = openai_destination_fingerprint("openai", endpoint)
    state = load_global_speech_tts_state(
        {
            "COMPREHENSIVE_CONFIG_RAW": {
                "app_tts": {
                    "OPENAI_BASE_URL": endpoint.speech_url,
                    "OPENAI_AUTH_MODE": "api_key",
                    "OPENAI_NONE_HTTP_CONFIRMATION": fingerprint,
                }
            }
        },
        environment={},
    )
    app = _PanelHarness(configure_provider="elevenlabs", state=state)

    async with app.run_test(size=(150, 60)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)

        await pilot.click("#settings-speech-save")
        await pilot.pause()
        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=app.events[0].request_id or 0,
                persisted=False,
                provider_statuses={},
                failure_phase="before_replace",
            )
        )
        await pilot.pause()

        assert panel.state.openai_plaintext_confirmation_cleanup_needed is True
        assert panel.original_state.openai_plaintext_confirmation_cleanup_needed is True
        assert panel.has_unsaved_changes() is True


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
        assert "outside Chatbook" in rendered
        assert "OPENAI_API_KEY" not in rendered
        assert "synthetic-local-fallback" not in rendered
        assert "synthetic-environment-value" not in rendered
        details = app.query_one("#settings-speech-details", Collapsible)
        technical_copy = " ".join(str(node.render()) for node in details.query(Static))
        assert "Environment" in technical_copy
        assert "OPENAI_API_KEY" in technical_copy
        assert "read-only" in technical_copy
        assert "shadowed" in technical_copy
        assert "synthetic-local-fallback" not in technical_copy
        assert "synthetic-environment-value" not in technical_copy

        edit = app.query_one("#settings-speech-openai-credential-edit", Button)
        edit.scroll_visible(animate=False)
        await pilot.pause()
        edit.press()
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
        primary_copy = " ".join(
            str(node.render())
            for node in app.query(".settings-detail-row")
            if node.is_on_screen
        )
        details = app.query_one("#settings-speech-details", Collapsible)
        technical_copy = " ".join(str(node.render()) for node in details.query(Static))

        assert field.disabled is True
        assert browse.disabled is True
        assert environment_variable not in primary_copy
        assert environment_variable in technical_copy
        assert "read-only" in technical_copy
        assert "/environment/runtime-path" not in technical_copy


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
async def test_managed_validation_expands_advanced_field_before_focusing() -> None:
    state = _audio_cpp_state()
    state.providers["audio_cpp"].update(
        {
            "managed_binary_path": "/private/test/audiocpp_server",
            "managed_server_json_path": "/private/test/server.json",
            "managed_health_check_interval_seconds": 1.0,
        }
    )
    app = _PanelHarness(configure_provider="audio_cpp", state=state)

    async with app.run_test(size=(150, 80)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        advanced = app.query_one("#settings-speech-audio-cpp-advanced", Collapsible)
        assert advanced.collapsed is True

        app.query_one("#settings-speech-audio_cpp-mode", Select).value = "managed"
        await pilot.pause()
        assert panel.request_save() is None
        await pilot.pause()

        field = app.query_one(
            "#settings-speech-audio_cpp-managed-health-check-interval-seconds",
            Input,
        )
        error = app.query_one(
            "#settings-speech-audio_cpp-managed-health-check-interval-seconds-error",
            Static,
        )
        assert advanced.collapsed is False
        assert app.focused is field
        assert error.renderable


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
async def test_details_and_scope_start_collapsed_on_every_mount() -> None:
    app = _StyledPanelHarness(
        configure_provider="audio_cpp",
        observation=_audio_cpp_observation(),
        current_configuration_revision=4,
    )
    async with app.run_test(size=(150, 55)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        details = panel.query_one("#settings-speech-details", Collapsible)
        scope = panel.query_one("#settings-speech-scope-inspector", Collapsible)

        assert details.collapsed is True
        assert scope.collapsed is True
        visible_copy = " ".join(
            str(widget.render())
            for widget in panel.query(Static)
            if widget.display and widget.is_on_screen
        ).casefold()
        assert "task: set up" in visible_copy
        assert "current status" in visible_copy
        assert "revision" not in visible_copy
        assert "effective source" not in visible_copy

        default_provider = panel.query_one("#settings-speech-default-provider", Select)
        assert default_provider.value == "openai"
        default_provider.value = "audio_cpp"
        await pilot.pause()
        assert "Unsaved" in str(
            panel.query_one("#settings-speech-default-status", Static).render()
        )

        details = panel.query_one("#settings-speech-details", Collapsible)
        scope = panel.query_one("#settings-speech-scope-inspector", Collapsible)
        details.collapsed = False
        scope.collapsed = False
        await pilot.pause()
        assert panel.query_one(
            "#settings-speech-status-provider-configuration", Static
        ).is_on_screen
        assert (
            "revision"
            in " ".join(
                str(widget.render())
                for widget in details.query(Static)
                if widget.is_on_screen
            ).casefold()
        )

        await panel.recompose()
        await pilot.pause()

        assert (
            panel.query_one("#settings-speech-details", Collapsible).collapsed is True
        )
        assert (
            panel.query_one("#settings-speech-scope-inspector", Collapsible).collapsed
            is True
        )


@pytest.mark.asyncio
async def test_production_bundle_applies_speech_disclosure_styles() -> None:
    bundled_css = app_css_text()
    for selector in (
        "#settings-speech-details,\n#settings-speech-scope-inspector",
        "#settings-speech-details > CollapsibleTitle,\n"
        "#settings-speech-scope-inspector > CollapsibleTitle",
        "#settings-speech-details > Contents,\n"
        "#settings-speech-scope-inspector > Contents",
    ):
        assert selector in bundled_css

    app = _StyledPanelHarness(configure_provider="audio_cpp")
    assert [Path(path).resolve() for path in app.CSS_PATH] == [
        _BUNDLE.resolve(),
        _SETTINGS_SHEET.resolve(),
    ]

    async with app.run_test(size=(120, 40)):
        details = app.query_one("#settings-speech-details", Collapsible)
        title = details.query_one("CollapsibleTitle")
        contents = details.query_one("Contents")

        assert details.styles.width.value == 100
        assert details.styles.width.unit.name == "WIDTH"
        assert str(details.styles.height) == "auto"
        assert details.styles.min_height.value == 3
        assert details.styles.margin.top == 1
        assert details.styles.padding.width == 0
        assert title.styles.height.value == 3
        assert title.styles.min_height.value == 3
        assert title.styles.padding.top == 1
        assert title.styles.padding.width == 2
        assert str(contents.styles.height) == "auto"
        assert contents.styles.padding.width == 2


@pytest.mark.asyncio
async def test_managed_guided_selection_provenance_stays_in_collapsed_details() -> None:
    state = _audio_cpp_state()
    state.providers["audio_cpp"].update(
        {
            "mode": "managed",
            "managed_setup_source": "guided",
            "guided_binary_source": "path",
        }
    )
    app = _StyledPanelHarness(configure_provider="audio_cpp", state=state)

    async with app.run_test(size=(150, 55)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        details = panel.query_one("#settings-speech-details", Collapsible)
        guided_fields = panel.query_one("#settings-speech-audio-cpp-guided-fields")

        assert guided_fields.display is True
        assert details.collapsed is True
        provenance = details.query_one(
            "#settings-speech-audio_cpp-guided-binary-source",
            Static,
        )
        assert provenance.is_on_screen is False
        visible_copy = " ".join(
            str(widget.render())
            for widget in panel.query(Static)
            if widget.display and widget.is_on_screen
        ).casefold()
        assert "task: set up audio.cpp" in visible_copy
        assert "current status:" in visible_copy
        assert "selection source" not in visible_copy

        details.collapsed = False
        await pilot.pause()
        assert provenance.is_on_screen is True
        assert (
            "guided binary selection source: path"
            in str(provenance.render()).casefold()
        )

        await panel.recompose()
        await pilot.pause()

        remounted_details = panel.query_one("#settings-speech-details", Collapsible)
        assert remounted_details.collapsed is True
        assert (
            remounted_details.query_one(
                "#settings-speech-audio_cpp-guided-binary-source",
                Static,
            ).is_on_screen
            is False
        )


@pytest.mark.asyncio
async def test_speech_shortcuts_defer_to_focused_text_entry_and_resume_after_blur() -> (
    None
):
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = await _open_speech_tts(host, pilot)
        panel = screen.query_one(
            "#settings-speech-tts-panel",
            SpeechTTSSettingsPanel,
        )
        request_save = Mock()
        revert_to_saved = AsyncMock()
        panel.request_save = request_save
        panel.revert_to_saved = revert_to_saved

        configure = screen.query_one("#settings-speech-configure-provider", Select)
        configure.value = "openai"
        await _wait_for_selector(
            screen,
            pilot,
            "#settings-speech-openai-base-url",
            timeout=4.0,
        )
        panel = screen.query_one(
            "#settings-speech-tts-panel",
            SpeechTTSSettingsPanel,
        )
        panel.request_save = request_save
        panel.revert_to_saved = revert_to_saved
        endpoint = screen.query_one("#settings-speech-openai-base-url", Input)
        endpoint.value = "http://127.0.0.1:8765/v1/audio/speech"
        endpoint.focus()
        await pilot.pause()
        await pilot.press("end")
        command_allowed = Mock(wraps=panel.command_allowed)
        panel.command_allowed = command_allowed
        await pilot.press("s", "r")

        assert endpoint.value.endswith("speechsr")
        assert command_allowed.call_args_list == []
        assert panel.command_allowed("s") is False
        assert panel.command_allowed("left") is True
        request_save.assert_not_called()
        revert_to_saved.assert_not_awaited()

        text_area = TextArea("")
        await panel.mount(text_area)
        text_area.focus()
        await pilot.press("s", "r")

        assert text_area.text == "sr"
        assert panel.command_allowed("r") is False
        request_save.assert_not_called()
        revert_to_saved.assert_not_awaited()

        save = panel.query_one("#settings-speech-save", Button)
        save.focus()
        await pilot.press("s")
        await pilot.pause()
        request_save.assert_called_once_with()
        assert any(item.args == ("s",) for item in command_allowed.call_args_list)

        await pilot.press("r")
        await pilot.pause()
        revert_to_saved.assert_awaited_once_with()
        assert any(item.args == ("r",) for item in command_allowed.call_args_list)


@pytest.mark.asyncio
async def test_speech_save_and_revert_clicks_work_while_a_field_is_focused() -> None:
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = await _open_speech_tts(host, pilot)
        panel = screen.query_one(
            "#settings-speech-tts-panel",
            SpeechTTSSettingsPanel,
        )
        request_save = Mock()
        revert_to_saved = AsyncMock()
        panel.request_save = request_save
        panel.revert_to_saved = revert_to_saved
        configure = screen.query_one("#settings-speech-configure-provider", Select)
        configure.value = "openai"
        await _wait_for_selector(
            screen,
            pilot,
            "#settings-speech-openai-base-url",
            timeout=4.0,
        )
        panel = screen.query_one(
            "#settings-speech-tts-panel",
            SpeechTTSSettingsPanel,
        )
        panel.request_save = request_save
        panel.revert_to_saved = revert_to_saved
        endpoint = screen.query_one("#settings-speech-openai-base-url", Input)

        save = screen.query_one("#settings-speech-save", Button)
        save.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        endpoint.focus(scroll_visible=False)
        await pilot.click("#settings-speech-save")
        request_save.assert_called_once_with()

        revert = screen.query_one("#settings-speech-revert", Button)
        revert.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        endpoint.focus(scroll_visible=False)
        await pilot.click("#settings-speech-revert")
        await pilot.pause()
        revert_to_saved.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_size", ((120, 40), (80, 24)))
async def test_speech_disclosures_and_actions_remain_reachable(
    terminal_size: tuple[int, int],
) -> None:
    host = _StyledDestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=terminal_size) as pilot:
        screen = await _open_speech_tts(host, pilot)
        for selector in (
            "#settings-speech-details",
            "#settings-speech-scope-inspector",
        ):
            disclosure = screen.query_one(selector, Collapsible)
            title = disclosure.query_one("CollapsibleTitle")
            title.scroll_visible(animate=False)
            await pilot.pause()
            assert title.can_focus
            assert 0 <= title.region.x
            assert title.region.x + title.region.width <= pilot.app.size.width
            assert 0 <= title.region.y < pilot.app.size.height

        save = screen.query_one("#settings-speech-save", Button)
        save.scroll_visible(animate=False)
        await pilot.pause()
        assert 0 <= save.region.y < pilot.app.size.height


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
async def test_realtime_turn_detection_fields_render_and_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate round 5: turn detection is the knob that stops speech being
    chopped into fragments, so it belongs in Settings and not only in a
    TOML file. The server_vad-only numbers are disabled while semantic
    mode is selected -- the provider REJECTS them there outright."""
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        speech_tts_settings_panel_module,
        "save_settings_to_cli_config",
        lambda *args, **kwargs: calls.append((args, kwargs)) or True,
    )
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 60)) as pilot:
        await _settle(pilot)
        mode = app.query_one("#settings-speech-realtime-turn-detection", Select)
        threshold = app.query_one("#settings-speech-realtime-vad-threshold", Input)
        silence = app.query_one("#settings-speech-realtime-vad-silence-ms", Input)

        assert mode.value == "semantic_vad"
        assert threshold.disabled is True
        assert silence.disabled is True

        mode.value = "server_vad"
        await pilot.pause()
        assert threshold.disabled is False
        assert silence.disabled is False

        threshold.value = "0.6"
        silence.value = "700"
        await pilot.pause()

        await pilot.click("#settings-speech-save")
        await pilot.pause()

        (section_values,), _kwargs = calls[0]
        assert section_values["realtime"]["turn_detection"] == "server_vad"
        assert section_values["realtime"]["vad_threshold"] == 0.6
        assert section_values["realtime"]["vad_silence_ms"] == 700


@pytest.mark.asyncio
async def test_realtime_semantic_mode_deletes_the_server_vad_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Switching back to semantic must not leave stale server_vad numbers
    in config for a later reader to hand the provider."""
    monkeypatch.setattr(
        speech_tts_settings_panel_module,
        "_read_realtime_turn_detection",
        lambda: "server_vad",
    )
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        speech_tts_settings_panel_module,
        "save_settings_to_cli_config",
        lambda *args, **kwargs: calls.append((args, kwargs)) or True,
    )
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 60)) as pilot:
        await _settle(pilot)
        app.query_one(
            "#settings-speech-realtime-turn-detection", Select
        ).value = "semantic_vad"
        await pilot.pause()

        await pilot.click("#settings-speech-save")
        await pilot.pause()

        (section_values,), kwargs = calls[0]
        assert section_values["realtime"]["turn_detection"] == "semantic_vad"
        assert "vad_threshold" not in section_values["realtime"]
        assert "vad_silence_ms" not in section_values["realtime"]
        assert set(kwargs["delete_keys"]["realtime"]) >= {
            "vad_threshold",
            "vad_silence_ms",
        }


@pytest.mark.asyncio
async def test_realtime_unsupported_configured_provider_is_not_silently_rewritten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final review M5: the provider Select used to compose with the
    DEFAULT value regardless of what was configured, so a config naming an
    unsupported provider made the panel report unsaved changes the instant
    it opened -- and Save silently rewrote the user's value to "openai"
    without anyone touching the field."""
    monkeypatch.setattr(
        speech_tts_settings_panel_module, "_read_realtime_provider", lambda: "gemini"
    )
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        speech_tts_settings_panel_module,
        "save_settings_to_cli_config",
        lambda *args, **kwargs: calls.append((args, kwargs)) or True,
    )
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 60)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        await _settle(pilot)
        provider = app.query_one("#settings-speech-realtime-provider", Select)

        assert provider.value == "gemini"
        assert "gemini" in [value for _label, value in provider._options]
        assert panel.has_unsaved_changes() is False

        await pilot.click("#settings-speech-save")
        await pilot.pause()
        assert calls == [], "an untouched unsupported provider was rewritten"


@pytest.mark.asyncio
async def test_realtime_idle_timeout_is_written_as_an_int(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M5: a whole number of minutes belongs in config as `5`, not `5.0` --
    the float form is what the user's own config file ends up carrying."""
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        speech_tts_settings_panel_module,
        "save_settings_to_cli_config",
        lambda *args, **kwargs: calls.append((args, kwargs)) or True,
    )
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 60)) as pilot:
        await _settle(pilot)
        app.query_one(
            "#settings-speech-realtime-idle-timeout-minutes", Input
        ).value = "8"
        await pilot.pause()

        await pilot.click("#settings-speech-save")
        await pilot.pause()

        (section_values,), _kwargs = calls[0]
        written = section_values["realtime"]["idle_timeout_minutes"]
        assert written == 8
        assert isinstance(written, int), f"wrote {written!r}"


@pytest.mark.asyncio
async def test_realtime_fractional_idle_timeout_stays_a_float(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The int coercion above must not round a deliberate 2.5 to 2."""
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        speech_tts_settings_panel_module,
        "save_settings_to_cli_config",
        lambda *args, **kwargs: calls.append((args, kwargs)) or True,
    )
    app = _PanelHarness(configure_provider="audio_cpp")
    async with app.run_test(size=(150, 60)) as pilot:
        await _settle(pilot)
        app.query_one(
            "#settings-speech-realtime-idle-timeout-minutes", Input
        ).value = "2.5"
        await pilot.pause()

        await pilot.click("#settings-speech-save")
        await pilot.pause()

        (section_values,), _kwargs = calls[0]
        assert section_values["realtime"]["idle_timeout_minutes"] == 2.5


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
        app.query_one(
            "#settings-speech-realtime-model", Input
        ).value = "gpt-realtime-mini"
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
                "idle_timeout_minutes": 8,
                "turn_detection": "semantic_vad",
            },
            "dictation": {"handsfree_engine": "realtime"},
        }
        assert kwargs["delete_keys"] == {
            "realtime": ("vad_threshold", "vad_silence_ms")
        }
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
        # Semantic turn detection (the default) also deletes the two
        # server_vad-only knobs: the provider rejects them in that mode,
        # so leaving them in config would arm a future rejection.
        assert kwargs["delete_keys"] == {
            "realtime": ("voice", "vad_threshold", "vad_silence_ms")
        }


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
        result = str(app.query_one("#settings-speech-save-result", Static).renderable)
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

        assert app.query_one("#settings-speech-realtime-enabled", Switch).value is False
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


@pytest.mark.asyncio
async def test_guided_save_keeps_later_realtime_edits_dirty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "audiocpp_server"
    binary.write_bytes(b"synthetic-binary")
    binary.chmod(0o700)
    package_root = tmp_path / "Supertonic-3-GGUF"
    package_root.mkdir()
    (package_root / "supertonic-3-orig.gguf").write_bytes(
        b"GGUF" + struct.pack("<I", 3)
    )
    validation_started = asyncio.Event()
    release_validation = asyncio.Event()
    persisted: list[tuple[tuple, dict]] = []

    async def block_revalidation(_packages: object) -> tuple[()]:
        validation_started.set()
        await release_validation.wait()
        return ()

    monkeypatch.setattr(
        speech_tts_settings_panel_module,
        "revalidate_audio_cpp_guided_packages",
        block_revalidation,
    )
    monkeypatch.setattr(
        speech_tts_settings_panel_module,
        "save_settings_to_cli_config",
        lambda *args, **kwargs: (persisted.append((args, kwargs)), True)[1],
    )
    app = _PanelHarness(configure_provider="audio_cpp")

    async with app.run_test(size=(170, 80)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        app.query_one("#settings-speech-audio_cpp-mode", Select).value = "managed"
        await pilot.pause()
        app.query_one(
            "#settings-speech-audio_cpp-guided-binary-path", Input
        ).value = str(binary)
        panel._audio_cpp_package_picker_result(package_root)
        await _settle(pilot)

        realtime_model = app.query_one("#settings-speech-realtime-model", Input)
        realtime_model.value = "snapshot-a"
        await pilot.pause()
        await pilot.click("#settings-speech-save")
        await asyncio.wait_for(validation_started.wait(), timeout=2.0)

        realtime_model.value = "later-edit-b"
        await pilot.pause()
        release_validation.set()
        await _settle(pilot)

        assert persisted[0][0][0]["realtime"]["model"] == "snapshot-a"
        assert len(app.events) == 1
        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=app.events[0].request_id or 0,
                persisted=True,
                provider_statuses={"audio_cpp": "pending"},
                provider_configuration_revisions={"audio_cpp": 1},
                provider_runtime_revisions={"audio_cpp": 0},
                staged_provider_ids=frozenset({"audio_cpp"}),
            )
        )
        await pilot.pause()

        assert panel._realtime_original.model == "snapshot-a"
        assert panel._realtime_draft.model == "later-edit-b"
        assert panel.has_unsaved_changes() is True


@pytest.mark.asyncio
async def test_selecting_a_default_voice_profile_saves_its_id() -> None:
    app = _PanelHarness(configure_provider="openai")
    async with app.run_test(size=(150, 60)) as pilot:
        select = app.query_one("#settings-speech-default-profile", Select)
        select.value = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
        await pilot.click("#settings-speech-save")
        await pilot.pause()

        assert app.events
        assert app.events[0].settings["default_profile_id"] == (
            "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
        )


@pytest.mark.asyncio
async def test_choosing_none_clears_the_default_voice_profile() -> None:
    # Seed a state whose saved default is already the picker's known profile,
    # then act by choosing "None — use the fields below".
    state = load_global_speech_tts_state({})
    state.defaults.default_profile_id = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
    app = _PanelHarness(configure_provider="openai", state=state)
    async with app.run_test(size=(150, 60)) as pilot:
        select = app.query_one("#settings-speech-default-profile", Select)
        assert select.value == "3f2504e0-4f89-11d3-9a0c-0305e82c3301"

        select.value = ""  # the panel's blank sentinel
        await pilot.click("#settings-speech-save")
        await pilot.pause()

        assert app.events
        assert "default_profile_id" in app.events[0].delete_setting_keys
        assert "default_profile_id" not in app.events[0].settings


@pytest.mark.asyncio
async def test_unavailable_profile_store_keeps_the_saved_id_and_says_so() -> None:
    # Seed a saved default, then construct with profiles=None + an explicit
    # "confirmed unavailable" flag -- distinct from the default "loading"
    # state (see the next test), which must never say "unavailable".
    state = load_global_speech_tts_state({})
    state.defaults.default_profile_id = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
    app = _PanelHarness(
        configure_provider="openai",
        state=state,
        profiles=None,
        profiles_unavailable=True,
    )
    async with app.run_test(size=(150, 60)) as pilot:
        await pilot.pause()
        rendered = str(
            app.query_one("#settings-speech-default-profile-note", Static).renderable
        )

        assert "3f2504e0-4f89-11d3-9a0c-0305e82c3301" in rendered
        assert "unavailable" in rendered.lower()
        # Never silently cleared or dropped: the Select still carries the
        # saved id as its value, and the raw pick survives untouched.
        select = app.query_one("#settings-speech-default-profile", Select)
        assert select.value == "3f2504e0-4f89-11d3-9a0c-0305e82c3301"


@pytest.mark.asyncio
async def test_default_profile_store_loading_state_is_distinct_from_unavailable() -> (
    None
):
    """Regression coverage for task 3 review round 1's IMPORTANT finding:
    `_render_detail_pane` starts the fetch and reads the cache in the same
    call, so it is always `None` on first paint -- that must render as
    "loading", never "unavailable" (a healthy store must not look broken).
    """
    state = load_global_speech_tts_state({})
    state.defaults.default_profile_id = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
    # profiles=None with the default profiles_unavailable=False -- exactly
    # what `_render_detail_pane` passes on a session's first visit, before
    # its background fetch has had a chance to resolve.
    app = _PanelHarness(configure_provider="openai", state=state, profiles=None)
    async with app.run_test(size=(150, 60)) as pilot:
        await pilot.pause()
        rendered = str(
            app.query_one("#settings-speech-default-profile-note", Static).renderable
        )

        assert "loading" in rendered.lower()
        assert "unavailable" not in rendered.lower()
        # Still never dropped while loading.
        select = app.query_one("#settings-speech-default-profile", Select)
        assert select.value == "3f2504e0-4f89-11d3-9a0c-0305e82c3301"


@pytest.mark.asyncio
async def test_dangling_default_profile_id_is_flagged_when_the_store_is_available() -> (
    None
):
    """MINOR finding from task 3 review round 1: the store answers
    successfully, but the saved id isn't among the known profiles (e.g. the
    profile was deleted elsewhere). Distinct from "store unavailable" --
    same honesty requirement, different cause."""
    state = load_global_speech_tts_state({})
    state.defaults.default_profile_id = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
    app = _PanelHarness(
        configure_provider="openai",
        state=state,
        profiles=[("Some Other Voice", "9c858901-8a57-4791-81fe-4c455b099bc9")],
    )
    async with app.run_test(size=(150, 60)) as pilot:
        await pilot.pause()
        rendered = str(
            app.query_one("#settings-speech-default-profile-note", Static).renderable
        )

        assert "3f2504e0-4f89-11d3-9a0c-0305e82c3301" in rendered
        assert "not found" in rendered.lower() or "deleted" in rendered.lower()
        select = app.query_one("#settings-speech-default-profile", Select)
        assert select.value == "3f2504e0-4f89-11d3-9a0c-0305e82c3301"


@pytest.mark.asyncio
async def test_apply_profile_choices_preserves_the_current_selection() -> None:
    """The narrow live-update mechanism (`apply_profile_choices`, called by
    settings_screen.py once its background fetch resolves) must never reset
    an in-progress pick -- `Select.set_options()` resets the selection by
    default, which is exactly the "silently reverts the user's choice" bug
    this test guards against.
    """
    app = _PanelHarness(configure_provider="openai")
    async with app.run_test(size=(150, 60)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        select = app.query_one("#settings-speech-default-profile", Select)
        select.value = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
        await pilot.pause()

        panel.apply_profile_choices(
            [
                ("Test Voice", "3f2504e0-4f89-11d3-9a0c-0305e82c3301"),
                ("Another Voice", "9c858901-8a57-4791-81fe-4c455b099bc9"),
            ]
        )
        await pilot.pause()

        assert select.value == "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
        note = str(
            app.query_one("#settings-speech-default-profile-note", Static).renderable
        )
        assert note == ""


@pytest.mark.asyncio
async def test_saving_only_the_default_profile_settles_the_dirty_state() -> None:
    """Mutation-check target for the request_save()/has_unsaved_changes() fix.

    `default_profile_id` lives outside `TTSPreferencesSnapshot`, so a save
    that changes *only* this field must still update `original_state.defaults`
    on success -- otherwise the panel keeps reporting unsaved changes forever
    after a successful save (see request_save()'s `defaults_changed`).
    """
    app = _PanelHarness(configure_provider="openai")
    async with app.run_test(size=(150, 60)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        select = app.query_one("#settings-speech-default-profile", Select)
        select.value = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
        await pilot.pause()
        assert panel.has_unsaved_changes() is True

        request_id = panel.request_save()
        assert request_id is not None
        await pilot.pause()

        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=request_id,
                persisted=True,
                provider_statuses={"openai": "unchanged"},
                provider_configuration_revisions={"openai": 1},
                provider_runtime_revisions={"openai": 1},
            )
        )

        assert panel.has_unsaved_changes() is False
        assert panel.original_state.defaults.default_profile_id == (
            "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
        )


@pytest.mark.asyncio
async def test_default_profile_dirty_check_does_not_depend_on_provider_validity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """has_unsaved_changes()'s per-provider loop incidentally folds in a
    default-profile diff too (build_global_speech_tts_save_proposal computes
    it independent of `configure_provider`) -- but only when at least one
    provider's own fields still validate. If every provider proposal raises
    (all providers incomplete/invalid), that fallback can't run, so the
    explicit default_profile_id comparison is the only thing left standing
    between a real change and an incorrectly "clean" dirty state.
    """

    def _always_invalid(*_args: object, **_kwargs: object) -> None:
        raise GlobalSpeechTTSValidationError("openai", "base_url", "boom")

    monkeypatch.setattr(
        speech_tts_settings_panel_module,
        "build_global_speech_tts_save_proposal",
        _always_invalid,
    )
    app = _PanelHarness(configure_provider="openai")
    async with app.run_test(size=(150, 60)) as pilot:
        panel = app.query_one("#panel", SpeechTTSSettingsPanel)
        select = app.query_one("#settings-speech-default-profile", Select)
        select.value = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
        await pilot.pause()

        assert panel.has_unsaved_changes() is True
