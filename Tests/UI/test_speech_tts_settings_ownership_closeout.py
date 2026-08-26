"""Cross-surface Speech & TTS ownership closeout tests (TASK-1988)."""

from __future__ import annotations

import asyncio
import io
import tomllib
import wave
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual import on
from textual.app import App
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Collapsible, Input, Select, Static, TextArea

from Tests.TTS.test_console_speak_autoplay import _FakeApp
from Tests.TTS.test_stts_audio_cpp_generation import (
    _CountingStream,
    _NativeService,
    _Response,
)
from Tests.UI.speech_playground_fixtures import FakeTTSService, _resolved
from Tests.UI.test_destination_shells import (
    _active_destination_screen,
    _build_test_app,
    _wait_for_selector,
)
from Tests.UI.test_settings_speech_tts_panel import (
    _audio_cpp_state,
    _StyledDestinationHarness,
    _StyledPanelHarness,
)
from Tests.UI.test_speech_profile_navigation import (
    _playground_ready,
    _SpeechHost,
    _wait_until,
)
from Tests.UI.test_speech_playground_pane_lifecycle import _runtime_observation
from Tests.UI.test_studio_tts_preferences import _Host, _Store
from tldw_chatbook import config as config_module
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSEventHandler,
    STTSPlaygroundGenerateEvent,
    STTSSettingsSaveEvent,
    STTSSettingsSaveResult,
)
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSCompleteEvent,
    TTSPlaybackEvent,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.studio_preferences import (
    StudioTTSLoadResult,
    StudioTTSLoadState,
)
from tldw_chatbook.TTS.TTS_Generation import (
    TTSSettingsPublication,
    TTSSettingsPublicationTicket,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.UI.Speech.speech_settings_pane import SpeechSettingsPane

pytestmark = pytest.mark.asyncio

_BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "css"
    / "tldw_cli_modular.tcss"
)


class _StyledStudioHost(_Host):
    CSS_PATH = _BUNDLE


class _AccessiblePanelHarness(_StyledPanelHarness):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.notices: list[tuple[str, str]] = []

    def notify(
        self,
        message: str,
        *,
        severity: str = "information",
        **_kwargs: object,
    ) -> None:
        self.notices.append((message, severity))


class _SettingsReadOnlyTTSService:
    """Expose cached metadata while failing any provider operation."""

    def __init__(self) -> None:
        self.metadata_reads: list[tuple[str, str]] = []
        self.provider_operations: list[str] = []
        self._generation = 0
        self._provider_revisions: dict[str, int] = {}

    def latest_native_capability_observation(self, provider_id: str):
        self.metadata_reads.append(("observation", provider_id))
        return None

    def configuration_revision(self, provider_id: str) -> int:
        self.metadata_reads.append(("runtime_revision", provider_id))
        return self._provider_revisions.get(provider_id, 0)

    def saved_configuration_revision(self, provider_id: str) -> int:
        self.metadata_reads.append(("saved_revision", provider_id))
        return self._provider_revisions.get(provider_id, 0)

    def applied_configuration_revision(self, provider_id: str) -> int:
        self.metadata_reads.append(("applied_revision", provider_id))
        return self._provider_revisions.get(provider_id, 0)

    def begin_preferences_publication(
        self,
        preferences: TTSPreferencesSnapshot,
        provider_configs: Mapping[str, Mapping[str, object]],
        persistence,
        **_kwargs: object,
    ) -> TTSSettingsPublicationTicket:
        """Run the real local writer while replacing provider I/O with a fake."""

        self._generation += 1
        generation = self._generation
        foreground = asyncio.get_running_loop().create_future()

        async def publish() -> TTSSettingsPublication:
            outcome = await asyncio.to_thread(persistence)
            statuses = {
                provider_id: "applied" if outcome.file_replaced else "unchanged"
                for provider_id in provider_configs
            }
            if outcome.file_replaced:
                for provider_id in provider_configs:
                    self._provider_revisions[provider_id] = generation
            publication = TTSSettingsPublication(
                generation=generation,
                preferences=preferences,
                persistence=outcome,
                provider_statuses=statuses,
                provider_revisions=self._provider_revisions,
                published=outcome.file_replaced,
            )
            foreground.set_result(publication)
            return publication

        completion = asyncio.create_task(publish())
        return TTSSettingsPublicationTicket(generation, foreground, completion)

    def preferences_generation(self) -> int:
        return self._generation

    async def get_catalog(self, *_args, **_kwargs):
        self.provider_operations.append("catalog")
        raise AssertionError("Settings must not contact a TTS provider")

    async def synthesize(self, *_args, **_kwargs):
        self.provider_operations.append("synthesize")
        raise AssertionError("Settings must not synthesize")


class _FirstTimeSettingsHost(_StyledDestinationHarness):
    def __init__(self, service: _SettingsReadOnlyTTSService) -> None:
        app_instance = _build_test_app()
        app_instance.tts_service = service
        super().__init__(app_instance, "settings")
        self.save_events: list[STTSSettingsSaveEvent] = []
        self.navigation: list[NavigateToScreen] = []
        self._settings_handler = STTSEventHandler(app=self)
        self._settings_handler._stts_service = service

    @on(STTSSettingsSaveEvent)
    async def capture_save(self, event: STTSSettingsSaveEvent) -> None:
        self.save_events.append(event)
        event.stop()
        await self._settings_handler.handle_settings_save(event)

    def on_navigate_to_screen(self, message: NavigateToScreen) -> None:
        self.navigation.append(message)
        message.stop()


class _FirstTimeSpeechHost(_SpeechHost):
    def __init__(
        self,
        context: dict[str, object],
        native_service: object,
        *,
        restored_state: dict[str, object] | None = None,
    ) -> None:
        # Match production navigation order: construct, restore the bounded
        # process-local screen snapshot, then apply the incoming target.
        super().__init__(None)
        if restored_state is not None:
            self.screen_under_test.restore_state(restored_state)
        self.screen_under_test.apply_navigation_context(context)
        self.navigation: list[NavigateToScreen] = []
        self.notices: list[tuple[str, str]] = []
        self._stts_handler = STTSEventHandler(app=self)
        self._stts_handler._stts_service = native_service

    @on(STTSPlaygroundGenerateEvent)
    def generate(self, event: STTSPlaygroundGenerateEvent) -> None:
        event.stop()
        self._stts_handler.start_playground_generation(event)

    @on(NavigateToScreen)
    def capture_navigation(self, event: NavigateToScreen) -> None:
        self.navigation.append(event)
        event.stop()

    def notify(
        self,
        message: str,
        *,
        title: str = "",
        severity: str = "information",
        timeout: float | None = None,
    ) -> None:
        del title, timeout
        self.notices.append((message, severity))


class _StudioNativeService(_NativeService):
    def __init__(self, response: _Response) -> None:
        super().__init__(response)
        self.effective_calls: list[dict[str, object]] = []

    async def synthesize_effective(self, **kwargs: object):
        self.effective_calls.append(kwargs)
        return self.response, SimpleNamespace(
            provider_id="audio_cpp",
            model_id="second-model",
            voice_id="second-voice",
            response_format="wav",
            speed=1.0,
            provider_options={},
            revisions=SimpleNamespace(provider_configuration=3),
            sources={},
            provider_option_sources={},
            studio_preview=False,
        )


def _complete_wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8_000)
        wav_file.writeframes(b"\x00\x00" * 80)
    return buffer.getvalue()


def _audio_cpp_studio_pane(store: _Store | None = None) -> SpeechSettingsPane:
    store = store or _Store()
    return SpeechSettingsPane(
        id="speech-settings-pane",
        store=store,
        global_preferences=TTSPreferencesSnapshot(
            provider_id="audio_cpp",
            model_mode="first_available",
            model_id=None,
            voice_mode="server_default",
            voice_id=None,
            response_format="wav",
            speed=1.0,
        ),
        load_result=StudioTTSLoadResult(
            store.snapshot,
            StudioTTSLoadState.LOADED,
        ),
    )


def _rendered_text(widget: Static) -> str:
    return str(widget.render()).strip()


def _assert_field_rows_are_labelled(
    root: Widget,
    *,
    row_selector: str,
    label_selector: str,
) -> None:
    rows_seen = 0
    controls_seen = 0
    for row in root.query(row_selector):
        controls = list(row.query("Input, Select, Switch"))
        if not controls:
            continue
        rows_seen += 1
        label = _rendered_text(row.query_one(label_selector, Static))
        assert label, f"Empty visible label in {row!r}"
        for control in controls:
            controls_seen += 1
            assert control.id, f"Labelled control has no stable id: {control!r}"
            assert control.tooltip, f"{control.id} has no programmatic label"
            assert label.casefold() in str(control.tooltip).casefold(), (
                f"{control.id} tooltip does not identify visible label {label!r}"
            )
    assert rows_seen
    assert controls_seen


async def _assert_keyboard_reaches_in_order(
    app: App[None],
    pilot,
    ordered_targets: tuple[Widget, ...],
) -> None:
    first, *remaining = ordered_targets
    first.focus()
    await pilot.pause()
    assert app.focused is first

    focus_chain = tuple(app.screen.focus_chain)
    first_index = focus_chain.index(first)
    target_indices = tuple(focus_chain.index(target) for target in remaining)
    assert target_indices == tuple(sorted(target_indices)), (
        "Expected controls do not follow their declared visual order"
    )
    assert all(index > first_index for index in target_indices), (
        "Expected control appears before the starting control in this focus cycle"
    )

    next_target = 0
    for _ in range(target_indices[-1] - first_index):
        await pilot.press("tab")
        await pilot.pause()
        focused = app.focused
        assert focused is not first, "Focus wrapped before reaching every target"
        if focused not in remaining:
            continue
        expected = remaining[next_target]
        assert focused is expected, (
            f"Focus reached {focused!r} before visually earlier {expected!r}"
        )
        next_target += 1
        if next_target == len(remaining):
            return
    raise AssertionError(f"Tab never reached {remaining[next_target]!r}")


async def test_keyboard_order_gate_rejects_targets_reached_only_after_wrap() -> None:
    """The accessibility gate cannot pass by traversing a second focus cycle."""

    app = _StyledPanelHarness(state=_audio_cpp_state(saved_provider=True))
    async with app.run_test(size=(150, 55)) as pilot:
        panel = app.query_one("#panel")
        with pytest.raises(AssertionError, match="before the starting control"):
            await _assert_keyboard_reaches_in_order(
                app,
                pilot,
                (
                    panel.query_one("#settings-speech-save", Button),
                    panel.query_one("#settings-speech-default-provider", Select),
                ),
            )


async def test_global_and_studio_controls_have_programmatic_labels_and_text_states() -> (
    None
):
    """Visible labels, state text, and fixed-control reasons do not depend on color."""

    global_app = _StyledPanelHarness(state=_audio_cpp_state(saved_provider=True))
    async with global_app.run_test(size=(150, 55)) as pilot:
        await pilot.pause()
        panel = global_app.query_one("#panel")
        _assert_field_rows_are_labelled(
            panel,
            row_selector=".settings-speech-input-row",
            label_selector=".settings-input-label",
        )
        assert "requires WAV output and speed 1.0" in _rendered_text(
            panel.query_one("#settings-speech-default-constraints", Static)
        )
        assert panel.query_one("#settings-speech-output-format", Select).disabled
        assert panel.query_one("#settings-speech-speed", Input).disabled
        for row_id in (
            "provider-configuration",
            "provider-runtime",
            "catalog-freshness",
            "stt-dependency",
            "kokoro-dependency",
            "chatterbox-dependency",
            "higgs-dependency",
        ):
            copy = _rendered_text(
                panel.query_one(f"#settings-speech-status-{row_id}", Static)
            )
            assert ":" in copy and len(copy.split(":", 1)[1].strip()) > 0

    store = _Store()
    studio = _audio_cpp_studio_pane(store)
    studio_app = _StyledStudioHost(studio)
    async with studio_app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        _assert_field_rows_are_labelled(
            studio,
            row_selector=".studio-tts-setting-row",
            label_selector=".speech-setting-label",
        )
        assert studio.query_one("#studio-tts-format", Select).disabled
        assert studio.query_one("#studio-tts-speed", Input).disabled
        assert "Fixed by audio.cpp: WAV" in _rendered_text(
            studio.query_one("#studio-tts-format-source", Static)
        )
        assert "Fixed by audio.cpp: 1.0" in _rendered_text(
            studio.query_one("#studio-tts-speed-source", Static)
        )
        assert "only the Speech Studio" in _rendered_text(
            studio.query_one("#studio-tts-scope", Static)
        )


async def test_global_details_disclosures_are_accessible_and_unframed() -> None:
    app = _AccessiblePanelHarness(
        state=_audio_cpp_state(saved_provider=True),
        observation=None,
        current_configuration_revision=41,
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        panel = app.query_one("#panel")
        focus_chain = tuple(app.screen.focus_chain)

        for selector in (
            "#settings-speech-details",
            "#settings-speech-scope-inspector",
        ):
            disclosure = panel.query_one(selector, Collapsible)
            title = disclosure.query_one("CollapsibleTitle")
            assert disclosure.collapsed is True
            assert title.can_focus
            assert title in focus_chain
            assert not any(
                ancestor.has_class("settings-focus-card")
                for ancestor in disclosure.ancestors
            )

        primary = " ".join(
            _rendered_text(widget)
            for widget in panel.query(Static)
            if widget.display and widget.is_on_screen
        ).casefold()
        assert "current status" in primary
        assert "revision" not in primary
        assert "provider_configuration" not in primary


async def test_keyboard_order_reaches_primary_actions_and_status_does_not_steal_focus() -> (
    None
):
    """Both owners are usable by Tab/Enter and status changes retain focus."""

    global_app = _AccessiblePanelHarness(state=_audio_cpp_state(saved_provider=True))
    async with global_app.run_test(size=(150, 55)) as pilot:
        panel = global_app.query_one("#panel")
        await pilot.pause()
        panel.query_one(
            "#settings-speech-audio_cpp-base-url", Input
        ).value = "http://127.0.0.1:18001"
        await _assert_keyboard_reaches_in_order(
            global_app,
            pilot,
            (
                panel.query_one("#settings-speech-default-provider", Select),
                panel.query_one("#settings-speech-model-policy", Select),
                panel.query_one("#settings-speech-voice-policy", Select),
                panel.query_one("#settings-speech-configure-provider", Select),
                panel.query_one("#settings-speech-audio_cpp-base-url", Input),
                panel.query_one("CollapsibleTitle"),
                panel.query_one("#settings-speech-save", Button),
            ),
        )
        focused = global_app.focused
        panel._set_result("Global settings saved.", severity="information")
        await pilot.pause()
        assert getattr(global_app.focused, "id", None) == focused.id
        assert (
            _rendered_text(panel.query_one("#settings-speech-save-result", Static))
            == "Global settings saved."
        )
        await pilot.press("enter")
        await _wait_until(pilot, lambda: len(global_app.events) == 1)
        save = global_app.events[0]
        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=save.request_id,
                persisted=True,
                provider_statuses={"audio_cpp": "applied"},
                provider_configuration_revisions={"audio_cpp": 1},
                provider_runtime_revisions={"audio_cpp": 1},
            )
        )
        await pilot.pause()
        assert getattr(global_app.focused, "id", None) == focused.id
        assert global_app.notices[-1][1] == "information"
        assert "Saved locally" in global_app.notices[-1][0]

    store = _Store()
    studio = _audio_cpp_studio_pane(store)
    studio_app = _StyledStudioHost(studio)
    async with studio_app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        await _assert_keyboard_reaches_in_order(
            studio_app,
            pilot,
            (
                studio.query_one("#studio-tts-save-btn", Button),
                studio.query_one("#studio-tts-revert-btn", Button),
                studio.query_one("#studio-tts-reset-btn", Button),
                studio.query_one("#studio-tts-open-global-btn", Button),
                studio.query_one("#studio-tts-provider", Select),
                studio.query_one("#studio-tts-model-mode", Select),
                studio.query_one("#studio-tts-voice-mode", Select),
            ),
        )
        save_button = studio.query_one("#studio-tts-save-btn", Button)
        save_button.focus()
        focused = save_button
        await pilot.press("enter")
        await _wait_until(pilot, lambda: len(store.saved) == 1)
        await studio_app.workers.wait_for_complete()
        assert studio_app.focused is focused
        assert _rendered_text(studio.query_one("#studio-tts-status", Static)) == (
            "Studio preferences saved."
        )
        assert studio_app.notices[-1] == (
            "Studio preferences saved.",
            "information",
        )


async def test_keyboard_invalid_save_and_dirty_leave_cancel_restore_focus() -> None:
    """Validation and dirty-leave recovery work through keyboard controls."""

    global_app = _AccessiblePanelHarness(state=_audio_cpp_state(saved_provider=True))
    async with global_app.run_test(size=(150, 55)) as pilot:
        panel = global_app.query_one("#panel")
        base_url = panel.query_one("#settings-speech-audio_cpp-base-url", Input)
        save = panel.query_one("#settings-speech-save", Button)
        base_url.value = "ftp://invalid.example"
        save.focus()
        await pilot.press("enter")
        await pilot.pause()

        assert global_app.events == []
        assert global_app.focused is base_url
        assert global_app.notices[-1][1] == "error"

        base_url.value = "http://127.0.0.1:18001"
        open_lab = panel.query_one("#settings-speech-open-lab-bottom", Button)
        open_lab.focus()
        leave_worker = panel.run_worker(
            panel.confirm_leave(),
            group="test-global-dirty-leave",
            exclusive=True,
            exit_on_error=False,
        )
        await _wait_until(
            pilot,
            lambda: (
                len(global_app.screen.query("#global-speech-tts-leave-cancel")) == 1
            ),
        )
        cancel = global_app.screen.query_one("#global-speech-tts-leave-cancel", Button)
        cancel.focus()
        await pilot.press("enter")
        assert await leave_worker.wait() is False
        await pilot.pause()

        assert global_app.navigation == []
        assert global_app.focused is open_lab

    store = _Store()
    studio = _audio_cpp_studio_pane(store)
    studio_app = _StyledStudioHost(studio)
    async with studio_app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        studio.query_one("#studio-tts-model-mode", Select).value = "exact"
        model_id = studio.query_one("#studio-tts-model-id", Input)
        model_id.value = ""
        save = studio.query_one("#studio-tts-save-btn", Button)
        save.focus()
        await pilot.press("enter")
        await studio_app.workers.wait_for_complete()
        await pilot.pause()

        assert store.saved == []
        assert studio_app.focused is model_id
        assert studio_app.notices[-1][1] == "error"

        model_id.value = "second-model"
        open_global = studio.query_one("#studio-tts-open-global-btn", Button)
        open_global.focus()
        await pilot.press("enter")
        await _wait_until(
            pilot,
            lambda: len(studio_app.screen.query("#studio-tts-leave-cancel")) == 1,
        )
        cancel = studio_app.screen.query_one("#studio-tts-leave-cancel", Button)
        cancel.focus()
        await pilot.press("enter")
        await studio_app.workers.wait_for_complete()
        await pilot.pause()

        assert studio_app.navigation == []
        assert studio_app.focused is open_global


async def test_delayed_global_save_completion_preserves_new_keyboard_focus() -> None:
    """A late result must not move focus back to the control that started Save."""

    app = _AccessiblePanelHarness(state=_audio_cpp_state(saved_provider=True))
    async with app.run_test(size=(150, 55)) as pilot:
        panel = app.query_one("#panel")
        panel.query_one(
            "#settings-speech-audio_cpp-base-url", Input
        ).value = "http://127.0.0.1:18001"
        save = panel.query_one("#settings-speech-save", Button)
        save.focus()
        await pilot.press("enter")
        await _wait_until(pilot, lambda: len(app.events) == 1)

        moved_to = panel.query_one("#settings-speech-configure-provider", Select)
        moved_to.focus()
        await pilot.pause()
        assert app.focused is moved_to

        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=app.events[0].request_id,
                persisted=True,
                provider_statuses={"audio_cpp": "applied"},
                provider_configuration_revisions={"audio_cpp": 1},
                provider_runtime_revisions={"audio_cpp": 1},
            )
        )
        await pilot.pause()

        assert getattr(app.focused, "id", None) == moved_to.id
        assert getattr(app.focused, "id", None) != save.id


async def test_programmatic_global_save_does_not_steal_later_field_focus() -> None:
    """A save begun off-button must not treat the first user move as automatic."""

    app = _AccessiblePanelHarness(state=_audio_cpp_state(saved_provider=True))
    async with app.run_test(size=(150, 55)) as pilot:
        panel = app.query_one("#panel")
        base_url = panel.query_one("#settings-speech-audio_cpp-base-url", Input)
        base_url.value = "http://127.0.0.1:18001"
        base_url.focus()
        await pilot.pause()
        assert app.focused is base_url
        panel.request_save()
        await _wait_until(pilot, lambda: len(app.events) == 1)

        moved_to = panel.query_one("#settings-speech-configure-provider", Select)
        moved_to.focus()
        await pilot.pause()
        assert app.focused is moved_to

        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=app.events[0].request_id,
                persisted=True,
                provider_statuses={"audio_cpp": "applied"},
                provider_configuration_revisions={"audio_cpp": 1},
                provider_runtime_revisions={"audio_cpp": 1},
            )
        )
        await pilot.pause()

        assert getattr(app.focused, "id", None) == moved_to.id


async def test_delayed_audio_cpp_save_preserves_advanced_disclosure_focus() -> None:
    """Recompose restores the id-less Advanced title the user moved to."""

    app = _AccessiblePanelHarness(state=_audio_cpp_state(saved_provider=True))
    async with app.run_test(size=(150, 55)) as pilot:
        panel = app.query_one("#panel")
        base_url = panel.query_one("#settings-speech-audio_cpp-base-url", Input)
        base_url.value = "http://127.0.0.1:18001"
        base_url.focus()
        await pilot.pause()
        panel.request_save()
        await _wait_until(pilot, lambda: len(app.events) == 1)

        panel.query_one("CollapsibleTitle").focus()
        await pilot.pause()
        assert app.focused is panel.query_one("CollapsibleTitle")

        panel.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=app.events[0].request_id,
                persisted=True,
                provider_statuses={"audio_cpp": "applied"},
                provider_configuration_revisions={"audio_cpp": 1},
                provider_runtime_revisions={"audio_cpp": 1},
            )
        )
        await pilot.pause()

        assert app.focused is panel.query_one("CollapsibleTitle")


async def test_supported_narrow_layout_uses_vertical_scroll_without_clipping_actions() -> (
    None
):
    """Primary global and Studio controls stay horizontally reachable at 80x24."""

    global_app = _StyledDestinationHarness(_build_test_app(), "settings")
    async with global_app.run_test(size=(80, 24)) as pilot:
        await global_app.workers.wait_for_complete()
        await pilot.pause()
        screen = _active_destination_screen(global_app)
        screen.query_one("#settings-category-speech-tts", Button).press()
        await _wait_for_selector(
            screen,
            pilot,
            "#settings-speech-tts-panel",
            timeout=8.0,
        )
        configure = screen.query_one("#settings-speech-configure-provider", Select)
        configure.value = "audio_cpp"
        await _wait_for_selector(
            screen,
            pilot,
            "#settings-speech-audio_cpp-base-url",
            timeout=8.0,
        )
        detail = screen.query_one("#settings-detail-pane-body", VerticalScroll)
        assert detail.max_scroll_x == 0
        for selector in (
            "#settings-speech-default-provider",
            "#settings-speech-configure-provider",
            "#settings-speech-audio_cpp-base-url",
            "#settings-speech-save",
            "#settings-speech-open-lab-bottom",
        ):
            control = screen.query_one(selector)
            control.scroll_visible(animate=False)
            await pilot.pause()
            assert 0 <= control.region.x
            assert control.region.x + control.region.width <= global_app.size.width
            assert 0 <= control.region.y < global_app.size.height

    studio = _audio_cpp_studio_pane()
    studio_app = _StyledStudioHost(studio)
    async with studio_app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        groups = studio.query_one("#speech-settings-groups", VerticalScroll)
        assert groups.max_scroll_x == 0
        assert studio.has_class("studio-tts-settings-stacked")
        for selector in (
            "#studio-tts-save-btn",
            "#studio-tts-open-global-btn",
            "#studio-tts-provider",
            "#studio-tts-reset-btn",
        ):
            control = studio.query_one(selector)
            control.scroll_visible(animate=False)
            await pilot.pause()
            assert 0 <= control.region.x
            assert control.region.x + control.region.width <= studio_app.size.width
            assert 0 <= control.region.y < studio_app.size.height


async def test_first_time_audio_cpp_setup_lab_generation_and_console_handoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A fake-only first run crosses real Settings, Lab, and playback seams."""

    config_path = tmp_path / "first-time-config.toml"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(config_module, "_CONFIG_CACHE", None)
    monkeypatch.setattr(config_module, "_CONFIG_CACHE_SOURCE", None)
    monkeypatch.setattr(config_module, "_SETTINGS_CACHE", None)
    monkeypatch.setattr(config_module, "_SETTINGS_CACHE_SOURCE", None)
    monkeypatch.setattr(config_module, "_CONFIG_GENERATION", 0)
    monkeypatch.setattr(
        config_module,
        "settings",
        {"COMPREHENSIVE_CONFIG_RAW": {"app_tts": {}}},
    )
    settings_service = _SettingsReadOnlyTTSService()
    settings_host = _FirstTimeSettingsHost(settings_service)
    async with settings_host.run_test(size=(150, 55)) as pilot:
        await settings_host.workers.wait_for_complete()
        await pilot.pause()
        screen = _active_destination_screen(settings_host)
        search = screen.query_one("#settings-category-search", Input)
        search.value = "audio.cpp"
        search.focus()
        await pilot.press("enter")
        await _wait_for_selector(
            screen,
            pilot,
            "#settings-speech-audio_cpp-base-url",
            timeout=8.0,
        )

        fake_url = "http://127.0.0.1:18001"
        screen.query_one("#settings-speech-audio_cpp-base-url", Input).value = fake_url
        screen.query_one("#settings-speech-save", Button).press()
        await _wait_until(
            pilot,
            lambda: (
                len(settings_host.save_events) == 1
                and settings_host.save_events[0].reply_to._latest_request_id is None
            ),
        )

        save = settings_host.save_events[0]
        assert save.settings["audio_cpp"]["base_url"] == fake_url
        assert settings_service.provider_operations == []
        persisted = tomllib.loads(config_path.read_text(encoding="utf-8"))
        assert persisted["app_tts"]["audio_cpp"]["base_url"] == fake_url
        await pilot.pause()

        screen.query_one("#settings-speech-open-lab-bottom", Button).press()
        await _wait_until(pilot, lambda: len(settings_host.navigation) == 1)

    lab_context = settings_host.navigation[0].screen_context
    assert settings_host.navigation[0].screen_name == "stts"
    assert lab_context == {
        "view": "playground",
        "provider": "audio_cpp",
        "intent": "test",
    }
    assert settings_service.provider_operations == []

    reloaded_settings = _FirstTimeSettingsHost(settings_service)
    async with reloaded_settings.run_test(size=(150, 55)) as pilot:
        await reloaded_settings.workers.wait_for_complete()
        await pilot.pause()
        screen = _active_destination_screen(reloaded_settings)
        search = screen.query_one("#settings-category-search", Input)
        search.value = "audio.cpp"
        search.focus()
        await pilot.press("enter")
        await _wait_for_selector(
            screen,
            pilot,
            "#settings-speech-audio_cpp-base-url",
            timeout=8.0,
        )
        assert (
            screen.query_one("#settings-speech-audio_cpp-base-url", Input).value
            == fake_url
        )
        assert settings_service.provider_operations == []

    catalog_service = FakeTTSService()
    runtime_observation = _runtime_observation(
        saved_mode="external",
        applied_mode="external",
        process_state="running",
        process_generation=1,
        capability="available",
        endpoint=fake_url,
        active_endpoint=fake_url,
        catalog_revision=11,
        catalog_fresh=True,
    )

    async def start_and_test_audio_cpp() -> object:
        return await catalog_service.get_catalog("audio_cpp", refresh=True)

    async def audio_cpp_runtime_observation(
        *, selected_model_id: str | None = None
    ) -> object:
        del selected_model_id
        return runtime_observation

    catalog_service.start_and_test_audio_cpp = start_and_test_audio_cpp  # type: ignore[attr-defined]
    catalog_service.audio_cpp_runtime_observation = audio_cpp_runtime_observation  # type: ignore[attr-defined]
    native_service = _StudioNativeService(
        _Response(
            _CountingStream((_complete_wav(),)),
            model_id="second-model",
        )
    )
    global_preferences = TTSPreferencesSnapshot(
        provider_id="audio_cpp",
        model_mode="first_available",
        model_id=None,
        voice_mode="server_default",
        voice_id=None,
        response_format="wav",
        speed=1.0,
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(catalog_service),
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_check_higgs_installation",
        lambda self: None,
    )
    monkeypatch.setattr(
        SpeechSettingsPane,
        "_read_global_preferences",
        staticmethod(lambda: global_preferences),
    )
    speech_host = _FirstTimeSpeechHost(dict(lab_context), native_service)
    speech_screen = speech_host.screen_under_test
    saved_speech_state: dict[str, object] | None = None

    try:
        async with speech_host.run_test(size=(150, 55)) as pilot:
            await _wait_until(
                pilot,
                lambda: (
                    _playground_ready(speech_screen)
                    and catalog_service.catalog_calls
                    and getattr(speech_host.focused, "id", None)
                    == "tts-test-connection-btn"
                ),
            )
            playground = speech_screen.query_one(SpeechPlaygroundPane)
            initial_refreshes = sum(
                refresh for _provider, refresh in catalog_service.catalog_calls
            )

            playground.query_one("#tts-refresh-catalog-btn", Button).press()
            await _wait_until(
                pilot,
                lambda: (
                    sum(refresh for _provider, refresh in catalog_service.catalog_calls)
                    == initial_refreshes + 1
                ),
            )
            await _wait_until(
                pilot,
                lambda: (
                    playground._audio_cpp_lifecycle_busy is None
                    and not playground.query_one(
                        "#tts-refresh-catalog-btn", Button
                    ).disabled
                ),
            )

            playground.query_one("#tts-model-select", Select).value = "second-model"
            await _wait_until(
                pilot,
                lambda: any(
                    provider == "audio_cpp" and model == "second-model"
                    for provider, model, _refresh in catalog_service.voice_calls
                ),
            )
            playground.query_one("#tts-voice-select", Select).value = "second-voice"
            await pilot.pause()
            playground.query_one("#tts-refresh-catalog-btn", Button).press()
            await _wait_until(
                pilot,
                lambda: (
                    sum(refresh for _provider, refresh in catalog_service.catalog_calls)
                    == initial_refreshes + 2
                ),
            )
            await pilot.pause()
            assert playground.query_one("#tts-model-select", Select).value == (
                "second-model"
            )
            assert playground.query_one("#tts-voice-select", Select).value == (
                "second-voice"
            )

            playground.query_one(
                "#tts-text-input", TextArea
            ).text = "Welcome back, traveler. The lanterns are waiting."
            await _wait_until(
                pilot,
                lambda: not playground.query_one("#tts-generate-btn", Button).disabled,
            )
            playground.query_one("#tts-generate-btn", Button).press()
            await _wait_until(
                pilot,
                lambda: (
                    speech_host._stts_handler.playground_state().artifact is not None
                ),
            )
            artifact = speech_host._stts_handler.playground_state().artifact
            assert artifact is not None
            assert playground.current_audio_artifact == artifact
            assert playground.current_audio_artifact is not artifact
            assert not playground.query_one("#audio-play-btn", Button).disabled
            with wave.open(str(artifact.path), "rb") as wav_file:
                assert wav_file.getnchannels() == 1
                assert wav_file.getsampwidth() == 2
                assert wav_file.getframerate() == 8_000
                assert wav_file.getnframes() == 80

            console = _FakeApp(widgets=())
            await TldwCli.handle_tts_complete_event(
                console,
                TTSCompleteEvent(
                    message_id="roleplay-response-1",
                    audio_file=artifact.path,
                ),
            )
            playback = [
                event for event in console.posted if isinstance(event, TTSPlaybackEvent)
            ]
            assert [(event.action, event.message_id) for event in playback] == [
                ("play", "roleplay-response-1")
            ]

            playground.query_one("#tts-open-studio-preferences-btn", Button).press()
            await _wait_until(
                pilot,
                lambda: (
                    speech_screen.stts_window is not None
                    and speech_screen.stts_window.current_view == "settings"
                    and len(speech_screen.query(SpeechSettingsPane)) == 1
                    and speech_screen.query_one(SpeechSettingsPane)._load_applied
                ),
            )
            studio = speech_screen.query_one(SpeechSettingsPane)
            studio.query_one("#studio-tts-open-global-btn", Button).press()
            await _wait_until(pilot, lambda: len(speech_host.navigation) == 1)
            saved_speech_state = speech_screen.save_state()

        assert native_service.legacy_calls == 0
        assert len(native_service.effective_calls) == 1
        assert speech_host.navigation[0].screen_name == "settings"
        assert speech_host.navigation[0].screen_context == {
            "category": "speech-tts",
            "provider": "audio_cpp",
            "intent": "configure",
        }
    finally:
        await speech_host._stts_handler.cleanup_tts_resources()

    assert saved_speech_state is not None
    assert set(saved_speech_state["speech_playground_axes"]) == {
        "tts-provider-select",
        "tts-model-select",
        "tts-voice-select",
        "tts-format-select",
        "tts-speed-input",
    }
    assert "Welcome back" not in repr(saved_speech_state)

    returned_host = _FirstTimeSpeechHost(
        {
            "view": "playground",
            "provider": "audio_cpp",
            "intent": "test",
        },
        native_service,
        restored_state=saved_speech_state,
    )
    returned_screen = returned_host.screen_under_test
    try:
        async with returned_host.run_test(size=(150, 55)) as pilot:
            await _wait_until(
                pilot,
                lambda: (
                    _playground_ready(returned_screen)
                    and getattr(returned_host.focused, "id", None)
                    == "tts-test-connection-btn"
                ),
            )
            returned_playground = returned_screen.query_one(SpeechPlaygroundPane)
            assert (
                returned_playground.query_one("#tts-model-select", Select).value
                == "second-model"
            )
            assert (
                returned_playground.query_one("#tts-voice-select", Select).value
                == "second-voice"
            )
    finally:
        await returned_host._stts_handler.cleanup_tts_resources()
