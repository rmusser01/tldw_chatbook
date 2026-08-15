"""Studio-only TTS preferences for the Speech Lab.

The global Speech & TTS category owns credentials, endpoints, initialization,
and application defaults.  This pane owns only sparse Studio overrides plus
the two Chatterbox values proven request-local by TASK-1981.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Literal

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import QueryError
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static, Switch

from tldw_chatbook.TTS.legacy_catalogs import (
    LEGACY_DEFAULT_MODELS,
    LEGACY_DEFAULT_VOICES,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.profile_service import TTSPlaygroundSelectionPreset
from tldw_chatbook.TTS.provider_ids import BUILT_IN_TTS_PROVIDER_IDS
from tldw_chatbook.TTS.studio_preferences import (
    STUDIO_TTS_PROVIDER_OPTION_KEYS,
    StudioTTSLoadResult,
    StudioTTSLoadState,
    StudioTTSPreferenceStore,
    StudioTTSPreferencesSnapshot,
    StudioTTSSelectionOverrides,
    StudioTTSWriteStatus,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.Widgets.prune_safe_select import PruneSafeSelect

from ..Workbench.workbench_state import WorkbenchAction
from .speech_action_strip import SpeechActionStrip
from .speech_runtime_status import speech_tts_navigation_context
from .speech_settings_mixin import SpeechSettingsMixin
from .speech_settings_contracts import (
    SpeechTTSNavigationIntent,
    SpeechTTSNavigationTarget,
)

_INHERIT = "__inherit__"
_PROVIDER_LABELS: Mapping[str, str] = {
    "openai": "OpenAI",
    "audio_cpp": "audio.cpp",
    "elevenlabs": "ElevenLabs",
    "kokoro": "Kokoro",
    "chatterbox": "Chatterbox",
    "higgs": "Higgs",
    "alltalk": "AllTalk",
}
_PROVIDER_OPTIONS = tuple(
    (label, provider_id) for provider_id, label in _PROVIDER_LABELS.items()
)
_FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")
_STUDIO_SETTINGS_STACK_WIDTH = 104

STUDIO_ACTIONS: tuple[WorkbenchAction, ...] = (
    WorkbenchAction(
        id="studio-tts-save-btn",
        label="Save Studio Preferences",
        tooltip="Persist only Speech Studio overrides",
        primary=True,
    ),
    WorkbenchAction(
        id="studio-tts-revert-btn",
        label="Revert",
        tooltip="Restore the last saved Studio snapshot",
    ),
    WorkbenchAction(
        id="studio-tts-reset-btn",
        label="Reset to Global",
        tooltip="Delete every Studio override",
    ),
    WorkbenchAction(
        id="studio-tts-open-global-btn",
        label="Open Global Speech & TTS Settings",
        tooltip="Edit application-wide provider setup and defaults",
    ),
)

VOICE_DESTINATION_ACTIONS: tuple[WorkbenchAction, ...] = (
    WorkbenchAction(
        id="voice-profiles",
        label="Voice Profiles",
        tooltip="Open provider-neutral saved voice profiles",
    ),
    WorkbenchAction(
        id="voice-blends",
        label="Voice Blends",
        tooltip="Open Kokoro voice blending",
    ),
)

VOICE_BLEND_ACTIONS: tuple[WorkbenchAction, ...] = (
    WorkbenchAction(
        id="add-voice-blend-btn",
        label="Add Voice Blend",
        tooltip="Create a reusable Kokoro voice blend",
    ),
    WorkbenchAction(
        id="import-blends-btn",
        label="Import Voice Blends",
        tooltip="Import Kokoro voice blends",
    ),
    WorkbenchAction(
        id="export-blends-btn",
        label="Export Voice Blends",
        tooltip="Export Kokoro voice blends",
    ),
)

LeaveChoice = Literal["save", "discard", "cancel"]


class StudioPreferencesSaved(Message):
    """Report a newly persisted Studio snapshot to the owning Lab window."""

    def __init__(
        self,
        snapshot: StudioTTSPreferencesSnapshot,
        *,
        reset_to_global: bool = False,
    ) -> None:
        super().__init__()
        if type(snapshot) is not StudioTTSPreferencesSnapshot:
            raise TypeError("Studio preferences message requires an exact snapshot")
        if type(reset_to_global) is not bool:
            raise TypeError("reset_to_global must be a bool")
        self.snapshot = snapshot
        self.reset_to_global = reset_to_global


class SpeechDestinationRequested(Message):
    """Request one exact Speech Lab destination from a child pane."""

    def __init__(self, destination_id: str) -> None:
        super().__init__()
        if destination_id not in {"voice-profiles", "voice-blends"}:
            raise ValueError("unknown Speech destination")
        self.destination_id = destination_id


class SpeechDestinationBackRequested(Message):
    """Return from a voice tool to its originating Speech view."""


class StudioTTSLeaveModal(ModalScreen[LeaveChoice]):
    """Protect one dirty Studio draft before its owning widget is removed."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def compose(self) -> ComposeResult:
        with Vertical(
            id="studio-tts-leave-modal",
            classes="settings-rag-profile-modal",
        ):
            yield Static(
                "Unsaved Studio TTS preferences", classes="destination-section"
            )
            yield Static(
                "Save these Studio-only changes before continuing, or discard them?",
                classes="settings-detail-row",
            )
            with Horizontal(classes="settings-action-row"):
                yield Button("Cancel", id="studio-tts-leave-cancel")
                yield Button(
                    "Discard and continue",
                    id="studio-tts-leave-discard",
                )
                yield Button(
                    "Save and continue",
                    id="studio-tts-leave-save",
                    variant="primary",
                )

    def action_cancel(self) -> None:
        self.dismiss("cancel")

    @on(Button.Pressed, "#studio-tts-leave-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss("cancel")

    @on(Button.Pressed, "#studio-tts-leave-discard")
    def _discard(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss("discard")

    @on(Button.Pressed, "#studio-tts-leave-save")
    def _save(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss("save")


def _field_row(
    label: str,
    control: Input | Select | Switch,
    *,
    source_id: str,
    error_id: str,
) -> Horizontal:
    """Build one compact setting row with source and validation copy."""

    if not control.tooltip:
        control.tooltip = label
    return Horizontal(
        Static(label, classes="speech-setting-label", markup=False),
        control,
        Static("", id=source_id, classes="studio-tts-source", markup=False),
        Static(
            "",
            id=error_id,
            classes="studio-tts-field-error hidden",
            markup=False,
        ),
        classes="speech-setting-row studio-tts-setting-row",
    )


class VoiceBlendsPane(SpeechSettingsMixin, Vertical):
    """Manage Kokoro-only voice blends without presenting them as profiles."""

    def __init__(self, **kwargs: Any) -> None:
        classes = kwargs.pop("classes", "")
        super().__init__(classes=f"speech-settings-pane {classes}".strip(), **kwargs)
        self.init_settings_state()

    def compose(self) -> ComposeResult:
        yield Static(
            "Voice Blends",
            id="voice-blends-heading",
            classes="speech-pane-title",
            markup=False,
        )
        yield Static(
            "Kokoro only. Blend IDs are available only while Kokoro is selected.",
            id="voice-blends-scope",
            classes="studio-tts-scope",
            markup=False,
        )
        yield SpeechActionStrip(VOICE_BLEND_ACTIONS, id="voice-blends-actions")
        yield Static(
            "Loading Kokoro voice blends…",
            id="kokoro-voice-blends-list",
            classes="studio-tts-helper",
        )
        yield Button(
            "Back to previous Speech view",
            id="speech-destination-back",
        )

    def on_mount(self) -> None:
        self._load_kokoro_voice_blends()

    @on(Button.Pressed, "#speech-destination-back")
    def _request_back(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(SpeechDestinationBackRequested())


class SpeechSettingsPane(SpeechSettingsMixin, Vertical):
    """Edit sparse Speech Studio preferences without touching global setup."""

    def __init__(
        self,
        *,
        store: StudioTTSPreferenceStore | None = None,
        global_preferences: TTSPreferencesSnapshot | None = None,
        load_result: StudioTTSLoadResult | None = None,
        adopted_preset: TTSPlaygroundSelectionPreset | None = None,
        **kwargs: Any,
    ) -> None:
        classes = kwargs.pop("classes", "")
        super().__init__(classes=f"speech-settings-pane {classes}".strip(), **kwargs)
        if global_preferences is not None and (
            type(global_preferences) is not TTSPreferencesSnapshot
        ):
            raise TypeError("global_preferences must be a TTS preferences snapshot")
        if load_result is not None and type(load_result) is not StudioTTSLoadResult:
            raise TypeError("load_result must be a Studio TTS load result")
        if adopted_preset is not None and (
            type(adopted_preset) is not TTSPlaygroundSelectionPreset
        ):
            raise TypeError("adopted_preset must be a TTS Playground preset")
        self._store = store or StudioTTSPreferenceStore()
        self._global_preferences = global_preferences or self._read_global_preferences()
        self._load_result = load_result
        self._adopted_preset = adopted_preset
        self._saved_snapshot = (
            load_result.snapshot
            if load_result is not None
            else StudioTTSPreferencesSnapshot()
        )
        self._active_provider_value = _INHERIT
        self._applying_controls = False
        self._dirty = False
        self._corrupt = False
        self._adoption_pending = False
        self._busy = False
        self._load_applied = False
        self._forced_global_provider_draft: StudioTTSPreferencesSnapshot | None = None

    @property
    def saved_snapshot(self) -> StudioTTSPreferencesSnapshot:
        """Return the last snapshot successfully loaded or persisted here."""

        return self._saved_snapshot

    @property
    def is_dirty(self) -> bool:
        """Return whether visible controls differ from the saved snapshot."""

        if not self.is_mounted:
            return self._dirty
        candidate = self._collect_candidate(show_errors=False)
        return candidate is None or candidate != self._ui_comparable_snapshot(
            self._saved_snapshot
        )

    def _ui_comparable_snapshot(
        self,
        snapshot: StudioTTSPreferencesSnapshot,
    ) -> StudioTTSPreferencesSnapshot:
        """Normalize fixed audio.cpp values the editor cannot meaningfully edit."""

        selection = snapshot.selection
        provider_id = selection.provider_id or self._global_preferences.provider_id
        if (
            provider_id != "audio_cpp"
            or selection.response_format not in (None, "wav")
            or selection.speed not in (None, 1.0)
        ):
            return snapshot
        return replace(
            snapshot,
            selection=replace(selection, response_format=None, speed=None),
        )

    @staticmethod
    def _read_global_preferences() -> TTSPreferencesSnapshot:
        """Read one cached global selection without provider I/O."""

        from tldw_chatbook.config import get_cli_setting

        missing = object()
        values: dict[str, object] = {}
        defaults: Mapping[str, object] = {
            "default_provider": "openai",
            "default_format": "mp3",
            "default_speed": 1.0,
        }
        for key in (
            "default_provider",
            "default_model_mode",
            "default_model",
            "default_voice_mode",
            "default_voice",
            "default_format",
            "default_speed",
        ):
            value = get_cli_setting("app_tts", key, missing)
            if value is not missing:
                values[key] = value
            elif key in defaults:
                values[key] = defaults[key]
        try:
            return TTSPreferencesSnapshot.from_settings({"app_tts": values})
        except (TypeError, ValueError):
            logger.warning(
                "Global TTS defaults are invalid; using safe OpenAI defaults"
            )
            return TTSPreferencesSnapshot.from_settings({})

    def refresh_global_preferences(self, snapshot: TTSPreferencesSnapshot) -> None:
        """Refresh inherited source truth without disturbing Studio edits."""

        if type(snapshot) is not TTSPreferencesSnapshot:
            raise TypeError("global preferences must be a TTS preferences snapshot")
        if not self.is_mounted:
            self._global_preferences = snapshot
            return

        provider_value = self.query_one("#studio-tts-provider", Select).value
        previous_effective_provider = self._effective_provider()
        draft = self._collect_candidate(show_errors=False)
        self._global_preferences = snapshot
        provider_changed = (
            provider_value == _INHERIT
            and previous_effective_provider != snapshot.provider_id
        )
        if provider_changed:
            retained_draft = self._forced_global_provider_draft
            if (
                previous_effective_provider == "audio_cpp"
                and retained_draft is not None
                and draft is not None
            ):
                draft = replace(
                    draft,
                    selection=replace(
                        draft.selection,
                        response_format=retained_draft.selection.response_format,
                        speed=retained_draft.selection.speed,
                    ),
                )
            draft = draft or retained_draft or self._saved_snapshot
            self._forced_global_provider_draft = (
                draft if snapshot.provider_id == "audio_cpp" else None
            )
            self._applying_controls = True
            try:
                self._populate_provider_controls(
                    _INHERIT,
                    draft,
                )
            finally:
                self._applying_controls = False
        self._sync_source_copy()
        self._sync_dirty_state()

    def compose(self) -> ComposeResult:
        yield Static(
            "Studio TTS Preferences",
            id="studio-tts-title",
            classes="speech-pane-title",
            markup=False,
        )
        yield Static(
            "These preferences affect only the Speech Studio. They never change "
            "global defaults or character TTS profiles.",
            id="studio-tts-scope",
            classes="studio-tts-scope",
            markup=False,
        )
        yield SpeechActionStrip(STUDIO_ACTIONS, id="speech-settings-actions")

        with VerticalScroll(id="speech-settings-groups"):
            yield Static(
                "Loading Studio preferences…",
                id="studio-tts-status",
                classes="speech-status-line",
                markup=False,
            )
            yield Static(
                "",
                id="studio-tts-adoption-status",
                classes="speech-status-line hidden",
                markup=False,
            )
            yield _field_row(
                "Provider override",
                PruneSafeSelect(
                    (("Inherit global", _INHERIT), *_PROVIDER_OPTIONS),
                    id="studio-tts-provider",
                    allow_blank=False,
                    classes="speech-setting-control",
                ),
                source_id="studio-tts-provider-source",
                error_id="studio-tts-provider-error",
            )
            yield _field_row(
                "Model policy",
                PruneSafeSelect(
                    (
                        ("Inherit", _INHERIT),
                        ("Exact", "exact"),
                        ("First available", "first_available"),
                    ),
                    id="studio-tts-model-mode",
                    allow_blank=False,
                    classes="speech-setting-control",
                ),
                source_id="studio-tts-model-source",
                error_id="studio-tts-model-mode-error",
            )
            yield _field_row(
                "Exact model ID",
                Input(
                    id="studio-tts-model-id",
                    classes="speech-setting-control",
                    placeholder="Case-sensitive model ID",
                ),
                source_id="studio-tts-model-id-source",
                error_id="studio-tts-model-id-error",
            )
            yield _field_row(
                "Voice policy",
                PruneSafeSelect(
                    (
                        ("Inherit", _INHERIT),
                        ("Exact", "exact"),
                        ("Server default", "server_default"),
                    ),
                    id="studio-tts-voice-mode",
                    allow_blank=False,
                    classes="speech-setting-control",
                ),
                source_id="studio-tts-voice-source",
                error_id="studio-tts-voice-mode-error",
            )
            yield _field_row(
                "Exact voice ID",
                Input(
                    id="studio-tts-voice-id",
                    classes="speech-setting-control",
                    placeholder="Case-sensitive voice ID",
                ),
                source_id="studio-tts-voice-id-source",
                error_id="studio-tts-voice-id-error",
            )
            yield _field_row(
                "Output format",
                PruneSafeSelect(
                    (
                        ("Inherit", _INHERIT),
                        *((value.upper(), value) for value in _FORMATS),
                    ),
                    id="studio-tts-format",
                    allow_blank=False,
                    classes="speech-setting-control",
                ),
                source_id="studio-tts-format-source",
                error_id="studio-tts-format-error",
            )
            yield _field_row(
                "Speed",
                Input(
                    id="studio-tts-speed",
                    classes="speech-setting-control",
                    placeholder="Inherit (0.25–4.0)",
                    type="number",
                ),
                source_id="studio-tts-speed-source",
                error_id="studio-tts-speed-error",
            )
            yield Static("Playback", classes="speech-section-head", markup=False)
            yield _field_row(
                "Play generated audio automatically",
                Switch(
                    id="studio-tts-auto-play",
                    classes="speech-setting-control",
                    value=False,
                ),
                source_id="studio-tts-auto-play-state",
                error_id="studio-tts-auto-play-error",
            )

            with Vertical(id="studio-tts-chatterbox-options", classes="hidden"):
                yield Static("Chatterbox request tuning", classes="speech-section-head")
                yield _field_row(
                    "Exaggeration",
                    Input(
                        id="chatterbox-exaggeration-input",
                        classes="speech-setting-control",
                        placeholder="Inherit (0.0–1.0)",
                        type="number",
                    ),
                    source_id="studio-tts-exaggeration-source",
                    error_id="studio-tts-exaggeration-error",
                )
                yield _field_row(
                    "CFG weight",
                    Input(
                        id="chatterbox-cfg-weight-input",
                        classes="speech-setting-control",
                        placeholder="Inherit (0.0–1.0)",
                        type="number",
                    ),
                    source_id="studio-tts-cfg-weight-source",
                    error_id="studio-tts-cfg-weight-error",
                )

            yield Static(
                "Voice tools",
                id="studio-tts-voice-tools-heading",
                classes="speech-section-head",
                markup=False,
            )
            yield Static(
                "Open provider-neutral Voice Profiles or Kokoro-only Voice Blends.",
                classes="studio-tts-helper",
                markup=False,
            )
            yield SpeechActionStrip(
                VOICE_DESTINATION_ACTIONS,
                id="studio-tts-voice-destination-actions",
            )

    def on_mount(self) -> None:
        """Apply injected state or load the Studio section off the UI pump."""

        self._sync_responsive_layout()
        if self._load_result is not None:
            self.call_after_refresh(self._apply_load_result, self._load_result)
            return
        self.run_worker(
            self._load_preferences(),
            group="studio-tts-preferences-load",
            exclusive=True,
            exit_on_error=False,
        )

    def on_resize(self) -> None:
        """Keep actions, sources, and validation copy visible when narrow."""

        self._sync_responsive_layout()

    def _sync_responsive_layout(self) -> None:
        """Stack the form when one horizontal row cannot hold all four cells."""

        self.set_class(
            self.size.width < _STUDIO_SETTINGS_STACK_WIDTH,
            "studio-tts-settings-stacked",
        )

    async def _load_preferences(self) -> None:
        try:
            result = await asyncio.to_thread(self._store.load)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Studio TTS preferences could not be loaded")
            result = StudioTTSLoadResult(
                StudioTTSPreferencesSnapshot(),
                StudioTTSLoadState.CORRUPT,
                ("speech_studio",),
            )
        if self.is_mounted:
            self._apply_load_result(result)

    def _apply_load_result(self, result: StudioTTSLoadResult) -> None:
        if not self.is_mounted or not self.query("#studio-tts-status"):
            return
        self._load_result = result
        self._saved_snapshot = result.snapshot
        self._corrupt = result.state is StudioTTSLoadState.CORRUPT
        self._apply_snapshot(result.snapshot)
        status = self.query_one("#studio-tts-status", Static)
        if self._corrupt:
            status.update(
                "Studio preferences are corrupt. Use Reset to Global to reset "
                "only Studio preferences."
            )
        elif result.issues:
            status.update(
                "Studio preferences recovered with safe field errors: "
                + ", ".join(result.issues)
            )
        elif result.state is StudioTTSLoadState.MIGRATED:
            status.update("Studio preferences migrated and saved.")
        else:
            status.update("Inherited values remain linked to Global defaults.")
        if result.issues and not self._corrupt:
            self._show_load_issues(result.issues)
        self._show_contextual_constraint_errors(result.snapshot)
        if self._adopted_preset is not None:
            adopted_preset = self._adopted_preset
            self._adopted_preset = None
            self._apply_adopted_preset(adopted_preset)
        self._load_applied = True

    def _show_load_issues(self, issues: tuple[str, ...]) -> None:
        """Map bounded storage issue paths to their owning visible fields."""

        for issue in issues:
            if issue.endswith(".exaggeration"):
                field = "exaggeration"
            elif issue.endswith(".cfg_weight"):
                field = "cfg-weight"
            elif ".model_id" in issue:
                field = "model-id"
            elif ".model_mode" in issue:
                field = "model-mode"
            elif ".voice_id" in issue:
                field = "voice-id"
            elif ".voice_mode" in issue:
                field = "voice-mode"
            elif ".response_format" in issue:
                field = "format"
            elif ".speed" in issue:
                field = "speed"
            elif issue.endswith(".auto_play"):
                field = "auto-play"
            else:
                field = "provider"
            self._set_error(field, "Ignored unsupported saved Studio value")

    def _show_contextual_constraint_errors(
        self,
        snapshot: StudioTTSPreferencesSnapshot,
    ) -> None:
        """Expose sparse values made invalid by the current global provider."""

        selection = snapshot.selection
        provider_id = selection.provider_id or self._global_preferences.provider_id
        if provider_id != "audio_cpp":
            return
        if selection.response_format not in (None, "wav"):
            self._set_error(
                "format",
                "Saved Studio format is incompatible with audio.cpp",
            )
        if selection.speed not in (None, 1.0):
            self._set_error(
                "speed",
                "Saved Studio speed is incompatible with audio.cpp",
            )

    def _apply_snapshot(self, snapshot: StudioTTSPreferencesSnapshot) -> None:
        self._forced_global_provider_draft = None
        selection = snapshot.selection
        provider_value = selection.provider_id or _INHERIT
        self._applying_controls = True
        try:
            self.query_one("#studio-tts-provider", Select).value = provider_value
            self.query_one("#studio-tts-auto-play", Switch).value = snapshot.auto_play
            self._active_provider_value = provider_value
            self._populate_provider_controls(provider_value, snapshot)
            self._sync_auto_play_copy()
        finally:
            self._applying_controls = False
        self._adoption_pending = False
        self.query_one("#studio-tts-adoption-status", Static).add_class("hidden")
        self._clear_errors()
        self._sync_dirty_state()

    def _selection_for_provider(
        self,
        provider_value: str,
        snapshot: StudioTTSPreferencesSnapshot,
    ) -> StudioTTSSelectionOverrides:
        selection = snapshot.selection
        effective_provider = (
            self._global_preferences.provider_id
            if provider_value == _INHERIT
            else provider_value
        )
        selection_owner = selection.provider_id or self._global_preferences.provider_id
        if effective_provider == selection_owner:
            return selection
        return StudioTTSSelectionOverrides(
            provider_id=None if provider_value == _INHERIT else provider_value
        )

    def _populate_provider_controls(
        self,
        provider_value: str,
        snapshot: StudioTTSPreferencesSnapshot,
    ) -> None:
        effective_provider = (
            self._global_preferences.provider_id
            if provider_value == _INHERIT
            else provider_value
        )
        selection = self._selection_for_provider(provider_value, snapshot)
        model_mode = selection.model_mode or _INHERIT
        voice_mode = selection.voice_mode or _INHERIT
        self.query_one("#studio-tts-model-mode", Select).value = model_mode
        self.query_one("#studio-tts-model-id", Input).value = selection.model_id or ""
        self.query_one("#studio-tts-voice-mode", Select).value = voice_mode
        self.query_one("#studio-tts-voice-id", Input).value = selection.voice_id or ""

        format_select = self.query_one("#studio-tts-format", Select)
        speed_input = self.query_one("#studio-tts-speed", Input)
        is_audio_cpp = effective_provider == "audio_cpp"
        if is_audio_cpp:
            format_select.value = "wav"
            speed_input.value = "1.0"
        else:
            format_select.value = selection.response_format or _INHERIT
            speed_input.value = "" if selection.speed is None else str(selection.speed)
        format_select.disabled = is_audio_cpp
        speed_input.disabled = is_audio_cpp

        model_input = self.query_one("#studio-tts-model-id", Input)
        voice_input = self.query_one("#studio-tts-voice-id", Input)
        model_input.disabled = model_mode != "exact"
        voice_input.disabled = voice_mode != "exact"

        options = snapshot.provider_options.get(effective_provider, {})
        self.query_one("#chatterbox-exaggeration-input", Input).value = (
            "" if "exaggeration" not in options else str(options["exaggeration"])
        )
        self.query_one("#chatterbox-cfg-weight-input", Input).value = (
            "" if "cfg_weight" not in options else str(options["cfg_weight"])
        )
        self.query_one("#studio-tts-chatterbox-options").set_class(
            effective_provider != "chatterbox", "hidden"
        )
        self._sync_source_copy(effective_provider)

    def _apply_adopted_preset(self, preset: TTSPlaygroundSelectionPreset) -> None:
        if preset.provider_id not in BUILT_IN_TTS_PROVIDER_IDS:
            self._set_error("provider", "Unsupported profile provider")
            return
        allowed = STUDIO_TTS_PROVIDER_OPTION_KEYS[preset.provider_id]
        if any(key not in allowed for key in preset.options):
            self._set_error("provider", "Profile contains unsupported Studio tuning")
            return
        self._applying_controls = True
        try:
            self.query_one("#studio-tts-provider", Select).value = preset.provider_id
            self._active_provider_value = preset.provider_id
            self._populate_provider_controls(
                preset.provider_id,
                StudioTTSPreferencesSnapshot(
                    revision=self._saved_snapshot.revision,
                    selection=StudioTTSSelectionOverrides(
                        provider_id=preset.provider_id,
                        model_mode="exact",
                        model_id=preset.model_id,
                        voice_mode=(
                            "server_default" if preset.voice_id is None else "exact"
                        ),
                        voice_id=preset.voice_id,
                        response_format=(
                            None
                            if preset.provider_id == "audio_cpp"
                            else preset.response_format
                        ),
                        speed=None
                        if preset.provider_id == "audio_cpp"
                        else preset.speed,
                    ),
                    provider_options=(
                        {preset.provider_id: dict(preset.options)}
                        if preset.options
                        else self._saved_snapshot.provider_options
                    ),
                ),
            )
        finally:
            self._applying_controls = False
        self._adoption_pending = True
        banner = self.query_one("#studio-tts-adoption-status", Static)
        banner.update(
            "Adopted profile preview — review these Studio-only values, then Save "
            "Studio Preferences to persist them."
        )
        banner.remove_class("hidden")
        self._sync_dirty_state()

    def _effective_provider(self) -> str:
        value = self.query_one("#studio-tts-provider", Select).value
        return self._global_preferences.provider_id if value == _INHERIT else str(value)

    def _fallback_axis(self, provider_id: str, axis: str) -> str:
        if provider_id == self._global_preferences.provider_id:
            values = {
                "model": (
                    self._global_preferences.model_id
                    if self._global_preferences.model_mode == "exact"
                    else "First available"
                ),
                "voice": (
                    self._global_preferences.voice_id
                    if self._global_preferences.voice_mode == "exact"
                    else "Server default"
                ),
                "format": self._global_preferences.response_format.upper(),
                "speed": str(self._global_preferences.speed),
            }
            return f"Inherited — Global defaults: {values[axis]}"
        values = {
            "model": (
                "First available"
                if provider_id == "audio_cpp"
                else LEGACY_DEFAULT_MODELS.get(provider_id, "Provider default")
            ),
            "voice": (
                "Server default"
                if provider_id == "audio_cpp"
                else LEGACY_DEFAULT_VOICES.get(provider_id, "Provider default")
            ),
            "format": "WAV" if provider_id not in {"openai", "elevenlabs"} else "MP3",
            "speed": "1.0",
        }
        return f"Inherited — Provider fallback: {values[axis]}"

    def _sync_source_copy(self, provider_id: str | None = None) -> None:
        if not self.is_mounted:
            return
        provider_id = provider_id or self._effective_provider()
        provider_value = self.query_one("#studio-tts-provider", Select).value
        if provider_value == _INHERIT:
            provider_copy = (
                "Inherited — Global defaults: "
                f"{_PROVIDER_LABELS[self._global_preferences.provider_id]}"
            )
        else:
            provider_copy = "Studio override"
        self.query_one("#studio-tts-provider-source", Static).update(provider_copy)

        model_mode = self.query_one("#studio-tts-model-mode", Select).value
        voice_mode = self.query_one("#studio-tts-voice-mode", Select).value
        fmt = self.query_one("#studio-tts-format", Select).value
        speed = self.query_one("#studio-tts-speed", Input).value.strip()
        self.query_one("#studio-tts-model-source", Static).update(
            self._fallback_axis(provider_id, "model")
            if model_mode == _INHERIT
            else "Studio override"
        )
        self.query_one("#studio-tts-model-id-source", Static).update(
            "Used only with Exact" if model_mode != "exact" else "Exact Studio ID"
        )
        self.query_one("#studio-tts-voice-source", Static).update(
            self._fallback_axis(provider_id, "voice")
            if voice_mode == _INHERIT
            else "Studio override"
        )
        self.query_one("#studio-tts-voice-id-source", Static).update(
            "Used only with Exact" if voice_mode != "exact" else "Exact Studio ID"
        )
        self.query_one("#studio-tts-format-source", Static).update(
            "Fixed by audio.cpp: WAV"
            if provider_id == "audio_cpp"
            else (
                self._fallback_axis(provider_id, "format")
                if fmt == _INHERIT
                else "Studio override"
            )
        )
        self.query_one("#studio-tts-speed-source", Static).update(
            "Fixed by audio.cpp: 1.0"
            if provider_id == "audio_cpp"
            else (
                self._fallback_axis(provider_id, "speed")
                if not speed
                else "Studio override"
            )
        )
        for key, source_id in (
            ("exaggeration", "#studio-tts-exaggeration-source"),
            ("cfg_weight", "#studio-tts-cfg-weight-source"),
        ):
            input_id = (
                "#chatterbox-exaggeration-input"
                if key == "exaggeration"
                else "#chatterbox-cfg-weight-input"
            )
            value = self.query_one(input_id, Input).value.strip()
            self.query_one(source_id, Static).update(
                "Inherited — Provider fallback" if not value else "Studio override"
            )

    def _clear_errors(self) -> None:
        for error in self.query(".studio-tts-field-error").results(Static):
            error.update("")
            error.add_class("hidden")

    def _set_error(self, field: str, copy: str) -> None:
        try:
            error = self.query_one(f"#studio-tts-{field}-error", Static)
        except QueryError:
            return
        error.update(copy)
        error.remove_class("hidden")

    def _focus_first_error(self) -> None:
        """Move keyboard focus to the first visible invalid Studio field."""

        for field, control_id in (
            ("provider", "studio-tts-provider"),
            ("model-mode", "studio-tts-model-mode"),
            ("model-id", "studio-tts-model-id"),
            ("voice-mode", "studio-tts-voice-mode"),
            ("voice-id", "studio-tts-voice-id"),
            ("format", "studio-tts-format"),
            ("speed", "studio-tts-speed"),
            ("auto-play", "studio-tts-auto-play"),
            ("exaggeration", "chatterbox-exaggeration-input"),
            ("cfg-weight", "chatterbox-cfg-weight-input"),
        ):
            try:
                error = self.query_one(f"#studio-tts-{field}-error", Static)
            except QueryError:
                continue
            if error.has_class("hidden"):
                continue
            control = self.query_one(f"#{control_id}")
            control.screen.set_focus(control)
            # In the stacked narrow layout the validation copy is the row
            # immediately after the focused control. Include that copy in
            # the viewport instead of stopping with it one line below. Wait
            # for removing ``hidden`` to participate in layout first.
            self.call_after_refresh(self._scroll_error_into_view, error)
            return

    def _scroll_error_into_view(self, error: Static) -> None:
        """Reveal one laid-out validation row inside the Studio scroller."""

        if not self.is_mounted or not error.is_mounted:
            return
        self.query_one("#speech-settings-groups", VerticalScroll).scroll_to_widget(
            error,
            animate=False,
            force=True,
            immediate=True,
        )

    def _collect_candidate(
        self,
        *,
        show_errors: bool,
    ) -> StudioTTSPreferencesSnapshot | None:
        if show_errors:
            self._clear_errors()
        provider_value = self.query_one("#studio-tts-provider", Select).value
        if provider_value == _INHERIT:
            provider_id = None
            effective_provider = self._global_preferences.provider_id
        elif (
            isinstance(provider_value, str)
            and provider_value in BUILT_IN_TTS_PROVIDER_IDS
        ):
            provider_id = provider_value
            effective_provider = provider_value
        else:
            if show_errors:
                self._set_error("provider", "Choose a supported provider or Inherit")
            return None

        model_mode_value = self.query_one("#studio-tts-model-mode", Select).value
        model_mode = None if model_mode_value == _INHERIT else model_mode_value
        model_id = self.query_one("#studio-tts-model-id", Input).value
        if model_mode == "exact" and not model_id.strip():
            if show_errors:
                self._set_error("model-id", "Exact model ID is required")
            return None
        if model_mode != "exact":
            model_id = None

        voice_mode_value = self.query_one("#studio-tts-voice-mode", Select).value
        voice_mode = None if voice_mode_value == _INHERIT else voice_mode_value
        voice_id = self.query_one("#studio-tts-voice-id", Input).value
        if voice_mode == "exact" and not voice_id.strip():
            if show_errors:
                self._set_error("voice-id", "Exact voice ID is required")
            return None
        if (
            voice_mode == "exact"
            and voice_id.startswith("blend:")
            and effective_provider != "kokoro"
        ):
            if show_errors:
                self._set_error(
                    "voice-id",
                    "Voice Blends are available only with Kokoro",
                )
            return None
        if voice_mode != "exact":
            voice_id = None

        if effective_provider == "audio_cpp":
            response_format = None
            speed = None
        else:
            format_value = self.query_one("#studio-tts-format", Select).value
            response_format = None if format_value == _INHERIT else format_value
            speed_text = self.query_one("#studio-tts-speed", Input).value.strip()
            try:
                speed = None if not speed_text else float(speed_text)
            except ValueError:
                if show_errors:
                    self._set_error("speed", "Speed must be a number from 0.25 to 4.0")
                return None

        options = {
            owner: dict(values)
            for owner, values in self._saved_snapshot.provider_options.items()
        }
        if effective_provider == "chatterbox":
            chatterbox: dict[str, float] = {}
            for key, control_id, error_field in (
                (
                    "exaggeration",
                    "#chatterbox-exaggeration-input",
                    "exaggeration",
                ),
                ("cfg_weight", "#chatterbox-cfg-weight-input", "cfg-weight"),
            ):
                raw = self.query_one(control_id, Input).value.strip()
                if not raw:
                    continue
                try:
                    number = float(raw)
                except ValueError:
                    if show_errors:
                        self._set_error(error_field, "Enter a number from 0.0 to 1.0")
                    return None
                chatterbox[key] = number
            if chatterbox:
                options["chatterbox"] = chatterbox
            else:
                options.pop("chatterbox", None)

        try:
            return StudioTTSPreferencesSnapshot(
                revision=self._saved_snapshot.revision,
                auto_play=self.query_one("#studio-tts-auto-play", Switch).value,
                selection=StudioTTSSelectionOverrides(
                    provider_id=provider_id,
                    model_mode=model_mode,
                    model_id=model_id,
                    voice_mode=voice_mode,
                    voice_id=voice_id,
                    response_format=response_format,
                    speed=speed,
                ),
                provider_options=options,
            )
        except (TypeError, ValueError) as error:
            if show_errors:
                message = str(error).casefold()
                field = "provider"
                for fragment, candidate in (
                    ("speed", "speed"),
                    ("model", "model-id"),
                    ("voice", "voice-id"),
                    ("format", "format"),
                    ("exaggeration", "exaggeration"),
                    ("cfg_weight", "cfg-weight"),
                ):
                    if fragment in message:
                        field = candidate
                        break
                self._set_error(field, "This Studio value is not supported")
            return None

    def _sync_dirty_state(self) -> None:
        if self._applying_controls or not self.is_mounted:
            return
        self._dirty = self.is_dirty
        status = self.query_one("#studio-tts-status", Static)
        if not self._corrupt and not self._busy:
            status.update(
                "Unsaved Studio changes"
                if self._dirty
                else "Studio preferences are saved"
            )

    @on(Input.Changed)
    def _input_changed(self, event: Input.Changed) -> None:
        if self._applying_controls:
            return
        if event.input.id in {
            "studio-tts-model-id",
            "studio-tts-voice-id",
            "studio-tts-speed",
            "chatterbox-exaggeration-input",
            "chatterbox-cfg-weight-input",
        }:
            self._sync_source_copy()
            self._sync_dirty_state()

    @on(Switch.Changed, "#studio-tts-auto-play")
    def _auto_play_changed(self, event: Switch.Changed) -> None:
        """Update consequence copy and dirty state for the Studio-only toggle."""

        if self._applying_controls:
            return
        self._sync_auto_play_copy()
        self._sync_dirty_state()

    def _sync_auto_play_copy(self) -> None:
        """State the toggle consequence without implying a global default."""

        enabled = self.query_one("#studio-tts-auto-play", Switch).value
        copy = (
            "On — auto-plays completed Studio results (Studio only)"
            if enabled
            else "Off — waits for Play after generation (Studio only)"
        )
        self.query_one("#studio-tts-auto-play-state", Static).update(copy)

    @on(Select.Changed)
    def _select_changed(self, event: Select.Changed) -> None:
        if self._applying_controls:
            return
        # Textual delivers programmatic Select changes on a later message-loop
        # turn.  A complete snapshot can therefore write several values before
        # the first queued event arrives; only the event that still describes
        # the mounted control is current.
        if event.value != event.select.value:
            return
        if event.select.id == "studio-tts-provider":
            target = event.value
            if not isinstance(target, str):
                return
            if target == self._active_provider_value:
                return
            was_dirty = self.is_dirty
            if was_dirty and target != self._active_provider_value:
                self._applying_controls = True
                try:
                    event.select.value = self._active_provider_value
                finally:
                    self._applying_controls = False
                self.run_worker(
                    self._confirm_provider_switch(target),
                    group="studio-tts-provider-switch",
                    exclusive=True,
                    exit_on_error=False,
                )
                return
            self._apply_provider(target)
            return
        if event.select.id == "studio-tts-model-mode":
            self.query_one("#studio-tts-model-id", Input).disabled = (
                event.value != "exact"
            )
        elif event.select.id == "studio-tts-voice-mode":
            self.query_one("#studio-tts-voice-id", Input).disabled = (
                event.value != "exact"
            )
        self._sync_source_copy()
        self._sync_dirty_state()

    def _apply_provider(self, provider_value: str) -> None:
        """Switch the visible provider and discard hidden unsaved values."""

        if (
            provider_value != _INHERIT
            and provider_value not in BUILT_IN_TTS_PROVIDER_IDS
        ):
            self._set_error("provider", "Choose a supported provider or Inherit")
            return
        self._forced_global_provider_draft = None
        self._applying_controls = True
        try:
            self.query_one("#studio-tts-provider", Select).value = provider_value
            self._active_provider_value = provider_value
            self._populate_provider_controls(provider_value, self._saved_snapshot)
        finally:
            self._applying_controls = False
        self._clear_errors()
        self._sync_dirty_state()

    async def _confirm_provider_switch(self, target: str) -> None:
        if await self.confirm_leave():
            self._apply_provider(target)

    def _set_status(
        self,
        copy: str,
        *,
        severity: str | None = None,
    ) -> None:
        """Update visible status and optionally use the app announcement channel."""

        self.query_one("#studio-tts-status", Static).update(copy)
        if severity is not None:
            self.app.notify(copy, severity=severity)

    def _set_busy(self, busy: bool, copy: str = "") -> None:
        self._busy = busy
        if not self.is_mounted:
            return
        for button_id in (
            "studio-tts-save-btn",
            "studio-tts-revert-btn",
            "studio-tts-reset-btn",
            "studio-tts-open-global-btn",
        ):
            self.query_one(f"#{button_id}", Button).disabled = busy
        for control in self.query(".speech-setting-control"):
            if isinstance(control, (Input, Select, Switch)):
                control.disabled = busy
        if not busy:
            provider_id = self._effective_provider()
            self.query_one("#studio-tts-model-id", Input).disabled = (
                self.query_one("#studio-tts-model-mode", Select).value != "exact"
            )
            self.query_one("#studio-tts-voice-id", Input).disabled = (
                self.query_one("#studio-tts-voice-mode", Select).value != "exact"
            )
            is_audio_cpp = provider_id == "audio_cpp"
            self.query_one("#studio-tts-format", Select).disabled = is_audio_cpp
            self.query_one("#studio-tts-speed", Input).disabled = is_audio_cpp
        if copy:
            self._set_status(copy, severity="information")

    async def save_preferences(self) -> bool:
        """Validate and atomically persist only the Studio snapshot."""

        if self._busy:
            return False
        if self._corrupt:
            self._set_status(
                "Reset only Studio preferences before saving this corrupt record.",
                severity="error",
            )
            return False
        candidate = self._collect_candidate(show_errors=True)
        if candidate is None:
            self._set_status(
                "Fix the highlighted Studio fields before saving.",
                severity="error",
            )
            self._focus_first_error()
            return False
        focus_id = self._focused_id()
        self._set_busy(True, "Saving Studio preferences…")
        try:
            result = await asyncio.to_thread(self._store.save, candidate)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Studio TTS preference save failed")
            self._set_status(
                "Studio preferences were not saved.",
                severity="error",
            )
            return False
        finally:
            self._set_busy(False)
            self._restore_focus(focus_id)

        if result.snapshot is None or result.status in {
            StudioTTSWriteStatus.CONFLICT,
            StudioTTSWriteStatus.FAILED,
        }:
            copy = (
                "Studio preferences changed elsewhere; Revert and try again."
                if result.status is StudioTTSWriteStatus.CONFLICT
                else "Studio preferences were not saved."
            )
            self._set_status(copy, severity="error")
            return False

        self._saved_snapshot = result.snapshot
        self._adoption_pending = False
        self.query_one("#studio-tts-adoption-status", Static).add_class("hidden")
        self._apply_snapshot(result.snapshot)
        self._set_status(
            (
                "Studio preferences saved."
                if result.status is not StudioTTSWriteStatus.SAVED_CACHE_RELOAD_FAILED
                else "Studio preferences saved; reopen the pane to refresh cached values."
            ),
            severity=(
                "information"
                if result.status is not StudioTTSWriteStatus.SAVED_CACHE_RELOAD_FAILED
                else "warning"
            ),
        )
        self.post_message(StudioPreferencesSaved(result.snapshot))
        return True

    def _restore_saved_preferences(self) -> None:
        """Discard the visible draft using the already loaded snapshot."""

        self._apply_snapshot(self._saved_snapshot)

    async def revert_preferences(self) -> bool:
        """Reload and restore the latest successfully saved Studio snapshot."""

        if self._busy:
            return False
        focus_id = self._focused_id()
        failed = False
        result: StudioTTSLoadResult | None = None
        self._set_busy(True, "Reloading saved Studio preferences…")
        try:
            result = await asyncio.to_thread(self._store.load, migrate=False)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Studio TTS preference reload failed")
            failed = True
        finally:
            self._set_busy(False)

        if failed or result is None:
            self._set_status(
                "Saved Studio preferences could not be reloaded.",
                severity="error",
            )
            self._restore_focus(focus_id)
            return False
        self._apply_load_result(result)
        if result.state is StudioTTSLoadState.CORRUPT:
            self._restore_focus(focus_id)
            return False
        self._set_status(
            "Reverted to saved Studio preferences.",
            severity="information",
        )
        self.post_message(StudioPreferencesSaved(result.snapshot))
        self._restore_focus(focus_id)
        return True

    async def reset_to_global(self) -> bool:
        """Delete all Studio overrides without touching any other owner."""

        if self._busy:
            return False
        focus_id = self._focused_id()
        self._set_busy(True, "Resetting only Studio preferences…")
        try:
            result = await asyncio.to_thread(
                self._store.reset_to_global,
                self._saved_snapshot,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Studio TTS preference reset failed")
            self._set_status(
                "Studio preferences were not reset.",
                severity="error",
            )
            return False
        finally:
            self._set_busy(False)
            self._restore_focus(focus_id)
        if result.snapshot is None or result.status in {
            StudioTTSWriteStatus.CONFLICT,
            StudioTTSWriteStatus.FAILED,
        }:
            self._set_status(
                "Studio preferences were not reset.",
                severity="error",
            )
            return False
        self._corrupt = False
        self._saved_snapshot = result.snapshot
        self._apply_snapshot(result.snapshot)
        self._set_status(
            "Studio overrides removed; values now inherit Global defaults.",
            severity="information",
        )
        self.post_message(StudioPreferencesSaved(result.snapshot, reset_to_global=True))
        return True

    async def _ask_leave_choice(self) -> LeaveChoice:
        return await self.app.push_screen_wait(StudioTTSLeaveModal())

    def _focused_id(self) -> str | None:
        focused = self.app.focused
        return focused.id if focused is not None else None

    def _restore_focus(self, control_id: str | None) -> None:
        if not control_id or not self.is_mounted:
            return
        try:
            control = self.query_one(f"#{control_id}")
            control.screen.set_focus(control)
        except QueryError:
            return

    async def confirm_leave(self) -> bool:
        """Resolve one dirty-draft prompt before this pane can be removed."""

        if not self.is_dirty:
            return True
        focus_id = self._focused_id()
        choice = await self._ask_leave_choice()
        if choice == "discard":
            self._restore_saved_preferences()
            return True
        if choice == "save":
            saved = await self.save_preferences()
            if not saved:
                self._restore_focus(focus_id)
            return saved
        self._restore_focus(focus_id)
        return False

    async def _open_global_settings(self) -> None:
        provider_id = self._effective_provider()
        if not await self.confirm_leave():
            return
        self.app.post_message(
            NavigateToScreen(
                "settings",
                {
                    "category": "speech-tts",
                    **speech_tts_navigation_context(
                        SpeechTTSNavigationTarget(
                            provider_id,
                            SpeechTTSNavigationIntent.CONFIGURE,
                        )
                    ),
                },
            )
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "studio-tts-save-btn":
            event.stop()
            self.run_worker(
                self.save_preferences(),
                group="studio-tts-save",
                exclusive=True,
                exit_on_error=False,
            )
        elif button_id == "studio-tts-revert-btn":
            event.stop()
            self.run_worker(
                self.revert_preferences(),
                group="studio-tts-revert",
                exclusive=True,
                exit_on_error=False,
            )
        elif button_id == "studio-tts-reset-btn":
            event.stop()
            self.run_worker(
                self.reset_to_global(),
                group="studio-tts-reset",
                exclusive=True,
                exit_on_error=False,
            )
        elif button_id == "studio-tts-open-global-btn":
            event.stop()
            self.run_worker(
                self._open_global_settings(),
                group="studio-tts-open-global",
                exclusive=True,
                exit_on_error=False,
            )
        elif button_id in {"voice-profiles", "voice-blends"}:
            event.stop()
            self.post_message(SpeechDestinationRequested(button_id))
        # Voice blend ids are handled once by the inherited legacy operation
        # handler.  Textual dispatches that handler separately through the MRO;
        # calling it here as well would open two file pickers or dialogs.


__all__ = [
    "SpeechDestinationBackRequested",
    "SpeechDestinationRequested",
    "SpeechSettingsPane",
    "StudioPreferencesSaved",
    "StudioTTSLeaveModal",
    "VOICE_BLEND_ACTIONS",
    "VOICE_DESTINATION_ACTIONS",
    "VoiceBlendsPane",
]
