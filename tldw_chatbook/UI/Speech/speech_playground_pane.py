"""TTS Playground in the Console grammar.

Replaces the legacy playground's stacked form. That form measured **93 rows
in a 34-row viewport** -- every label/input pair cost 4 rows and two
explanatory notices cost 5 each -- which put `Generate Speech` at y=60,
21 rows below the fold on arrival.

The Console idiom this follows, read off the running Console screen rather
than invented here:

- one row per thing, never a box per control
- visible commands as a text action strip (``CommandStrip``), not chunky
  buttons stacked down the page
- settings as ``Label: value`` chips on one line, the way Console states
  provider/model/character/RAG above its composer
- a single bordered input, like Console's composer
- recovery copy as one line, not a five-row block

Nothing here wraps a legacy STTS widget: the point is the grammar, and the
old widgets carry the old rhythm with them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from textual import on
from textual.binding import Binding
from textual.app import ComposeResult
from textual.css.query import NoMatches
from textual.containers import Horizontal, Vertical
from textual.message import Message
from tldw_chatbook.TTS import (
    AudioCppRuntimeObservation,
    CanonicalTTSCloneReference,
    STTSGeneratedAudio,
    STTSPlaygroundCloneSnapshot,
    TTSPlaygroundSelectionPreset,
    TTSPreferencesSnapshot,
)
from tldw_chatbook.TTS.provider_ids import BUILT_IN_TTS_PROVIDER_IDS
from tldw_chatbook.TTS.studio_preferences import StudioTTSPreferencesSnapshot
from tldw_chatbook.UI.Lab_Modules.lab_speech_status import (
    speech_local_dependency_availability,
)

from textual.widgets import (
    Button,
    Collapsible,
    ProgressBar,
    RichLog,
    Input,
    Select,
    Static,
    Switch,
    TextArea,
)

from tldw_chatbook.UI.stts_playground_catalog import UNAVAILABLE_SELECT_VALUE

from ..Workbench.workbench_state import WorkbenchAction
from .speech_action_strip import SpeechActionStrip
from .speech_axis_row import AXIS_EMPTY_PROMPTS, SpeechAxisRow
from .audio_cpp_runtime_card import (
    AudioCppCloneDraftState,
    AudioCppRuntimeAction,
    AudioCppRuntimeCard,
    AudioCppRuntimeCardObservation,
    AudioCppRuntimeOperation,
    AudioCppSampleState,
    project_audio_cpp_runtime_card,
    project_audio_cpp_unknown_action,
)
from .speech_catalog_mixin import SpeechCatalogMixin
from .speech_clone_setup import SpeechCloneSetup
from .speech_playback_mixin import EXAMPLE_TEXTS, SpeechPlaybackMixin
from .speech_playground_model import AXIS_CONTROLS
from .speech_profile_mixin import (
    AdoptStudioPreferencesRequested,
    SpeechProfileMixin,
)
from .speech_synthesis_mixin import SpeechSynthesisMixin
from .speech_param_group import SpeechParamGroup
from .speech_runtime_status import (
    SpeechLocalDependencyAvailability,
    SpeechTTSRuntimeStatusStore,
    newest_speech_tts_status,
    project_speech_tts_status,
    speech_tts_runtime_status_from_catalog,
    speech_tts_navigation_context,
)
from .speech_settings_contracts import (
    SpeechTTSConfigurationState,
    SpeechTTSDiagnosticCategory,
    SpeechTTSNavigationIntent,
    SpeechTTSNavigationTarget,
    SpeechTTSRuntimeState,
    SpeechTTSRuntimeStatus,
    SpeechTTSStatusFreshness,
    speech_tts_model_scope,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.TTS.profile_reference_audio import canonicalize_reference_wav
from tldw_chatbook.TTS.profile_reference_types import (
    MAX_REFERENCE_TEXT_CHARACTERS,
    validate_reference_text,
)
from tldw_chatbook.Third_Party.textual_fspicker import Filters
from tldw_chatbook.Utils.input_validation import validate_text_input
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen

_AUDIO_CPP_RUNTIME_POLL_SECONDS = 5.0


def _validate_clone_transcript_input(value: str) -> str:
    """Apply shared boundary checks before clone-specific normalization."""

    if not validate_text_input(
        value,
        max_length=MAX_REFERENCE_TEXT_CHARACTERS,
        allow_html=True,
    ):
        raise ValueError("reference_text")
    return validate_reference_text(value)

if TYPE_CHECKING:
    pass

#: Below this pane width the split and action rows stack. Two panes need
#: roughly 30 cells each, while the fixed six-action Playground strip needs
#: 83 cells including its spacing. A small margin above that measured width
#: keeps the final action inside the pane. Mirrors
#: `PERSONAS_COMPACT_WORKBENCH_MAX_WIDTH`'s approach -- a measured threshold,
#: toggled from `on_resize`, because Textual has no media queries.
SPEECH_SPLIT_MIN_WIDTH = 86


class OpenStudioPreferencesRequested(Message):
    """Ask the owning Speech window to show its Studio-only editor."""


class OpenVoiceProfilesRequested(Message):
    """Ask the owning Speech window to open its existing Voice Profiles view."""


#: Providers that synthesize from a reference clip, so the clip picker is
#: only mounted for them -- legacy mounted it always and hid it, which left
#: it focusable while inert.
REFERENCE_AUDIO_PROVIDERS = ("chatterbox",)

#: Providers with their own voice-sample upload flow.
VOICE_UPLOAD_PROVIDERS = ("higgs",)

#: The playground's visible commands, in the order Console orders its own:
#: the thing you came to do first, then what you do to the result.
#: The text actions. Ids are the legacy control ids, and `SpeechActionStrip`
#: mounts them verbatim: the handler matches `event.button.id` against these
#: exact strings, so a strip that renames its buttons never fires.
PLAYGROUND_ACTIONS: tuple[WorkbenchAction, ...] = (
    WorkbenchAction(
        id="tts-generate-btn",
        label="Generate",
        primary=True,
    ),
    WorkbenchAction(
        id="tts-random-text-btn",
        label="Sample text",
        tooltip="Insert sample text (Ctrl+R)",
    ),
    WorkbenchAction(
        id="tts-clear-text-btn",
        label="Clear text",
        tooltip="Clear the text (Ctrl+L)",
    ),
    WorkbenchAction(
        id="tts-test-connection-btn",
        label="Test",
        tooltip="Test the selected provider connection",
    ),
    WorkbenchAction(
        id="tts-refresh-catalog-btn",
        label="Refresh",
        tooltip="Re-read available models and voices",
    ),
    WorkbenchAction(
        id="tts-open-studio-preferences-btn",
        label="Studio preferences",
        tooltip="Configure saved overrides used only by the Speech Studio",
    ),
)

#: Playback actions, kept beside the result rather than mixed into the strip
#: that acts on the text.
PLAYER_ACTIONS: tuple[WorkbenchAction, ...] = (
    WorkbenchAction(
        id="audio-play-btn",
        label="Play",
        tooltip="Generate audio before playing the current result",
        disabled=True,
        primary=True,
    ),
    WorkbenchAction(
        id="pause-audio-btn",
        label="Pause",
        tooltip="Generate audio before pausing playback",
        disabled=True,
    ),
    WorkbenchAction(
        id="stop-audio-btn",
        label="Stop",
        tooltip="Generate audio before stopping playback",
        disabled=True,
    ),
    WorkbenchAction(
        id="audio-generate-again-btn",
        label="Generate again",
        tooltip="Generate audio before generating another result",
        disabled=True,
    ),
    WorkbenchAction(
        id="audio-export-btn",
        label="Save as…",
        tooltip="Generate audio before saving a current result",
        disabled=True,
    ),
)


class SpeechChip(Static):
    """One `Label: value` cell in the settings line.

    A chip, not a form field. The legacy screen spent four rows on each of
    these -- a label row, an input row, and two rows of padding -- for
    values that are one short token each.
    """

    def __init__(self, label: str, value: str, **kwargs: Any) -> None:
        """Create a settings chip.

        Args:
            label: The setting's name, e.g. ``"Voice"``.
            value: Its current value, rendered plainly beside the label.
            kwargs: Forwarded to ``Static``.
        """
        classes = kwargs.pop("classes", "")
        super().__init__(
            f"{label}: {value}",
            classes=f"speech-chip {classes}".strip(),
            markup=False,
            **kwargs,
        )


class SpeechPlaygroundPane(
    SpeechSynthesisMixin,
    SpeechCatalogMixin,
    SpeechPlaybackMixin,
    SpeechProfileMixin,
    Vertical,
):
    """The TTS Playground body: title, actions, input, settings, status.

    Synthesis comes from `SpeechSynthesisMixin`, shared with the legacy
    widget rather than reimplemented: the generate path is 322 lines of
    provider resolution and request building, and a second copy would drift
    from the first. The mixin queries its controls by id, which is why this
    rebuild kept the legacy ids.
    """

    _tts_service: Any

    #: Pane-local accelerators that do not shadow ADR-031 global or terminal
    #: keys. The containing STTS screen owns its plain single-letter actions.
    BINDINGS = [
        Binding("ctrl+g", "generate_tts", "Generate Speech"),
        Binding("ctrl+l", "clear_text", "Clear Text"),
    ]

    def __init__(
        self,
        *,
        provider: str = "audio_cpp",
        profile_preset: Any = None,
        profile_context_token: UUID | None = None,
        axis_values: dict[str, str] | None = None,
        axis_defaults: dict[str, str] | None = None,
        capability_line: str = "",
        studio_preferences: StudioTTSPreferencesSnapshot | None = None,
        global_preferences: TTSPreferencesSnapshot | None = None,
        navigation_target: SpeechTTSNavigationTarget | None = None,
        provider_configuration_states: Mapping[str, SpeechTTSConfigurationState]
        | None = None,
        runtime_status_store: SpeechTTSRuntimeStatusStore | None = None,
        local_dependencies: SpeechLocalDependencyAvailability | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the pane.

        Args:
            provider: Selected provider; decides the parameter group.
            axis_values: Effective value per comparison axis.
            axis_defaults: Persisted default per axis, for override marking.
            capability_line: One-line local-speech status.
            local_dependencies: Shared local-capability snapshot. When omitted,
                a fresh non-importing module-presence probe is used.
            kwargs: Forwarded to ``Vertical``.
        """
        classes = kwargs.pop("classes", "")
        super().__init__(classes=f"speech-pane {classes}".strip(), **kwargs)
        #: None until first measured, so the first sync always applies.
        self._stacked: bool | None = None
        self._provider_regions_replacing = False
        #: Effective axis values for this session, and the persisted
        #: defaults they are compared against. The pane never writes the
        #: defaults -- overrides are session-scoped by design.
        self.axis_values: dict[str, str] = dict(axis_values or {})
        self.axis_defaults: dict[str, str] = dict(axis_defaults or {})
        #: Selected provider, which decides the parameter group's contents.
        self.provider = provider
        #: One-line capability status, sourced from lab_speech_status by the
        #: screen rather than re-derived here.
        self.capability_line = capability_line
        if studio_preferences is not None and (
            type(studio_preferences) is not StudioTTSPreferencesSnapshot
        ):
            raise TypeError("studio_preferences must be a Studio snapshot")
        if global_preferences is not None and (
            type(global_preferences) is not TTSPreferencesSnapshot
        ):
            raise TypeError("global_preferences must be a TTS preferences snapshot")
        self.studio_preferences = studio_preferences
        self._global_preferences = global_preferences
        self.global_preferences = global_preferences
        if navigation_target is not None and (
            type(navigation_target) is not SpeechTTSNavigationTarget
        ):
            raise TypeError("navigation_target must be a Speech TTS target")
        self._navigation_target = navigation_target
        self._navigation_focus_pending = navigation_target is not None
        self._navigation_seed_active = navigation_target is not None
        self._provider_configuration_states = dict(provider_configuration_states or {})
        if any(
            type(state) is not SpeechTTSConfigurationState
            for state in self._provider_configuration_states.values()
        ):
            raise TypeError("provider configuration states are invalid")
        self._runtime_status_store = (
            runtime_status_store or SpeechTTSRuntimeStatusStore()
        )
        if local_dependencies is None:
            local_dependencies = speech_local_dependency_availability(refresh=True)
        elif type(local_dependencies) is not SpeechLocalDependencyAvailability:
            raise TypeError("local_dependencies must be a Speech dependency snapshot")
        self._speech_local_dependencies = local_dependencies
        self._audio_cpp_runtime_observation: AudioCppRuntimeObservation | None = None
        self._audio_cpp_runtime_status_failed = False
        self._audio_cpp_runtime_request_generation = 0
        self._audio_cpp_lifecycle_busy: AudioCppRuntimeOperation | None = None
        self._audio_cpp_lifecycle_action: AudioCppRuntimeAction | None = None
        self._audio_cpp_primary_action: AudioCppRuntimeAction | None = None
        self._audio_cpp_primary_observation: AudioCppRuntimeCardObservation | None = (
            None
        )
        self._audio_cpp_action_generation = 0
        self._audio_cpp_sample_state: AudioCppSampleState = "not_attempted"
        self._audio_cpp_sample_identity: tuple[int, int, int] | None = None
        self._audio_cpp_sample_focus_target: str | None = None
        self._clone_setup_context: tuple[str, str, int, int] | None = None
        self._clone_setup_source_path: Path | None = None
        self._clone_setup_canonical: CanonicalTTSCloneReference | None = None
        self._clone_setup_draft_revision = 0
        self._clone_setup_validation_task: asyncio.Task[None] | None = None
        self._clone_setup_retained_tasks: set[asyncio.Task[None]] = set()
        self._clone_setup_validation_lock = asyncio.Lock()
        self._clone_setup_error: str | None = None
        self._profile_mount_generation = 0
        self.init_synthesis_state()
        self.init_catalog_state()
        self._seed_session_control_snapshot()
        self.init_playback_state()
        self.init_profile_state(profile_preset, profile_context_token)

    def _generation_complete(self, artifact: Any) -> None:
        """Retain exact profile-test provenance before playback sanitizes it."""

        if type(artifact) is STTSGeneratedAudio:
            self._retain_profile_generation_artifact(artifact)
        super()._generation_complete(artifact)

    def _seed_session_control_snapshot(self) -> None:
        """Restore bounded process-local axes after an internal Lab view switch."""

        provider_id = self.axis_values.get("tts-provider-select")
        if provider_id not in BUILT_IN_TTS_PROVIDER_IDS:
            return
        snapshot: dict[str, Any] = {}
        for control_id, snapshot_key in (
            ("tts-model-select", "model_id"),
            ("tts-voice-select", "voice_id"),
            ("tts-format-select", "response_format"),
        ):
            value = self.axis_values.get(control_id)
            if isinstance(value, str) and value:
                snapshot[snapshot_key] = value
        speed = self.axis_values.get("tts-speed-input")
        if isinstance(speed, str):
            try:
                snapshot["speed"] = float(speed)
            except ValueError:
                pass
        if snapshot:
            self._provider_control_snapshots[provider_id] = snapshot

    @staticmethod
    def _project_axis_defaults(
        studio: StudioTTSPreferencesSnapshot | None,
        global_preferences: TTSPreferencesSnapshot,
    ) -> dict[str, str]:
        """Project saved Studio overrides over one global preference snapshot."""

        defaults = {
            "tts-provider-select": global_preferences.provider_id,
            "tts-format-select": global_preferences.response_format,
            "tts-speed-input": str(global_preferences.speed),
        }
        if global_preferences.model_mode == "exact":
            assert global_preferences.model_id is not None
            defaults["tts-model-select"] = global_preferences.model_id
        if global_preferences.voice_mode == "exact":
            assert global_preferences.voice_id is not None
            defaults["tts-voice-select"] = global_preferences.voice_id
        if studio is None:
            return defaults

        selection = studio.selection
        if selection.provider_id is not None:
            if selection.provider_id != global_preferences.provider_id:
                defaults = {}
            defaults["tts-provider-select"] = selection.provider_id
        if selection.model_mode == "exact" and selection.model_id is not None:
            defaults["tts-model-select"] = selection.model_id
        elif selection.model_mode == "first_available":
            defaults.pop("tts-model-select", None)
        if selection.voice_mode == "exact" and selection.voice_id is not None:
            defaults["tts-voice-select"] = selection.voice_id
        elif selection.voice_mode == "server_default":
            defaults.pop("tts-voice-select", None)
        if selection.response_format is not None:
            defaults["tts-format-select"] = selection.response_format
        if selection.speed is not None:
            defaults["tts-speed-input"] = str(selection.speed)
        return defaults

    def refresh_global_preferences(self, snapshot: TTSPreferencesSnapshot) -> None:
        """Rebase inherited axes while preserving every session override."""

        if type(snapshot) is not TTSPreferencesSnapshot:
            raise TypeError("global preferences must be a TTS preferences snapshot")
        old_defaults = dict(self.axis_defaults)
        new_defaults = self._project_axis_defaults(self.studio_preferences, snapshot)
        protected_by_profile = self._profile_preset is not None
        changed_inherited_axes: set[str] = set()
        missing = object()
        for axis in set(old_defaults) | set(new_defaults):
            current = self.axis_values.get(axis, missing)
            old_default = old_defaults.get(axis, missing)
            new_default = new_defaults.get(axis, missing)
            if old_default == new_default:
                continue
            is_session_override = protected_by_profile or (
                current is not missing
                and (old_default is missing or current != old_default)
            )
            if is_session_override:
                continue
            changed_inherited_axes.add(axis)
            if new_default is missing:
                self.axis_values.pop(axis, None)
            else:
                self.axis_values[axis] = new_default

        self._global_preferences = snapshot
        self.global_preferences = snapshot
        self.axis_defaults = new_defaults
        if not changed_inherited_axes or not self.is_mounted:
            self._refresh_axis_markers()
            return

        provider_id = self.axis_values.get("tts-provider-select")
        provider_select = self.query_one("#tts-provider-select", Select)
        if (
            "tts-provider-select" in changed_inherited_axes
            and isinstance(provider_id, str)
            and provider_select.value != provider_id
        ):
            provider_select.value = provider_id
        elif isinstance(provider_id, str):
            provider_snapshot = self._provider_control_snapshots.get(provider_id)
            if provider_snapshot is not None:
                snapshot_keys = {
                    "tts-model-select": "model_id",
                    "tts-voice-select": "voice_id",
                    "tts-format-select": "response_format",
                    "tts-speed-input": "speed",
                }
                for axis in changed_inherited_axes:
                    snapshot_key = snapshot_keys.get(axis)
                    if snapshot_key is not None:
                        provider_snapshot.pop(snapshot_key, None)
            self._reproject_current_catalog()
        self._refresh_axis_markers()

    def _cli_setting(self, section: str, key: str, default: Any = None) -> Any:
        """Project saved Studio inheritance into the existing catalog loader.

        The loader already owns provider/model/voice restoration.  Supplying
        its normal cached-setting seam with the effective Studio seed avoids
        a second catalog path and performs no persistence or discovery.
        """

        studio = self.studio_preferences
        global_preferences = self._global_preferences
        if (
            section == "app_tts"
            and key == "default_provider"
            and self._navigation_target is not None
            and self._navigation_seed_active
        ):
            return self._navigation_target.provider_id
        if section == "app_tts" and key == "default_provider":
            session_provider = self.axis_values.get("tts-provider-select")
            if session_provider in BUILT_IN_TTS_PROVIDER_IDS:
                return session_provider
        if (
            section != "app_tts"
            or type(studio) is not StudioTTSPreferencesSnapshot
            or type(global_preferences) is not TTSPreferencesSnapshot
        ):
            return super()._cli_setting(section, key, default)

        selection = studio.selection
        saved_provider_id = selection.provider_id or global_preferences.provider_id
        navigation_provider_id = (
            self._navigation_target.provider_id
            if self._navigation_target is not None and self._navigation_seed_active
            else None
        )
        provider_id = navigation_provider_id or saved_provider_id
        if key == "default_provider":
            return provider_id
        navigation_changes_provider = (
            navigation_provider_id is not None
            and navigation_provider_id != saved_provider_id
        )
        global_applies = provider_id == global_preferences.provider_id
        if key == "default_model":
            if navigation_changes_provider:
                return None
            if selection.model_mode == "exact":
                return selection.model_id
            if selection.model_mode == "first_available":
                return None
            return (
                global_preferences.model_id
                if global_applies and global_preferences.model_mode == "exact"
                else None
            )
        if key == "default_voice":
            if navigation_changes_provider:
                return None
            if selection.voice_mode == "exact":
                return selection.voice_id
            if selection.voice_mode == "server_default":
                return None
            return (
                global_preferences.voice_id
                if global_applies and global_preferences.voice_mode == "exact"
                else None
            )
        if key == "default_format":
            if provider_id == "audio_cpp":
                return "wav"
            return selection.response_format or (
                global_preferences.response_format if global_applies else default
            )
        if key == "default_speed":
            if provider_id == "audio_cpp":
                return 1.0
            return (
                selection.speed
                if selection.speed is not None
                else (global_preferences.speed if global_applies else default)
            )
        return super()._cli_setting(section, key, default)

    def apply_navigation_target(self, target: SpeechTTSNavigationTarget) -> None:
        """Restore provider and focus only; never execute the requested action."""

        if type(target) is not SpeechTTSNavigationTarget:
            raise TypeError("target must be a Speech TTS navigation target")
        self._navigation_target = target
        self._navigation_focus_pending = True
        self._navigation_seed_active = True
        if not self.is_mounted:
            return
        try:
            provider_select = self.query_one("#tts-provider-select", Select)
        except NoMatches:
            return
        options = getattr(provider_select, "_options", ()) or ()
        provider_ids = {value for _label, value in options if isinstance(value, str)}
        if target.provider_id in provider_ids:
            provider_select.value = target.provider_id
        self.call_after_refresh(self._focus_navigation_target)

    def _consume_navigation_target_on_provider_change(
        self,
        provider_id: str,
    ) -> None:
        """Drop one-shot deep-link overrides when the user leaves its provider."""

        target = self._navigation_target
        if target is None or provider_id == target.provider_id:
            return
        self._navigation_target = None
        self._navigation_focus_pending = False
        self._navigation_seed_active = False

    def _consume_navigation_target_after_catalog(self, provider_id: str) -> None:
        """Retire the one-shot selection seed after its first projection."""

        target = self._navigation_target
        if target is None or provider_id != target.provider_id:
            return
        self._navigation_seed_active = False
        if not self._navigation_focus_pending:
            self._navigation_target = None

    def _focus_navigation_target(self, retries_remaining: int = 4) -> None:
        target = self._navigation_target
        if target is None or not self._navigation_focus_pending or not self.is_mounted:
            return
        selector = (
            "#tts-provider-select"
            if target.intent in {None, SpeechTTSNavigationIntent.CONFIGURE}
            else "#tts-test-connection-btn"
            if target.intent is SpeechTTSNavigationIntent.TEST
            else "#tts-voice-select"
            if target.intent is SpeechTTSNavigationIntent.REFRESH_VOICES
            else "#tts-refresh-catalog-btn"
        )
        try:
            control = self.query_one(selector)
            control.focus()
        except NoMatches:
            return
        if getattr(self.app.focused, "id", None) == control.id:
            self._navigation_focus_pending = False
            if not self._navigation_seed_active:
                self._navigation_target = None
            return
        if retries_remaining > 0:
            self.call_after_refresh(
                self._focus_navigation_target,
                retries_remaining - 1,
            )

    @on(Button.Pressed, "#tts-adopt-studio-preferences-btn")
    def _on_adopt_studio_preferences(self, event: Button.Pressed) -> None:
        """Move an exact preview to the Studio editor only on explicit intent."""

        event.stop()
        preset = self._profile_preset
        if type(preset) is not TTSPlaygroundSelectionPreset:
            return
        self.post_message(AdoptStudioPreferencesRequested(preset))

    @on(Button.Pressed, "#tts-open-studio-preferences-btn")
    def _on_open_studio_preferences(self, event: Button.Pressed) -> None:
        """Open the Studio-only editor without persisting Playground state."""

        event.stop()
        self.post_message(OpenStudioPreferencesRequested())

    @on(Button.Pressed, "#audio-cpp-runtime-open-settings")
    def _on_open_audio_cpp_global_settings(self, event: Button.Pressed) -> None:
        """Open the one canonical owner of durable audio.cpp configuration."""

        event.stop()
        self.app.post_message(
            NavigateToScreen(
                "settings",
                {
                    "category": "speech-tts",
                    **speech_tts_navigation_context(
                        SpeechTTSNavigationTarget(
                            "audio_cpp",
                            SpeechTTSNavigationIntent.CONFIGURE,
                        )
                    ),
                },
            )
        )

    @on(Button.Pressed, "#speech-clone-reference-choose")
    def _on_choose_clone_reference(self, event: Button.Pressed) -> None:
        """Open the bounded WAV picker for the visible clone setup."""

        event.stop()
        picker = FileOpen(
            title="Choose reference WAV",
            filters=Filters(("WAV audio", lambda path: path.suffix.lower() == ".wav")),
            context="speech_clone_reference",
        )
        self.app.push_screen(picker, self._handle_clone_reference_selection)

    @on(Button.Pressed, "#speech-clone-reference-clear")
    def _on_clear_clone_reference(self, event: Button.Pressed) -> None:
        """Drop only the staged clone reference and retain the transcript."""

        event.stop()
        self._replace_clone_reference_source(None)

    @on(Button.Pressed, "#speech-clone-use-profile")
    def _on_use_existing_voice_profile(self, event: Button.Pressed) -> None:
        """Open the existing Voice Profiles library without duplicating a picker."""

        event.stop()
        self.post_message(OpenVoiceProfilesRequested())

    def _handle_clone_reference_selection(self, path: str | Path | None) -> None:
        """Accept picker cancellation or stage one source path for validation."""

        if path is None:
            return
        self._replace_clone_reference_source(Path(path))

    def _replace_clone_reference_source(self, path: Path | None) -> None:
        """Fence the old draft, retain no basename, and validate the new source."""

        self._clone_setup_draft_revision += 1
        previous = self._clone_setup_validation_task
        if path is None:
            if previous is not None and not previous.done():
                previous.cancel()
            self._clone_setup_validation_task = None
        self._clone_setup_source_path = path
        self._clone_setup_canonical = None
        self._clone_setup_error = None
        if path is not None:
            self._schedule_clone_reference_validation(delay_seconds=0.0)
        self._refresh_clone_setup_presentation()

    def _clone_transcript(self) -> str:
        try:
            return self.query_one("#speech-clone-reference-text", TextArea).text
        except NoMatches:
            return ""

    def _schedule_clone_reference_validation(self, *, delay_seconds: float) -> None:
        """Retain one revision-fenced canonicalization task off the event loop."""

        source = self._clone_setup_source_path
        context = self._clone_setup_context
        if source is None or context is None:
            return
        transcript = self._clone_transcript()
        try:
            normalized_text = _validate_clone_transcript_input(transcript)
        except Exception:
            self._clone_setup_canonical = None
            self._clone_setup_error = "Enter the exact spoken transcript."
            self._refresh_clone_setup_presentation()
            return
        self._clone_setup_draft_revision += 1
        revision = self._clone_setup_draft_revision
        previous = self._clone_setup_validation_task
        if previous is not None and not previous.done():
            previous.cancel()
        task = asyncio.create_task(
            self._validate_clone_reference(
                source=source,
                transcript=normalized_text,
                context=context,
                revision=revision,
                delay_seconds=delay_seconds,
            ),
            name="speech_clone_reference_validation",
        )
        self._clone_setup_validation_task = task
        self._clone_setup_retained_tasks.add(task)
        task.add_done_callback(self._clone_setup_retained_tasks.discard)
        self._clone_setup_error = None
        self._refresh_clone_setup_presentation()

    async def _validate_clone_reference(
        self,
        *,
        source: Path,
        transcript: str,
        context: tuple[str, str, int, int],
        revision: int,
        delay_seconds: float,
    ) -> None:
        """Join superseded work and publish only exact current canonical bytes."""

        current = asyncio.current_task()
        assert current is not None
        try:
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)
            async with self._clone_setup_validation_lock:
                if (
                    self._clone_setup_validation_task is not current
                    or self._clone_setup_draft_revision != revision
                    or self._clone_setup_context != context
                    or self._clone_setup_source_path != source
                ):
                    return
                thread_task = asyncio.create_task(
                    asyncio.to_thread(canonicalize_reference_wav, source, transcript),
                    name="speech_clone_reference_canonicalize",
                )
                try:
                    canonical = await asyncio.shield(thread_task)
                except asyncio.CancelledError:
                    try:
                        await thread_task
                    except BaseException:
                        pass
                    raise
                except Exception:
                    if (
                        self._clone_setup_validation_task is current
                        and self._clone_setup_draft_revision == revision
                        and self._clone_setup_context == context
                        and self._clone_setup_source_path == source
                    ):
                        self._clone_setup_canonical = None
                        self._clone_setup_error = (
                            "Choose a valid, bounded PCM WAV reference."
                        )
                        self.call_after_refresh(
                            self._focus_audio_cpp_action_target,
                            "#speech-clone-reference-choose",
                        )
                    return
                if (
                    self.is_mounted
                    and self._clone_setup_validation_task is current
                    and self._clone_setup_draft_revision == revision
                    and self._clone_setup_context == context
                    and self._clone_setup_source_path == source
                ):
                    self._clone_setup_canonical = canonical
                    self._clone_setup_error = None
        finally:
            if self._clone_setup_validation_task is current:
                self._clone_setup_validation_task = None
                self._refresh_clone_setup_presentation()

    async def _close_clone_setup(self) -> None:
        """Seal and join the exact retained canonicalization task on teardown."""

        self._clone_setup_draft_revision += 1
        task = self._clone_setup_validation_task
        self._clone_setup_validation_task = None
        tasks = set(self._clone_setup_retained_tasks)
        if task is not None:
            tasks.add(task)
        for retained in tasks:
            if not retained.done() and retained.cancelling() == 0:
                retained.cancel()
        for retained in tasks:
            try:
                await retained
            except BaseException:
                pass
        self._clone_setup_retained_tasks.clear()
        self._clone_setup_source_path = None
        self._clone_setup_canonical = None
        self._clone_setup_error = None
        self._clone_setup_context = None

    def _refresh_clone_setup_presentation(self) -> None:
        """Reproject setup and primary action from current immutable facts."""

        observation = self._audio_cpp_runtime_observation
        if observation is None or not self.is_mounted:
            return
        try:
            setup = self.query_one("#speech-clone-setup", SpeechCloneSetup)
            setup.apply_draft_state(
                self._clone_setup_draft_state(observation),
                source_selected=self._clone_setup_source_path is not None,
                error_copy=self._clone_setup_error,
            )
            card_observation = self._audio_cpp_card_observation(observation)
            projection = project_audio_cpp_runtime_card(card_observation)
            self._set_audio_cpp_primary_action(
                projection.primary_action,
                card_observation,
            )
            primary = self.query_one("#tts-test-connection-btn", Button)
            primary.label = projection.primary_action.label
            primary.disabled = not projection.primary_action.enabled
            primary.tooltip = projection.primary_action.tooltip
            primary.refresh(layout=True)
            self._sync_generate_enabled()
        except NoMatches:
            return

    def _generate_clone_audition(self) -> None:
        """Generate only when the visible exact clone action remains enabled."""

        observation = self._audio_cpp_runtime_observation
        if observation is None:
            return
        card_observation = self._audio_cpp_card_observation(observation)
        action = project_audio_cpp_runtime_card(card_observation).primary_action
        if action.operation != "clone_generate" or not action.enabled:
            self.app.notify(
                action.disabled_reason or "Voice setup is not ready.",
                severity="warning",
            )
            return
        self._generate_tts(clone_action=True)

    def _clone_setup_generation_error(
        self,
        provider_id: object,
        model_id: object,
        *,
        clone_action: bool,
    ) -> str | None:
        """Reserve clone-capable models for their exact projected action."""

        if (
            provider_id != "audio_cpp"
            or not isinstance(model_id, str)
            or self._profile_preset is not None
        ):
            return None
        observation = self._audio_cpp_runtime_observation
        projection = None if observation is None else observation.clone_setup
        if projection is None or projection.model_id != model_id:
            return None
        card_observation = self._audio_cpp_card_observation(observation)
        action = project_audio_cpp_runtime_card(card_observation).primary_action
        if clone_action and action.operation == "clone_generate" and action.enabled:
            return None
        if action.operation == "clone_generate" and not action.enabled:
            return action.disabled_reason or "Complete voice setup before generating."
        return "Use Create Voice & Generate for this reference-based model."

    def _clone_audition_for_request(
        self,
        provider_id: str,
        model_id: str,
    ) -> STTSPlaygroundCloneSnapshot | None:
        """Freeze the exact current private draft below the public UI boundary."""

        if provider_id != "audio_cpp" or self._profile_preset is not None:
            return None
        observation = self._audio_cpp_runtime_observation
        canonical = self._clone_setup_canonical
        projection = None if observation is None else observation.clone_setup
        if (
            observation is None
            or projection is None
            or projection.model_id != model_id
            or self._clone_setup_context
            != self._clone_setup_observation_context(observation)
            or canonical is None
            or self._clone_setup_draft_state(observation) != "ready"
            or observation.pending_configuration
            or observation.process.state != "running"
            or observation.tts_capability != "available"
            or not observation.catalog_fresh
        ):
            return None
        return STTSPlaygroundCloneSnapshot(
            draft_revision=self._clone_setup_draft_revision,
            canonical_reference=canonical,
        )

    def _accept_clone_generation_result(
        self,
        operation_id: str,
        draft_revision: int,
    ) -> None:
        """Drop only the exact staged draft accepted by the handler result."""

        if (
            self._generation_operation_id != operation_id
            or self._clone_setup_draft_revision != draft_revision
            or self._clone_setup_canonical is None
        ):
            return
        self._clone_setup_draft_revision += 1
        self._clone_setup_source_path = None
        self._clone_setup_canonical = None
        self._clone_setup_error = None
        self._refresh_clone_setup_presentation()

    @on(Button.Pressed, "#audio-cpp-runtime-restart")
    def _on_audio_cpp_restart(self, event: Button.Pressed) -> None:
        """Run the existing managed replacement operation from Speech Lab."""

        event.stop()
        self._request_audio_cpp_lifecycle("restart")

    @on(Button.Pressed, "#audio-cpp-runtime-shutdown")
    def _on_audio_cpp_shutdown(self, event: Button.Pressed) -> None:
        """Run process shutdown without touching the current audio result."""

        event.stop()
        self._request_audio_cpp_lifecycle("shutdown")

    def _handle_connection_action(self) -> bool:
        """Route audio.cpp's state-specific primary action through the service."""

        if self._selected_runtime_provider() != "audio_cpp":
            return False
        action = self._audio_cpp_primary_action or project_audio_cpp_unknown_action()
        if not action.enabled:
            if action.disabled_reason:
                self.app.notify(action.disabled_reason, severity="warning")
            return True
        if action.operation == "clone_generate":
            self._generate_clone_audition()
            return True
        self._request_audio_cpp_lifecycle(
            action.operation,
            projected_action=action,
            projected_observation=self._audio_cpp_primary_observation,
        )
        return True

    @on(Switch.Changed)
    def on_tts_option_switch_changed(self, event: Switch.Changed) -> None:
        """Delegate to the shared catalog mixin.

        Args:
            event: Any Switch change in this pane.
        """
        self.handle_option_switch_changed(event)

    @on(Input.Changed)
    def on_tts_speed_changed(self, event: Input.Changed) -> None:
        """Mirror an axis edit, then delegate to the shared catalog mixin.

        dev wired the delegation up while the rebuild was in flight -- it
        had been an undispatched name, which is why an earlier version of
        this branch deleted it as dead. It is live now, so the pane carries
        it. The mirroring is this ruling's addition: the speed axis is the
        only Input among the axes, so this is also the pane's one
        Input.Changed chokepoint.

        Args:
            event: Any Input change in this pane.
        """
        self._mirror_axis_edit(event.input.id, event.value)
        self.handle_speed_changed(event)

    @on(TextArea.Changed)
    def on_tts_text_changed(self, event: TextArea.Changed) -> None:
        """Delegate to the shared catalog mixin.

        Args:
            event: Any TextArea change in this pane.
        """
        if event.text_area.id == "speech-clone-reference-text":
            try:
                setup = self.query_one("#speech-clone-setup", SpeechCloneSetup)
                setup.update_transcript_guidance(event.text_area.text)
            except NoMatches:
                return
            if self._clone_setup_source_path is None:
                self._clone_setup_error = "Choose a reference WAV."
                self._refresh_clone_setup_presentation()
                return
            canonical = self._clone_setup_canonical
            try:
                transcript = _validate_clone_transcript_input(event.text_area.text)
            except Exception:
                self._clone_setup_canonical = None
                self._clone_setup_error = "Enter the exact spoken transcript."
                self._refresh_clone_setup_presentation()
                return
            if canonical is not None:
                self._clone_setup_draft_revision += 1
                self._clone_setup_canonical = CanonicalTTSCloneReference(
                    wav_bytes=canonical.wav_bytes,
                    reference_text=transcript,
                    sha256=canonical.sha256,
                    byte_length=canonical.byte_length,
                    duration_ms=canonical.duration_ms,
                    sample_rate_hz=canonical.sample_rate_hz,
                    channels=canonical.channels,
                    sample_encoding=canonical.sample_encoding,
                )
                self._clone_setup_error = None
                self._refresh_clone_setup_presentation()
                return
            self._schedule_clone_reference_validation(delay_seconds=0.3)
            return
        self.handle_text_changed(event)

    @on(Select.Changed)
    def on_tts_provider_select_changed(self, event: Select.Changed) -> None:
        """Mirror an axis edit, then delegate to the shared catalog mixin.

        The decorator has to live on this class: Textual collects `@on`
        handlers per-class in its metaclass, so one declared in the mixin is
        never registered. Despite the name, this sees every Select in the
        pane, not only the provider one -- which is exactly why it is also
        the pane's one Select.Changed chokepoint for mirroring axis edits.

        Args:
            event: Any Select change in this pane.
        """
        self._mirror_axis_edit(event.select.id, event.value)
        self.handle_provider_select_changed(event)
        if (
            event.select.id == "tts-model-select"
            and isinstance(event.value, str)
            and self._selected_runtime_provider() == "audio_cpp"
        ):
            self._clear_clone_setup(remove_component=True)
            self._audio_cpp_runtime_observation = None
            self._request_audio_cpp_runtime_observation()

    def _mirror_axis_edit(self, control_id: str | None, value: object) -> None:
        """Mirror a user's direct edit of an axis control into the model.

        No `_applying_catalog_controls` guard: an earlier version of this
        method skipped mirroring while that flag was True, on the theory
        that it would otherwise double-mirror a catalog-driven write. It
        cannot, and never did -- `_apply_controls` and
        `_prime_profile_preset_controls` never hold the flag True across an
        `await`, so it is always back to False by the time Textual delivers
        the `Select.Changed`/`Input.Changed` that write produced (a
        five-scenario instrumented check found zero guard-True entries).
        Reusing that dead branch would have masked the real reason mirroring
        one of those deferred, catalog-driven messages is harmless: a
        widget's own `Changed` queue is FIFO, so this always sees the LAST
        value written to that control, and `_apply_controls`/
        `_prime_profile_preset_controls` already wrote that identical value
        into `axis_values` synchronously, before the message was even
        queued -- mirroring it again is a redundant no-op, not a
        correctness risk.

        Args:
            control_id: The changed widget's id, or ``None``.
            value: The widget's new value. Ignored unless it is a plain
                ``str`` -- the sentinel values `Select` uses for "loading"
                and "unavailable" are not session edits.
        """
        if control_id not in AXIS_CONTROLS or not isinstance(value, str):
            return
        self.axis_values[control_id] = value
        self._refresh_axis_markers()

    def _refresh_axis_markers(self) -> None:
        """Repaint the axis row's override markers from the current model.

        Tolerates the row not being mounted yet (`NoMatches`): this is
        called from `_apply_controls` and `_prime_profile_preset_controls`,
        which can run before `compose()` has yielded `#speech-axis-row`, or
        in a unit-test context that never mounts it.
        """
        try:
            row = self.query_one("#speech-axis-row", SpeechAxisRow)
        except NoMatches:
            return
        row.update_defaults(self.axis_defaults)
        row.update_values(self.axis_values)
        self._mark_profile_test_axes()

    # NOTE (task-15476): there used to be a second handler here,
    # `on_text_area_changed`, matched by Textual's implicit
    # `on_<message>` naming convention. Every `TextArea.Changed` in this
    # pane is ALSO caught by the `@on(TextArea.Changed)`-decorated
    # `on_tts_text_changed` above, which falls through to
    # `self.handle_text_changed(event)` -> `_sync_generate_enabled()` for
    # any text area other than the clone-reference one -- including
    # `tts-text-input`. The two handlers fired on every keystroke into the
    # main TTS text box, each independently materializing the full
    # `TextArea.text` inside `_sync_generate_enabled`. Deleted as a pure
    # duplicate; `_sync_generate_enabled()` still runs exactly once per
    # keystroke via the fallthrough above.

    def _show_provider_specific_controls(self, provider: str) -> None:
        """Re-scope the parameter group and clip picker to `provider`.

        This is the one method in the catalog closure that was bound to the
        legacy layout, where it toggled a `hidden` class on five per-provider
        container boxes. Those containers are gone -- `SpeechParamGroup`
        mounts only the selected provider's knobs -- so the override does the
        equivalent thing for this layout.

        Swaps the two provider-scoped regions rather than recomposing the
        pane. `refresh(recompose=True)` destroys and rebuilds every child,
        including the axis selects the catalog loader is in the middle of
        populating -- the exact hazard `SpeechCatalogMixin` documents, which
        this method previously walked into. It cost five provider-switch
        tests, each timing out waiting for state written to a widget that no
        longer existed.

        Args:
            provider: The newly selected provider id.
        """
        # Before the early return: this runs on catalog application as well
        # as on user change, and the language axis needs settling either way.
        self._settle_language_axis(provider)
        self._sync_audio_cpp_runtime_visibility(provider)

        if provider == self.provider:
            return
        self.provider = provider

        try:
            group = self.query_one("#speech-param-group")
            _text_pane = self.query_one("#speech-text-pane")
        except NoMatches:
            return

        # Textual's `remove()` is DEFERRED: the widget is still mounted
        # when this returns, so mounting the replacement immediately raises
        # DuplicateIds on `speech-param-group`. The shared parameters
        # (`AUDIO_PARAMS`, `REQUEST_PARAMS`) are in every provider's group,
        # so their ids collide too.
        #
        # The callback RECONCILES to whatever provider is current rather than
        # mounting the one captured here. Two provider changes in quick
        # succession -- which is normal, since loading a catalog can trigger
        # another -- otherwise queue two mounts and the second duplicates the
        # first.
        source_rows = tuple(self.query(".speech-source-row"))
        if self._provider_regions_replacing:
            return
        self._provider_regions_replacing = True
        self.run_worker(
            self._replace_provider_regions(group, source_rows),
            group="speech-provider-regions",
            exclusive=False,
            exit_on_error=False,
        )

    async def _replace_provider_regions(
        self,
        group: Any,
        source_rows: tuple[Any, ...],
    ) -> None:
        """Await deferred removals before reconciling the current provider."""

        try:
            if group.is_mounted:
                await group.remove()
            for row in source_rows:
                if row.is_mounted:
                    await row.remove()
        finally:
            self._provider_regions_replacing = False
        if self.is_mounted:
            self._reconcile_provider_regions()

    def _reconcile_provider_regions(self) -> None:
        """Mount the provider-scoped regions if they are not already there.

        Idempotent by design: it is scheduled once per provider change but
        several may be pending, and only the current provider should end up
        on screen.
        """
        if not self.is_mounted:
            return
        try:
            anchor = self.query_one("#speech-connection-details")
            text_pane = self.query_one("#speech-text-pane")
        except NoMatches:
            return

        groups = list(self.query("#speech-param-group"))
        if not groups:
            self.mount(
                SpeechParamGroup(
                    provider=self.provider,
                    values=self._saved_studio_param_values(self.provider),
                    id="speech-param-group",
                ),
                before=anchor,
            )
        if not self.query(".speech-source-row"):
            for widget in self._compose_voice_source():
                text_pane.mount(widget)

    def _saved_studio_param_values(self, provider: str) -> dict[str, object]:
        """Return saved request-scoped values keyed by Playground control ID."""

        preferences = self.studio_preferences
        if type(preferences) is not StudioTTSPreferencesSnapshot:
            return {}
        options = preferences.provider_options.get(provider, {})
        controls = {
            "exaggeration": "#tts-exaggeration-input",
            "cfg_weight": "#tts-cfg-weight-input",
        }
        return {
            controls[option].removeprefix("#"): value
            for option, value in options.items()
            if option in controls
        }

    def _settle_language_axis(self, provider: str) -> None:
        """Say something terminal on the language axis once a catalog is in.

        Language is Kokoro-only; the legacy screen hid the row entirely for
        every other provider. As a permanent axis it cannot be hidden, so it
        has to say what is true instead -- and "Waiting for provider…" stops
        being true the moment that provider's catalog has arrived without
        languages. Left alone it waits forever.

        Args:
            provider: The provider whose catalog is being applied.
        """
        try:
            select = self.query_one("#tts-language-select", Select)
            cell = self.query_one("#speech-axis-cell-tts-language-select")
        except NoMatches:
            return
        applicable = provider == "kokoro"
        cell.set_class(not applicable, "hidden")
        cell.display = applicable
        if applicable:
            select.disabled = False
            return
        label = AXIS_EMPTY_PROMPTS["tts-language-select"]
        select.set_options([(label, UNAVAILABLE_SELECT_VALUE)])
        select.value = UNAVAILABLE_SELECT_VALUE
        select.disabled = True

    def _refresh_provider_ids(self) -> None:
        """Record the provider values on offer.

        `_generation_readiness_error` distinguishes a selection that has gone
        stale from one that was never valid, and needs the current option
        set to do it.
        """
        try:
            select = self.query_one("#tts-provider-select", Select)
        except Exception:  # noqa: BLE001 - not yet mounted
            return
        options = getattr(select, "_options", ()) or ()
        self._provider_ids = frozenset(
            value for _label, value in options if isinstance(value, str)
        )

    def _selected_configuration_state(
        self, provider_id: str
    ) -> SpeechTTSConfigurationState:
        return self._provider_configuration_states.get(
            provider_id,
            SpeechTTSConfigurationState.DEFAULT,
        )

    def _provider_revision(
        self,
        provider_id: str,
        reader_name: str,
        fallback: int | None,
    ) -> int | None:
        service = self._tts_service
        revision_reader = getattr(service, reader_name, None)
        if not callable(revision_reader):
            return fallback
        try:
            revision = revision_reader(provider_id)
        except (KeyError, RuntimeError, TypeError, ValueError):
            return fallback
        return revision if type(revision) is int and revision >= 0 else fallback

    def _current_runtime_revision(self, provider_id: str) -> int | None:
        return self._provider_revision(
            provider_id,
            "configuration_revision",
            self._catalog_configuration_revisions.get(provider_id),
        )

    def _current_saved_configuration_revision(
        self,
        provider_id: str,
        runtime_revision: int | None,
    ) -> int | None:
        return self._provider_revision(
            provider_id,
            "saved_configuration_revision",
            runtime_revision,
        )

    def _current_applied_configuration_revision(
        self,
        provider_id: str,
        saved_configuration_revision: int | None,
    ) -> int | None:
        return self._provider_revision(
            provider_id,
            "applied_configuration_revision",
            saved_configuration_revision,
        )

    @staticmethod
    def _newest_status(
        first: SpeechTTSRuntimeStatus | None,
        second: SpeechTTSRuntimeStatus | None,
        *,
        catalog_axis: bool,
    ) -> SpeechTTSRuntimeStatus | None:
        return newest_speech_tts_status(
            first,
            second,
            catalog_axis=catalog_axis,
        )

    def _operation_status(
        self,
        provider_id: str,
        *,
        saved_configuration_revision: int,
        runtime_revision: int | None,
        catalog_revision: int | None,
        model_id: str | None,
        runtime_state: SpeechTTSRuntimeState,
        diagnostic_category: SpeechTTSDiagnosticCategory | None = None,
        recovery_action: SpeechTTSNavigationIntent | None = None,
        observed_at: datetime | None = None,
    ) -> SpeechTTSRuntimeStatus:
        return SpeechTTSRuntimeStatus(
            provider_id=provider_id,
            saved_configuration_revision=saved_configuration_revision,
            runtime_revision=runtime_revision,
            catalog_revision=catalog_revision,
            model_scope=speech_tts_model_scope(model_id),
            runtime_state=runtime_state,
            observed_at=(
                observed_at
                or self._catalog_runtime_observed_at.get(
                    provider_id,
                    datetime.now(timezone.utc),
                )
            ),
            freshness=(
                SpeechTTSStatusFreshness.STALE
                if runtime_state is SpeechTTSRuntimeState.STALE
                else SpeechTTSStatusFreshness.FRESH
            ),
            diagnostic_category=diagnostic_category,
            recovery_action=recovery_action,
        )

    def _runtime_status_for_selected(
        self,
        provider_id: str,
        *,
        saved_configuration_revision: int | None,
        current_runtime_revision: int | None,
        applied_configuration_revision: int | None,
    ) -> SpeechTTSRuntimeStatus | None:
        shared = self._runtime_status_store.runtime_status(provider_id)
        if saved_configuration_revision is None:
            return shared
        local: SpeechTTSRuntimeStatus | None = None
        if provider_id in self._catalog_checking_providers:
            local = self._operation_status(
                provider_id,
                saved_configuration_revision=saved_configuration_revision,
                runtime_revision=current_runtime_revision,
                catalog_revision=None,
                model_id=None,
                runtime_state=SpeechTTSRuntimeState.CHECKING,
            )
        elif provider_id in self._catalog_unavailable_providers:
            local = self._operation_status(
                provider_id,
                saved_configuration_revision=saved_configuration_revision,
                runtime_revision=current_runtime_revision,
                catalog_revision=None,
                model_id=None,
                runtime_state=SpeechTTSRuntimeState.UNAVAILABLE,
                diagnostic_category=SpeechTTSDiagnosticCategory.CONNECTION,
                recovery_action=SpeechTTSNavigationIntent.TEST,
            )
        else:
            observed_runtime_revision = self._catalog_configuration_revisions.get(
                provider_id
            )
            catalog = self._catalogs.get(provider_id)
            if (
                observed_runtime_revision is not None
                and current_runtime_revision is not None
                and applied_configuration_revision is not None
                and catalog is not None
            ):
                observed_at = self._catalog_observed_at.get(
                    provider_id,
                    self._catalog_runtime_observed_at.get(
                        provider_id,
                        datetime.now(timezone.utc),
                    ),
                )
                local = speech_tts_runtime_status_from_catalog(
                    provider_id=provider_id,
                    saved_configuration_revision=saved_configuration_revision,
                    applied_configuration_revision=applied_configuration_revision,
                    observed_runtime_revision=observed_runtime_revision,
                    current_runtime_revision=current_runtime_revision,
                    catalog=catalog,
                    model_id=None,
                    observed_at=observed_at,
                )
        return self._newest_status(local, shared, catalog_axis=False)

    def _catalog_status_for_selected(
        self,
        provider_id: str,
        *,
        saved_configuration_revision: int | None,
        current_runtime_revision: int | None,
        applied_configuration_revision: int | None,
        model_id: str | None,
    ) -> SpeechTTSRuntimeStatus | None:
        shared = self._runtime_status_store.catalog_status(provider_id, model_id)
        if saved_configuration_revision is None:
            return shared
        local: SpeechTTSRuntimeStatus | None = None
        catalog = self._catalogs.get(provider_id)
        observed_runtime_revision = self._catalog_configuration_revisions.get(
            provider_id
        )
        catalog_revision = catalog.revision if catalog is not None else None
        if provider_id in self._catalog_checking_providers:
            local = self._operation_status(
                provider_id,
                saved_configuration_revision=saved_configuration_revision,
                runtime_revision=current_runtime_revision,
                catalog_revision=catalog_revision,
                model_id=model_id,
                runtime_state=SpeechTTSRuntimeState.CHECKING,
            )
        elif provider_id in self._catalog_unavailable_providers:
            if (
                catalog is not None
                and observed_runtime_revision is not None
                and current_runtime_revision is not None
                and applied_configuration_revision is not None
            ):
                local = speech_tts_runtime_status_from_catalog(
                    provider_id=provider_id,
                    saved_configuration_revision=saved_configuration_revision,
                    applied_configuration_revision=applied_configuration_revision,
                    observed_runtime_revision=observed_runtime_revision,
                    current_runtime_revision=current_runtime_revision,
                    catalog=catalog,
                    model_id=model_id,
                    observed_at=self._catalog_runtime_observed_at.get(
                        provider_id,
                        self._catalog_observed_at.get(
                            provider_id,
                            datetime.now(timezone.utc),
                        ),
                    ),
                    catalog_axis=True,
                )
                if (
                    provider_id in self._stale_providers
                    and local.runtime_state is not SpeechTTSRuntimeState.STALE
                ):
                    local = replace(
                        local,
                        runtime_state=SpeechTTSRuntimeState.STALE,
                        freshness=SpeechTTSStatusFreshness.STALE,
                        diagnostic_category=SpeechTTSDiagnosticCategory.CATALOG,
                        recovery_action=SpeechTTSNavigationIntent.REFRESH_MODELS,
                    )
            else:
                local = self._operation_status(
                    provider_id,
                    saved_configuration_revision=saved_configuration_revision,
                    runtime_revision=current_runtime_revision,
                    catalog_revision=None,
                    model_id=model_id,
                    runtime_state=SpeechTTSRuntimeState.UNAVAILABLE,
                    diagnostic_category=SpeechTTSDiagnosticCategory.CONNECTION,
                    recovery_action=SpeechTTSNavigationIntent.TEST,
                )
        elif not (
            catalog is None
            or observed_runtime_revision is None
            or current_runtime_revision is None
            or applied_configuration_revision is None
        ):
            local = speech_tts_runtime_status_from_catalog(
                provider_id=provider_id,
                saved_configuration_revision=saved_configuration_revision,
                applied_configuration_revision=applied_configuration_revision,
                observed_runtime_revision=observed_runtime_revision,
                current_runtime_revision=current_runtime_revision,
                catalog=catalog,
                model_id=model_id,
                observed_at=self._catalog_observed_at.get(
                    provider_id,
                    datetime.now(timezone.utc),
                ),
                catalog_axis=True,
            )
            voice_scope = (provider_id, model_id or "")
            voice_observed_at = self._voice_runtime_observed_at.get(voice_scope)
            if voice_observed_at is not None:
                local = replace(local, observed_at=voice_observed_at)
            if voice_scope in self._voice_checking_scopes:
                local = replace(
                    local,
                    runtime_state=SpeechTTSRuntimeState.CHECKING,
                    freshness=SpeechTTSStatusFreshness.FRESH,
                    diagnostic_category=None,
                    recovery_action=None,
                )
            elif (
                provider_id in self._stale_providers
                and local.runtime_state is not SpeechTTSRuntimeState.STALE
            ):
                local = replace(
                    local,
                    runtime_state=SpeechTTSRuntimeState.STALE,
                    freshness=SpeechTTSStatusFreshness.STALE,
                    diagnostic_category=SpeechTTSDiagnosticCategory.CATALOG,
                    recovery_action=SpeechTTSNavigationIntent.REFRESH_MODELS,
                )
            elif (
                voice_scope in self._voice_unavailable_scopes
                and local.runtime_state is not SpeechTTSRuntimeState.STALE
            ):
                local = replace(
                    local,
                    runtime_state=SpeechTTSRuntimeState.STALE,
                    freshness=SpeechTTSStatusFreshness.STALE,
                    diagnostic_category=SpeechTTSDiagnosticCategory.CATALOG,
                    recovery_action=SpeechTTSNavigationIntent.REFRESH_VOICES,
                )
        return self._newest_status(local, shared, catalog_axis=True)

    def _truthful_status_rows(self):
        provider_id = self._selected_provider_id
        if provider_id is None:
            configured = self._cli_setting(
                "app_tts",
                "default_provider",
                "audio_cpp",
            )
            provider_id = configured if isinstance(configured, str) else "audio_cpp"
        try:
            model_value = self._current_select_value("#tts-model-select")
        except NoMatches:
            model_value = None
        model_id = model_value if isinstance(model_value, str) else None
        current_runtime_revision = self._current_runtime_revision(provider_id)
        saved_configuration_revision = self._current_saved_configuration_revision(
            provider_id,
            current_runtime_revision,
        )
        applied_configuration_revision = self._current_applied_configuration_revision(
            provider_id,
            saved_configuration_revision,
        )
        runtime_status = self._runtime_status_for_selected(
            provider_id,
            saved_configuration_revision=saved_configuration_revision,
            current_runtime_revision=current_runtime_revision,
            applied_configuration_revision=applied_configuration_revision,
        )
        catalog_status = self._catalog_status_for_selected(
            provider_id,
            saved_configuration_revision=saved_configuration_revision,
            current_runtime_revision=current_runtime_revision,
            applied_configuration_revision=applied_configuration_revision,
            model_id=model_id,
        )
        projection = project_speech_tts_status(
            provider_id=provider_id,
            configuration_state=self._selected_configuration_state(provider_id),
            current_configuration_revision=saved_configuration_revision,
            current_runtime_revision=current_runtime_revision,
            applied_configuration_revision=applied_configuration_revision,
            model_id=model_id,
            observation=None,
            local_dependencies=self._speech_local_dependencies,
            runtime_status=runtime_status,
            catalog_status=catalog_status,
        )
        if projection.runtime_status is not None:
            self._runtime_status_store.publish_runtime(projection.runtime_status)
        if projection.catalog_status is not None:
            self._runtime_status_store.publish_catalog(projection.catalog_status)
        return projection.rows()

    def _sync_truthful_status_rows(self) -> None:
        """Render independent bounded rows from accepted in-memory facts."""

        if not self.is_mounted:
            return
        for row in self._truthful_status_rows():
            try:
                self.query_one(f"#speech-status-{row.row_id}", Static).update(row.copy)
            except NoMatches:
                continue
        if self._navigation_focus_pending:
            self.call_after_refresh(self._focus_navigation_target)

    def _selected_runtime_provider(self) -> str:
        """Return the provider whose runtime card may currently be observed."""

        return self._selected_provider_id or self.provider

    def _generation_readiness_error(
        self,
        provider_id: object,
        model_id: object,
        *,
        clone_action: bool = False,
    ) -> str | None:
        """Fence both button and keyboard generation during managed transitions."""

        if provider_id == "audio_cpp" and self._audio_cpp_runtime_status_failed:
            return "Runtime status must be checked before this action is available."
        error = super()._generation_readiness_error(
            provider_id,
            model_id,
            clone_action=clone_action,
        )
        if error is not None:
            return error
        if (
            getattr(self, "_audio_cpp_lifecycle_busy", None) is not None
            and provider_id == "audio_cpp"
        ):
            return "Wait for the audio.cpp lifecycle change to finish."
        return None

    def _sync_audio_cpp_runtime_visibility(self, provider_id: str) -> None:
        """Show the mounted card only for audio.cpp and invalidate old reads."""

        try:
            card = self.query_one("#audio-cpp-runtime-card", AudioCppRuntimeCard)
        except NoMatches:
            return
        self._audio_cpp_action_generation += 1
        selected = provider_id == "audio_cpp"
        card.display = selected
        if selected:
            self._audio_cpp_runtime_status_failed = False
            if self._audio_cpp_lifecycle_busy is not None:
                lifecycle_action = self._audio_cpp_lifecycle_action
                if lifecycle_action is not None:
                    self._render_audio_cpp_lifecycle_busy(lifecycle_action)
            else:
                self._audio_cpp_runtime_observation = None
                self._set_audio_cpp_primary_action(
                    project_audio_cpp_unknown_action(),
                    None,
                )
                loading_reason = (
                    "Runtime status is loading; Test Connection starts a fresh "
                    "audio.cpp check."
                )
                try:
                    self.query_one("#audio-cpp-runtime-status", Static).update(
                        "[CHECKING] Reading audio.cpp runtime status."
                    )
                    self.query_one(
                        "#audio-cpp-runtime-action-reason",
                        Static,
                    ).update(loading_reason)
                    primary = self.query_one("#tts-test-connection-btn", Button)
                    primary.label = "Test Connection"
                    primary.disabled = False
                    primary.tooltip = loading_reason
                    primary.refresh(layout=True)
                    for selector in (
                        "#audio-cpp-runtime-restart",
                        "#audio-cpp-runtime-shutdown",
                    ):
                        action = self.query_one(selector, Button)
                        action.disabled = True
                        action.tooltip = "Wait for the runtime status check to finish."
                except NoMatches:
                    pass
                self._sync_audio_cpp_probe_label(None)
            self._request_audio_cpp_runtime_observation()
            return
        self._clear_clone_setup(remove_component=True)
        self._audio_cpp_runtime_request_generation += 1
        self._audio_cpp_runtime_status_failed = False
        self._audio_cpp_primary_action = None
        self._audio_cpp_primary_observation = None
        self._sync_audio_cpp_probe_label(None)
        try:
            primary = self.query_one("#tts-test-connection-btn", Button)
            primary.label = "Test"
            primary.disabled = False
            primary.tooltip = "Test the selected provider connection"
            primary.refresh(layout=True)
        except NoMatches:
            pass

    @staticmethod
    def _audio_cpp_observation_identity(
        observation: AudioCppRuntimeObservation,
    ) -> tuple[int, int, int]:
        return (
            observation.saved_configuration_generation,
            observation.applied_configuration_generation,
            observation.process.process_generation,
        )

    def _audio_cpp_card_observation(
        self,
        observation: AudioCppRuntimeObservation,
    ) -> AudioCppRuntimeCardObservation:
        """Combine service truth with only matching pane-owned sample truth."""

        sample_state = self._audio_cpp_sample_state
        identity = self._audio_cpp_observation_identity(observation)
        if (
            self._audio_cpp_sample_identity is not None
            and self._audio_cpp_sample_identity != identity
        ):
            self._audio_cpp_sample_state = "not_attempted"
            self._audio_cpp_sample_identity = None
            sample_state = "not_attempted"
        return AudioCppRuntimeCardObservation(
            runtime=observation,
            sample_state=sample_state,
            clone_draft_state=self._clone_setup_draft_state(observation),
        )

    def _clone_setup_draft_state(
        self,
        observation: AudioCppRuntimeObservation,
    ) -> AudioCppCloneDraftState:
        """Return draft truth only for the exact currently projected recipe."""

        projection = observation.clone_setup
        if projection is None:
            return "missing"
        if self._clone_setup_context != self._clone_setup_observation_context(
            observation
        ):
            return "missing"
        task = self._clone_setup_validation_task
        if task is not None and not task.done():
            return "processing"
        if self._clone_setup_canonical is not None:
            return "ready"
        if self._clone_setup_error is not None:
            return "invalid"
        return "missing"

    @staticmethod
    def _clone_setup_observation_context(
        observation: AudioCppRuntimeObservation,
    ) -> tuple[str, str, int, int] | None:
        """Bind a private draft to the exact applied recipe generation."""

        projection = observation.clone_setup
        if projection is None:
            return None
        return (
            projection.model_id,
            projection.recipe_id,
            projection.recipe_revision,
            observation.applied_configuration_generation,
        )

    def _clear_clone_setup(self, *, remove_component: bool) -> None:
        """Fence staged private reference state without touching source files."""

        self._clone_setup_draft_revision += 1
        task = self._clone_setup_validation_task
        if task is not None and not task.done():
            task.cancel()
        self._clone_setup_validation_task = None
        self._clone_setup_source_path = None
        self._clone_setup_canonical = None
        self._clone_setup_error = None
        self._clone_setup_context = None
        if not remove_component or not self.is_mounted:
            return
        try:
            self.query_one("#speech-clone-setup-host", Vertical).remove_children()
        except NoMatches:
            pass

    def _sync_clone_setup_component(
        self,
        observation: AudioCppRuntimeObservation,
    ) -> None:
        """Mount only the exact selected recipe's focused clone setup."""

        projection = observation.clone_setup
        try:
            selected_model = self._current_select_value("#tts-model-select")
            host = self.query_one("#speech-clone-setup-host", Vertical)
        except NoMatches:
            return
        if projection is None or selected_model != projection.model_id:
            self._clear_clone_setup(remove_component=True)
            return
        setup_ready = bool(
            not observation.pending_configuration
            and observation.applied_mode == "managed"
            and observation.applied_managed_setup_source == "guided"
            and observation.process.state == "running"
            and observation.tts_capability == "available"
            and observation.catalog_fresh
        )
        context = self._clone_setup_observation_context(observation)
        assert context is not None
        try:
            setup = host.query_one(SpeechCloneSetup)
        except NoMatches:
            setup = None
        if self._clone_setup_context != context:
            if not setup_ready:
                self._clear_clone_setup(remove_component=True)
                return
            self._clear_clone_setup(remove_component=False)
            self._clone_setup_context = context
            if setup is not None:
                setup.apply_projection(projection)
        if setup is None:
            if not setup_ready:
                return
            setup = SpeechCloneSetup(
                projection,
                draft_state=self._clone_setup_draft_state(observation),
                id="speech-clone-setup",
            )
            host.mount(setup)
        setup.apply_draft_state(
            self._clone_setup_draft_state(observation),
            source_selected=self._clone_setup_source_path is not None,
            error_copy=self._clone_setup_error,
        )

    def _set_audio_cpp_primary_action(
        self,
        action: AudioCppRuntimeAction,
        observation: AudioCppRuntimeCardObservation | None,
    ) -> None:
        """Retain the exact immutable action represented by the visible control."""

        self._audio_cpp_primary_action = action
        self._audio_cpp_primary_observation = observation

    def _sync_audio_cpp_probe_label(
        self,
        observation: AudioCppRuntimeObservation | None,
    ) -> None:
        """Label catalog refresh as the explicit probe required for Unhealthy."""

        is_unhealthy_probe = bool(
            observation is not None
            and observation.applied_mode == "managed"
            and observation.process.state == "unhealthy"
            and not observation.pending_configuration
        )
        try:
            probe = self.query_one("#tts-refresh-catalog-btn", Button)
        except NoMatches:
            return
        probe.label = "Test Connection" if is_unhealthy_probe else "Refresh"
        probe.tooltip = (
            "Probe the unhealthy managed server without restarting it"
            if is_unhealthy_probe
            else "Re-read available models and voices"
        )
        probe.refresh(layout=True)

    def _poll_audio_cpp_runtime_observation(self) -> None:
        """Poll only the passive service seam while audio.cpp is selected."""

        if self._selected_runtime_provider() == "audio_cpp":
            self._request_audio_cpp_runtime_observation()

    def _request_audio_cpp_lifecycle(
        self,
        operation: AudioCppRuntimeOperation,
        *,
        projected_action: AudioCppRuntimeAction | None = None,
        projected_observation: AudioCppRuntimeCardObservation | None = None,
    ) -> None:
        """Accept one non-overlapping deliberate lifecycle action."""

        if self._selected_runtime_provider() != "audio_cpp":
            return
        if self._audio_cpp_lifecycle_busy is not None:
            self.app.notify(
                "An audio.cpp lifecycle change is already in progress.",
                severity="warning",
            )
            return
        action = projected_action
        card_observation = projected_observation
        observation = self._audio_cpp_runtime_observation
        if action is None:
            if observation is None:
                action = project_audio_cpp_unknown_action()
            else:
                card_observation = self._audio_cpp_card_observation(observation)
                projection = project_audio_cpp_runtime_card(card_observation)
                action = {
                    "test": project_audio_cpp_unknown_action(),
                    "sample": projection.primary_action,
                    "restart": projection.restart_action,
                    "shutdown": projection.shutdown_action,
                }[operation]
        if action.operation != operation:
            self.app.notify(
                "The visible audio.cpp action changed; review it and try again.",
                severity="warning",
            )
            return
        if not action.enabled:
            if action.disabled_reason:
                self.app.notify(action.disabled_reason, severity="warning")
            return
        action_generation = self._audio_cpp_action_generation
        self._audio_cpp_lifecycle_busy = operation
        self._audio_cpp_lifecycle_action = action
        self._render_audio_cpp_lifecycle_busy(action)
        self.app.run_worker(
            self._run_audio_cpp_lifecycle(
                action,
                card_observation,
                action_generation,
            ),
            name=f"speech_audio_cpp_{operation}",
            group="speech-audio-cpp-lifecycle",
            exclusive=True,
            exit_on_error=False,
        )

    def _render_audio_cpp_lifecycle_busy(
        self,
        action: AudioCppRuntimeAction,
    ) -> None:
        """Render immediate busy state without replacing focused controls."""

        busy_reason = "An audio.cpp operation is in progress."
        operation = action.operation
        observation = self._audio_cpp_runtime_observation
        if observation is not None and observation.applied_mode == "managed":
            busy_state = {
                "test": (
                    "starting"
                    if observation.process.state in {"stopped", "unavailable"}
                    else observation.process.state
                ),
                "sample": (
                    "starting"
                    if observation.process.state in {"stopped", "unavailable"}
                    else observation.process.state
                ),
                "restart": "draining",
                "shutdown": "draining",
            }[operation]
            busy_observation = replace(
                observation,
                process=replace(observation.process, state=busy_state),
            )
            try:
                self.query_one(
                    "#audio-cpp-runtime-card",
                    AudioCppRuntimeCard,
                ).apply_observation(busy_observation)
                primary_label = action.progress_label
            except NoMatches:
                primary_label = "Working…"
        else:
            primary_label = action.progress_label
            try:
                self.query_one("#audio-cpp-runtime-status", Static).update(
                    "[CHECKING] Testing the external audio.cpp connection."
                )
                self.query_one(
                    "#audio-cpp-runtime-action-reason",
                    Static,
                ).update(busy_reason)
            except NoMatches:
                pass

        try:
            self.query_one(
                "#audio-cpp-runtime-action-reason",
                Static,
            ).update(busy_reason)
        except NoMatches:
            pass

        for selector in (
            "#tts-test-connection-btn",
            "#tts-refresh-catalog-btn",
            "#tts-generate-btn",
            "#audio-cpp-runtime-restart",
            "#audio-cpp-runtime-shutdown",
        ):
            try:
                control = self.query_one(selector, Button)
                control.disabled = True
                control.tooltip = busy_reason
            except NoMatches:
                continue
        try:
            primary = self.query_one("#tts-test-connection-btn", Button)
            primary.label = primary_label
            primary.tooltip = busy_reason
            primary.refresh(layout=True)
        except NoMatches:
            pass

    async def _run_audio_cpp_lifecycle(
        self,
        action: AudioCppRuntimeAction,
        accepted_observation: AudioCppRuntimeCardObservation | None,
        action_generation: int,
    ) -> None:
        """Execute an accepted service operation and reconcile passive UI state."""

        operation = action.operation
        catalog = None
        completed_observation: AudioCppRuntimeObservation | None = None
        failure_copy: str | None = None
        try:
            service = self._tts_service
            if service is None:
                service = await self._tts_service_factory()
                self._tts_service = service
            if operation in {"test", "sample"}:
                catalog = await service.start_and_test_audio_cpp()
            elif operation == "restart":
                catalog = await service.restart_audio_cpp()
            else:
                await service.shutdown_audio_cpp()
            if operation == "sample":
                selected_model_id = self._audio_cpp_sample_model(accepted_observation)
                completed_observation = await service.audio_cpp_runtime_observation(
                    selected_model_id=selected_model_id
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            failure_copy = self._catalog_error_copy(error, "audio_cpp")
        finally:
            self._audio_cpp_lifecycle_busy = None
            self._audio_cpp_lifecycle_action = None

        if not self.is_mounted:
            return
        audio_cpp_selected = self._selected_runtime_provider() == "audio_cpp"
        action_is_current = bool(
            audio_cpp_selected
            and action_generation == self._audio_cpp_action_generation
        )
        if operation == "sample" and not action_is_current:
            self._stale_providers.add("audio_cpp")
            if audio_cpp_selected:
                self._request_audio_cpp_runtime_observation()
            return
        if failure_copy is None and audio_cpp_selected:
            try:
                self.query_one("#tts-refresh-catalog-btn", Button).disabled = False
            except NoMatches:
                pass
        if failure_copy is not None:
            if operation == "sample":
                self._audio_cpp_sample_state = "failed"
                self._audio_cpp_sample_identity = (
                    self._audio_cpp_observation_identity(accepted_observation.runtime)
                    if accepted_observation is not None
                    else None
                )
            self.app.notify(failure_copy, severity="error")
            if audio_cpp_selected:
                try:
                    self.query_one("#tts-refresh-catalog-btn", Button).disabled = False
                except NoMatches:
                    pass
                self._sync_generate_enabled()
        elif operation == "sample" and catalog is not None:
            expected_model = self._audio_cpp_sample_model(accepted_observation)
            sample_error = self._audio_cpp_sample_catalog_error(
                catalog,
                expected_model,
                accepted_observation,
                completed_observation,
            )
            if sample_error is not None:
                self._audio_cpp_sample_state = "failed"
                self._audio_cpp_sample_identity = (
                    self._audio_cpp_observation_identity(completed_observation)
                    if completed_observation is not None
                    else None
                )
                self.app.notify(sample_error, severity="error")
                self._stale_providers.add("audio_cpp")
                self._catalog_generation_allowed = False
                self._sync_generate_enabled()
                self.call_after_refresh(
                    self._focus_audio_cpp_action_target,
                    "#tts-test-connection-btn",
                )
            else:
                assert expected_model is not None
                assert completed_observation is not None
                self._audio_cpp_runtime_observation = completed_observation
                self._accept_audio_cpp_sample_catalog(catalog, expected_model)
                self._audio_cpp_sample_state = "generating"
                self._audio_cpp_sample_identity = self._audio_cpp_observation_identity(
                    completed_observation
                )
                self._audio_cpp_sample_focus_target = action.post_operation_focus
                card_observation = self._audio_cpp_card_observation(
                    completed_observation
                )
                try:
                    projection = self.query_one(
                        "#audio-cpp-runtime-card",
                        AudioCppRuntimeCard,
                    ).apply_observation(card_observation)
                    self._set_audio_cpp_primary_action(
                        projection.primary_action,
                        card_observation,
                    )
                    primary = self.query_one("#tts-test-connection-btn", Button)
                    primary.label = projection.primary_action.label
                    primary.disabled = not projection.primary_action.enabled
                    primary.tooltip = projection.primary_action.tooltip
                    primary.refresh(layout=True)
                except NoMatches:
                    pass
                text_area = self.query_one("#tts-text-input", TextArea)
                if not text_area.text.strip():
                    text_area.text = (
                        "Hello from Chatbook. This is a sample generated by your "
                        "local audio.cpp model."
                    )
                self._generate_tts()
                if self._generation_operation_id is None:
                    self._audio_cpp_sample_state = "failed"
                    self._audio_cpp_sample_focus_target = None
                    self._render_current_audio_cpp_observation()
                    self.call_after_refresh(
                        self._focus_audio_cpp_action_target,
                        "#tts-test-connection-btn",
                    )
        elif catalog is not None:
            if audio_cpp_selected:
                self._load_provider_catalog("audio_cpp", refresh=False)
            else:
                self._stale_providers.add("audio_cpp")
        else:
            self._stale_providers.add("audio_cpp")
            if audio_cpp_selected:
                self._catalog_generation_allowed = False
                self._sync_generate_enabled()
                try:
                    self.query_one("#tts-refresh-catalog-btn", Button).disabled = False
                except NoMatches:
                    pass
        observe_runtime = getattr(
            self._tts_service,
            "audio_cpp_runtime_observation",
            None,
        )
        if audio_cpp_selected and not callable(observe_runtime):
            self._render_audio_cpp_runtime_observation_failure()
        else:
            self._request_audio_cpp_runtime_observation()
        if operation != "sample" or failure_copy is not None:
            self.call_after_refresh(
                self._focus_audio_cpp_action_target,
                action.post_operation_focus
                if failure_copy is None
                else "#tts-test-connection-btn",
            )

    def _focus_audio_cpp_action_target(self, selector: str) -> None:
        """Focus a retained action target only while audio.cpp remains visible."""

        if self._selected_runtime_provider() != "audio_cpp":
            return
        try:
            self.query_one(selector).focus()
        except NoMatches:
            return

    @staticmethod
    def _audio_cpp_sample_model(
        observation: AudioCppRuntimeCardObservation | None,
    ) -> str | None:
        if observation is None:
            return None
        runtime = observation.runtime
        if runtime.process.state in {"running", "unhealthy", "draining"}:
            return runtime.applied_guided_default_model_id
        return runtime.saved_guided_default_model_id

    @staticmethod
    def _audio_cpp_sample_catalog_error(
        catalog: Any,
        expected_model: str | None,
        accepted: AudioCppRuntimeCardObservation | None,
        completed: AudioCppRuntimeObservation | None,
    ) -> str | None:
        if expected_model is None or accepted is None:
            return "The Guided sample selection is no longer available."
        if completed is None:
            return "The audio.cpp runtime result could not be verified."
        source = accepted.runtime
        if (
            completed.saved_configuration_generation
            != source.saved_configuration_generation
            or completed.applied_configuration_generation
            != source.saved_configuration_generation
            or completed.saved_guided_default_model_id != expected_model
            or completed.applied_guided_default_model_id != expected_model
            or completed.process.state != "running"
            or completed.tts_capability != "available"
            or not completed.catalog_fresh
        ):
            return "The saved Guided setup changed before the sample could run."
        if catalog.provider_id != "audio_cpp":
            return "The audio.cpp server returned an incompatible model catalog."
        if catalog.health.state != "available" or not catalog.health.fresh:
            return "The audio.cpp model catalog is not ready for a sample."
        model = next(
            (
                candidate
                for candidate in catalog.models
                if candidate.model_id == expected_model
            ),
            None,
        )
        if model is None or model.upstream_mode != "tts":
            return "The saved Guided default model was not exposed for text-to-speech."
        return None

    def _accept_audio_cpp_sample_catalog(
        self,
        catalog: Any,
        expected_model: str,
    ) -> None:
        """Publish the exact lifecycle catalog before using the normal generate path."""

        service = self._tts_service
        if service is None:
            return
        model = next(
            candidate
            for candidate in catalog.models
            if candidate.model_id == expected_model
        )
        observed_at = datetime.now(timezone.utc)
        self._catalogs["audio_cpp"] = catalog
        self._catalog_configuration_revisions["audio_cpp"] = (
            service.configuration_revision("audio_cpp")
        )
        self._catalog_observed_at["audio_cpp"] = observed_at
        self._catalog_runtime_observed_at["audio_cpp"] = observed_at
        self._catalog_checking_providers.discard("audio_cpp")
        self._catalog_unavailable_providers.discard("audio_cpp")
        self._stale_providers.discard("audio_cpp")
        self._pending_voice_selections.pop("audio_cpp", None)
        self._discovered_voices[("audio_cpp", expected_model)] = tuple(model.voices)
        snapshot = self._provider_control_snapshots.setdefault("audio_cpp", {})
        snapshot["model_id"] = expected_model
        snapshot["voice_id"] = None
        snapshot["response_format"] = "wav"
        snapshot["speed"] = 1.0
        self._apply_catalog("audio_cpp", catalog)

    def _render_current_audio_cpp_observation(self) -> None:
        observation = self._audio_cpp_runtime_observation
        if (
            observation is None
            or self._selected_runtime_provider() != "audio_cpp"
            or self._audio_cpp_lifecycle_busy is not None
        ):
            return
        card_observation = self._audio_cpp_card_observation(observation)
        try:
            projection = self.query_one(
                "#audio-cpp-runtime-card",
                AudioCppRuntimeCard,
            ).apply_observation(card_observation)
            primary = self.query_one("#tts-test-connection-btn", Button)
        except NoMatches:
            return
        self._set_audio_cpp_primary_action(
            projection.primary_action,
            card_observation,
        )
        primary.label = projection.primary_action.label
        primary.disabled = not projection.primary_action.enabled
        primary.tooltip = projection.primary_action.tooltip
        primary.refresh(layout=True)

    def _on_generation_result(self, artifact: Any) -> None:
        """Project Retry/ready state only for the guided sample now in flight."""

        self._handle_profile_generation_result(artifact)

        if self._audio_cpp_sample_state != "generating":
            return
        observation = self._audio_cpp_runtime_observation
        expected_model = (
            None if observation is None else observation.applied_guided_default_model_id
        )
        succeeded = bool(
            artifact is not None
            and artifact.provider_id == "audio_cpp"
            and artifact.model_id == expected_model
        )
        focus_target = self._audio_cpp_sample_focus_target
        self._audio_cpp_sample_focus_target = None
        self._audio_cpp_sample_state = "ready" if succeeded else "failed"
        self._render_current_audio_cpp_observation()
        if (
            succeeded
            and focus_target is not None
            and self._audio_cpp_lifecycle_busy is None
        ):
            self.call_after_refresh(
                self._focus_audio_cpp_action_target,
                focus_target,
            )

    def _request_audio_cpp_runtime_observation(self) -> None:
        """Reserve stale-result identity before starting a passive read worker."""

        if not self.is_mounted or self._selected_runtime_provider() != "audio_cpp":
            return
        self._audio_cpp_runtime_request_generation += 1
        request_generation = self._audio_cpp_runtime_request_generation
        try:
            selected_model = self._current_select_value("#tts-model-select")
        except NoMatches:
            selected_model = None
        selected_model_id = (
            selected_model if isinstance(selected_model, str) else None
        )
        self.run_worker(
            partial(
                self._observe_audio_cpp_runtime,
                request_generation,
                selected_model_id,
            ),
            group="speech-audio-cpp-runtime-observation",
            exclusive=True,
            exit_on_error=False,
        )

    async def _observe_audio_cpp_runtime(
        self,
        request_generation: int,
        selected_model_id: str | None,
    ) -> None:
        """Read and apply one coherent runtime observation without provider work."""

        try:
            service = self._tts_service
            if service is None:
                candidate = await self._tts_service_factory()
                if self._tts_service is None:
                    self._tts_service = candidate
                service = self._tts_service
            observe = getattr(service, "audio_cpp_runtime_observation", None)
            if not callable(observe):
                return
            observation = await observe(selected_model_id=selected_model_id)
            if type(observation) is not AudioCppRuntimeObservation:
                raise TypeError("Unexpected audio.cpp runtime observation")
        except asyncio.CancelledError:
            raise
        except Exception:
            if self._audio_cpp_runtime_result_is_current(request_generation):
                self._render_audio_cpp_runtime_observation_failure()
            return

        if not self._audio_cpp_runtime_result_is_current(request_generation):
            return
        try:
            current_model = self._current_select_value("#tts-model-select")
        except NoMatches:
            return
        if current_model != selected_model_id:
            return
        if not self._audio_cpp_runtime_observation_is_current_enough(observation):
            return
        self._audio_cpp_runtime_status_failed = False
        self._audio_cpp_runtime_observation = observation
        if self._audio_cpp_lifecycle_busy is not None:
            return
        self._sync_clone_setup_component(observation)
        card_observation = self._audio_cpp_card_observation(observation)
        try:
            card = self.query_one("#audio-cpp-runtime-card", AudioCppRuntimeCard)
            projection = card.apply_observation(card_observation)
            primary = self.query_one("#tts-test-connection-btn", Button)
        except NoMatches:
            return
        self._set_audio_cpp_primary_action(
            projection.primary_action,
            card_observation,
        )
        primary.label = projection.primary_action.label
        primary.disabled = not projection.primary_action.enabled
        primary.tooltip = projection.primary_action.tooltip
        primary.refresh(layout=True)
        self._sync_audio_cpp_probe_label(observation)

    def _audio_cpp_runtime_result_is_current(self, request_generation: int) -> bool:
        return (
            self.is_mounted
            and request_generation == self._audio_cpp_runtime_request_generation
            and self._selected_runtime_provider() == "audio_cpp"
        )

    def _audio_cpp_runtime_observation_is_current_enough(
        self,
        candidate: AudioCppRuntimeObservation,
    ) -> bool:
        """Reject a result older than any provider/process evidence on screen."""

        current = self._audio_cpp_runtime_observation
        if current is None:
            return True
        if (
            candidate.saved_configuration_generation
            < current.saved_configuration_generation
            or candidate.applied_configuration_generation
            < current.applied_configuration_generation
            or candidate.provider_configuration_revision
            < current.provider_configuration_revision
            or candidate.process.process_generation < current.process.process_generation
        ):
            return False
        return not (
            candidate.process.process_generation == current.process.process_generation
            and candidate.process.observation_version
            < current.process.observation_version
        )

    def _render_audio_cpp_runtime_observation_failure(self) -> None:
        """Show a fixed passive-read failure without exposing exception details."""

        if self._audio_cpp_lifecycle_busy is not None:
            return
        retry_reason = "Runtime status could not be read. Test the connection to retry."
        unchecked_reason = (
            "Runtime status must be checked before this action is available."
        )
        self._stale_providers.add("audio_cpp")
        self._catalog_generation_allowed = False
        self._audio_cpp_runtime_status_failed = True
        self._audio_cpp_runtime_observation = None
        self._set_audio_cpp_primary_action(
            project_audio_cpp_unknown_action(),
            None,
        )
        try:
            self.query_one("#audio-cpp-runtime-status", Static).update(
                "[UNAVAILABLE] Runtime status could not be read."
            )
            self.query_one("#audio-cpp-runtime-action-reason", Static).update(
                retry_reason
            )
            primary = self.query_one("#tts-test-connection-btn", Button)
            primary.label = "Test Connection"
            primary.disabled = False
            primary.tooltip = retry_reason
            primary.refresh(layout=True)
            refresh = self.query_one("#tts-refresh-catalog-btn", Button)
            refresh.disabled = False
            refresh.tooltip = retry_reason
            generate = self.query_one("#tts-generate-btn", Button)
            generate.disabled = True
            generate.tooltip = unchecked_reason
            for selector in (
                "#audio-cpp-runtime-restart",
                "#audio-cpp-runtime-shutdown",
            ):
                action = self.query_one(selector, Button)
                action.disabled = True
                action.tooltip = unchecked_reason
        except NoMatches:
            return

    def on_resize(self) -> None:
        """Stack the split when the pane is too narrow to hold two columns."""
        self._sync_split_layout()

    def on_mount(self) -> None:
        """Apply the split layout, then load the catalog.

        The catalog is what fills the axes. Without this call the pane
        renders correct, empty selects -- controls with nothing to choose --
        which looks like a finished screen and cannot synthesize anything.
        """
        self._sync_split_layout()
        self._refresh_provider_ids()
        # The language Select may still be composing its nested current-label
        # widget during the pane's Mount event. Mutate it after first refresh,
        # before the catalog callback queued below.
        self.call_after_refresh(self._settle_language_axis, self.provider)
        self._rehydrate_handler_state()
        self._sync_truthful_status_rows()
        self._sync_audio_cpp_runtime_visibility(self.provider)
        self.set_interval(
            _AUDIO_CPP_RUNTIME_POLL_SECONDS,
            self._poll_audio_cpp_runtime_observation,
        )
        # dev's mount sequence for the profile preset, kept in order: a pane
        # opened on an exact profile shows that profile immediately, before
        # discovery runs, rather than showing a catalog the user did not ask
        # for and then replacing it.
        if self._profile_preset is not None:
            self._schedule_profile_preset_mount()
        else:
            self._sync_profile_preview_status()
            self.call_after_refresh(
                self._load_provider_catalog,
                initialize=True,
            )
        if self._navigation_target is not None:
            self.call_after_refresh(self._focus_navigation_target)

    def _schedule_profile_preset_mount(self) -> None:
        """Schedule one callback owned by the current pane/preset generation."""

        self._profile_mount_generation += 1
        self.call_after_refresh(
            self._finish_profile_preset_mount,
            self._profile_mount_generation,
        )

    def invalidate_profile_mount_callbacks(self) -> None:
        """Fence callbacks queued by a superseded preset or pane mount."""

        self._profile_mount_generation += 1

    def _profile_mount_callback_is_current(self, generation: int) -> bool:
        return self.is_mounted and generation == self._profile_mount_generation

    def _finish_profile_preset_mount(self, generation: int) -> None:
        """Prime an exact preset after nested Select children are mounted."""

        if not self._profile_mount_callback_is_current(generation):
            return
        if self._profile_preset is not None:
            if not self._profile_mount_callback_is_current(generation):
                return
            self._prime_profile_preset_controls()
            if not self._profile_mount_callback_is_current(generation):
                return
            self.query_one("#tts-text-input", TextArea).focus()
        else:
            self._sync_profile_preview_status()
        if not self._profile_mount_callback_is_current(generation):
            return
        self._load_provider_catalog(initialize=True)

    def apply_profile_preset(
        self,
        preset: TTSPlaygroundSelectionPreset,
        *,
        context_token: UUID | None = None,
    ) -> None:
        """Apply one exact process-local preset to this mounted Playground.

        Args:
            preset: Exact profile selection to apply to the Playground.

        Raises:
            TypeError: If ``preset`` is not a selection preset.
        """

        if type(preset) is not TTSPlaygroundSelectionPreset:
            raise TypeError("preset must be TTSPlaygroundSelectionPreset")
        self.invalidate_profile_mount_callbacks()
        self._retire_profile_generation_context()
        self._retire_profile_test_authority()
        self.init_profile_state(preset, context_token)
        if self.is_mounted:
            self._schedule_profile_preset_mount()

    def _sync_split_layout(self) -> None:
        """Toggle the stacked class from the pane's measured width."""
        stacked = self.size.width < SPEECH_SPLIT_MIN_WIDTH
        if stacked == self._stacked:
            return
        self._stacked = stacked
        self.set_class(stacked, "speech-split-stacked")

    def compose(self) -> ComposeResult:
        """Compose the Playground.

        Order is the comparison loop: what you are about to do, the
        variables you are comparing, the text, then the results and the
        knobs you rarely touch.

        Returns:
            A ``ComposeResult`` yielding the title, action strip, axis
            chips, the Text/Result split, the collapsed provider parameters
            and the capability line.
        """
        yield Static("TTS Playground", classes="speech-pane-title", markup=False)
        # dev's profile-preview status line and save action. Both start
        # hidden and are revealed by `SpeechProfileMixin`, which queries
        # them by these exact ids -- so the rebuild mounts them unchanged.
        yield Static(
            "",
            id="tts-profile-preview-status",
            classes="speech-status-line hidden",
            markup=False,
        )
        yield Button(
            "Adopt as Studio Preferences",
            id="tts-adopt-studio-preferences-btn",
            classes="workbench-action hidden",
            compact=True,
            disabled=True,
        )

        yield SpeechActionStrip(PLAYGROUND_ACTIONS, id="speech-playground-actions")

        yield SpeechAxisRow(
            values=self.axis_values,
            defaults=self.axis_defaults,
            id="speech-axis-row",
        )

        with Horizontal(id="speech-split", classes="speech-split"):
            with Vertical(id="speech-text-pane", classes="speech-split-pane"):
                yield Static("Text", classes="speech-section-head")
                # Seeded with the first example, as legacy was. The screen
                # exists to let someone hear the options and pick one, so it
                # should be one keypress from audio -- not a blank box that
                # asks you to think of something to say first.
                editor = TextArea(
                    EXAMPLE_TEXTS[0],
                    id="tts-text-input",
                    classes="speech-input",
                    placeholder="Type or paste the text to synthesize...",
                )
                editor.show_line_numbers = False
                yield editor
                yield from self._compose_voice_source()

            with Vertical(id="speech-result-pane", classes="speech-split-pane"):
                yield from self._compose_player()

        yield from self._compose_generation_status()

        runtime_card = AudioCppRuntimeCard(id="audio-cpp-runtime-card")
        runtime_card.display = self.provider == "audio_cpp"
        yield runtime_card
        yield Vertical(id="speech-clone-setup-host", classes="speech-clone-setup-host")

        yield Static(
            "Loading TTS providers…",
            id="tts-provider-status",
            classes="speech-readiness-line",
            markup=False,
        )

        yield SpeechParamGroup(
            provider=self.provider,
            values=self._saved_studio_param_values(self.provider),
            id="speech-param-group",
        )

        yield self._compose_connection_details()
        yield self._compose_generation_log()

    def _compose_voice_source(self) -> ComposeResult:
        """Yield the clip picker for providers that synthesize from one.

        Mounted only for the providers that use it. Legacy mounted both rows
        for every provider and toggled a `hidden` class, which leaves a
        focusable control that does nothing -- the same reason the parameter
        group scopes knobs by provider.

        Builds each row by CONSTRUCTION rather than with the `with
        Horizontal(...)` compose idiom. That idiom only works inside
        `compose()`, and this is also called from
        `_reconcile_provider_regions` when the provider changes -- where it
        raised IndexError off Textual's empty compose stack.

        Returns:
            A ``ComposeResult`` yielding a reference-audio row, a voice
            upload row, or nothing.
        """
        if self.provider in REFERENCE_AUDIO_PROVIDERS:
            yield Horizontal(
                Static(
                    "No reference clip",
                    id="reference-audio-status",
                    classes="speech-source-status",
                    markup=False,
                ),
                SpeechActionStrip(
                    (
                        WorkbenchAction(
                            id="reference-audio-btn",
                            label="Choose clip",
                            tooltip="Pick a reference audio file",
                        ),
                        WorkbenchAction(
                            id="clear-reference-audio-btn",
                            label="Clear",
                            tooltip="Forget the reference clip",
                        ),
                    ),
                    id="speech-reference-actions",
                ),
                id="reference-audio-row",
                classes="speech-source-row",
            )

        if self.provider in VOICE_UPLOAD_PROVIDERS:
            yield Horizontal(
                Static(
                    "No voice sample",
                    id="higgs-voice-status",
                    classes="speech-source-status",
                    markup=False,
                ),
                SpeechActionStrip(
                    (
                        WorkbenchAction(
                            id="higgs-voice-upload-btn",
                            label="Upload voice",
                            tooltip="Pick a voice sample to clone",
                        ),
                        WorkbenchAction(
                            id="higgs-clear-voice-btn",
                            label="Clear",
                            tooltip="Forget the voice sample",
                        ),
                    ),
                    id="speech-voice-upload-actions",
                ),
                id="higgs-voice-upload-row",
                classes="speech-source-row",
            )

    def _compose_player(self) -> ComposeResult:
        """Yield the playback region for the take being auditioned.

        Returns:
            A ``ComposeResult`` yielding the player container and its
            transport actions.
        """
        with Vertical(id="audio-player-container", classes="speech-player"):
            yield Static("Current result", classes="speech-section-head", markup=False)
            yield Static(
                "No audio generated yet",
                id="audio-player-status",
                classes="speech-player-status",
                markup=False,
            )
            yield Static(
                "Generate speech to create a temporary result.",
                id="audio-result-lifecycle",
                classes="speech-result-lifecycle",
                markup=False,
            )
            yield Static(
                "No captured provider or model yet.",
                id="audio-result-provenance",
                classes="speech-result-provenance",
                markup=False,
            )
            with Horizontal(
                id="audio-player-transport",
                classes="speech-player-transport hidden",
            ):
                # `total` and `hidden` both matter. Without `total` a
                # ProgressBar renders its indeterminate pulse, so an idle
                # screen animates two bars forever; without `hidden` it is
                # on screen with nothing to report. Legacy carried both and
                # the rebuild dropped them -- visible only on a live run.
                yield ProgressBar(
                    id="audio-progress-bar",
                    total=100,
                    show_eta=False,
                    show_percentage=False,
                    classes="audio-progress hidden",
                )
                yield Static(
                    "",
                    id="audio-time-display",
                    classes="speech-player-time hidden",
                    markup=False,
                )
            yield SpeechActionStrip(PLAYER_ACTIONS, id="speech-result-actions")
            yield Button(
                "Save result as profile",
                id="audio-save-profile-btn",
                classes="workbench-action hidden",
                disabled=True,
                compact=True,
            )

    def _compose_generation_status(self) -> ComposeResult:
        """Yield the generation status line and progress.

        Returns:
            A ``ComposeResult`` yielding the status container.
        """
        # Hidden until a generation starts, as legacy was: otherwise the
        # placeholder ETA (`--% --:--:--`) sits on an idle screen.
        with Horizontal(
            id="generation-status-container",
            classes="speech-generation-status hidden",
        ):
            yield Static(
                "Ready to generate",
                id="generation-status-text",
                classes="speech-generation-text",
                markup=False,
            )
            yield ProgressBar(
                id="generation-progress",
                total=100,
                show_eta=True,
                show_percentage=True,
            )

    def _compose_connection_details(self) -> Collapsible:
        """Build secondary runtime and dependency facts behind disclosure."""

        details: list[Static] = [
            Static(
                self.capability_line,
                id="speech-capability-line",
                classes="speech-detail-line",
                markup=False,
            )
        ]
        details.extend(
            Static(
                row.copy,
                id=f"speech-status-{row.row_id}",
                classes="speech-detail-line",
                markup=False,
            )
            for row in self._truthful_status_rows()
        )
        details.append(
            Static(
                "audio.cpp returns one complete WAV and currently uses speed 1.0.",
                id="tts-audio-cpp-restrictions",
                classes="speech-detail-line hidden",
                markup=False,
            )
        )
        return Collapsible(
            *details,
            title="Connection details",
            id="speech-connection-details",
            collapsed=True,
        )

    @staticmethod
    def _compose_generation_log() -> Collapsible:
        """Build the diagnostic generation log as a collapsed disclosure."""

        return Collapsible(
            RichLog(id="tts-generation-log", classes="speech-log", markup=False),
            title="Generation log",
            id="speech-log-group",
            collapsed=True,
        )
