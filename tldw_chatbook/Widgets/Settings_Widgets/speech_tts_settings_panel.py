"""Global Speech & TTS Settings panel.

This is a bounded built-in surface.  Provider forms are deliberately explicit
so moving a field between global and Studio ownership is a reviewed code
change rather than a dynamic schema side effect.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import QueryError
from textual.events import DescendantFocus
from textual.message import Message
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Collapsible, Input, Select, Static, Switch
from textual.worker import NoActiveWorker, get_current_worker

from tldw_chatbook.Chat.console_voice_input import (
    DEFAULT_REALTIME_IDLE_TIMEOUT_MINUTES,
    DEFAULT_REALTIME_MODEL,
    DEFAULT_REALTIME_PROVIDER,
    handsfree_engine as _read_handsfree_engine,
    realtime_enabled as _read_realtime_enabled,
    realtime_model as _read_realtime_model,
    realtime_provider as _read_realtime_provider,
    realtime_voice as _read_realtime_voice,
)
from tldw_chatbook.config import get_cli_setting, save_settings_to_cli_config
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
    STTSSettingsSaveResult,
)
from tldw_chatbook.TTS.adapter_types import TTSNativeCapabilityObservation
from tldw_chatbook.Third_Party.textual_fspicker import (
    FileOpen,
    Filters,
    SelectDirectory,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Lab_Modules.lab_speech_status import (
    speech_local_dependency_availability,
)
from tldw_chatbook.UI.Speech.speech_runtime_status import (
    SpeechTTSRuntimeStatusStore,
    newest_speech_tts_status,
    project_speech_tts_status,
    speech_tts_navigation_context,
)
from tldw_chatbook.UI.Speech.speech_settings_contracts import (
    SpeechTTSConfigurationState,
    SpeechTTSDiagnosticCategory,
    SpeechTTSNavigationIntent,
    SpeechTTSNavigationTarget,
    SpeechTTSRuntimeState,
    SpeechTTSRuntimeStatus,
    SpeechTTSStatusFreshness,
)
from tldw_chatbook.UI.Screens.settings_speech_tts import (
    BUILT_IN_TTS_PROVIDER_ORDER,
    GLOBAL_TTS_PROVIDER_ENVIRONMENT_FIELDS,
    GLOBAL_TTS_PROVIDER_FIELD_IDS,
    TTS_PROVIDER_LABELS,
    AudioCppGlobalChoices,
    CredentialIntent,
    CredentialSource,
    GlobalSpeechTTSEffectiveSource,
    GlobalSpeechTTSCredentialMutation,
    GlobalSpeechTTSState,
    GlobalSpeechTTSValidationError,
    audio_cpp_transport_warning,
    build_credential_mutation,
    build_global_speech_tts_save_proposal,
    global_speech_tts_provider_configuration_changed,
    global_speech_tts_provider_configuration_state,
    project_audio_cpp_global_choices,
    restore_non_secret_defaults,
)

_PROVIDER_OPTIONS = [
    (TTS_PROVIDER_LABELS[provider_id], provider_id)
    for provider_id in BUILT_IN_TTS_PROVIDER_ORDER
]
_LANGUAGE_OPTIONS = [
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Chinese", "zh"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
]

LeaveChoice = Literal["save", "discard", "cancel"]
_GLOBAL_SPEECH_TTS_STACK_WIDTH = 104
_COLLAPSIBLE_TITLE_FOCUS_SUFFIX = "::collapsible-title"

_REALTIME_PROVIDER_OPTIONS = [("OpenAI", DEFAULT_REALTIME_PROVIDER)]
_REALTIME_HANDSFREE_ENGINE_OPTIONS = [
    ("Auto (realtime when enabled)", "auto"),
    ("Pipeline (record / transcribe / reply / speak)", "pipeline"),
    ("Realtime (forced)", "realtime"),
]


@dataclass
class _RealtimeSettingsDraft:
    """Local editable copy of the realtime engine's plain config keys.

    `realtime`/`dictation` are plain top-level config sections, not TTS
    provider adapters -- there is no `GlobalSpeechTTSState` provider entry
    for them and no TTS service adapter to reconfigure at runtime. This
    stays a self-contained sibling draft, persisted through the same atomic
    config writer other Settings surfaces use (`save_settings_to_cli_config`,
    which is `apply_settings_mutation_to_cli_config` underneath), never a
    second, bespoke config writer (TASK-2111).
    """

    enabled: bool
    provider: str
    model: str
    voice: str
    idle_timeout_minutes: str
    handsfree_engine: str

    def snapshot(self) -> tuple[bool, str, str, str, str, str]:
        return (
            self.enabled,
            self.provider,
            self.model,
            self.voice,
            self.idle_timeout_minutes,
            self.handsfree_engine,
        )


@dataclass(frozen=True)
class _RealtimeSavePayload:
    """One validated realtime/dictation mutation, ready for the shared writer."""

    section_values: dict[str, dict[str, Any]]
    delete_keys: dict[str, tuple[str, ...]]


def _read_realtime_settings_draft() -> _RealtimeSettingsDraft:
    """Read the realtime engine's live config into an editable draft."""

    return _RealtimeSettingsDraft(
        enabled=_read_realtime_enabled(),
        provider=_read_realtime_provider(),
        model=_read_realtime_model(),
        voice=_read_realtime_voice() or "",
        idle_timeout_minutes=str(
            get_cli_setting(
                "realtime",
                "idle_timeout_minutes",
                DEFAULT_REALTIME_IDLE_TIMEOUT_MINUTES,
            )
        ),
        handsfree_engine=_read_handsfree_engine(),
    )


class _CredentialEditorModal(ModalScreen[str | None]):
    """Empty masked Set/Replace editor; display placeholders are never input."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, provider_label: str, intent: CredentialIntent) -> None:
        super().__init__()
        self.provider_label = provider_label
        self.intent = intent

    def compose(self) -> ComposeResult:
        action = "Set" if self.intent is CredentialIntent.SET else "Replace"
        with Vertical(classes="settings-speech-credential-modal"):
            yield Static(
                f"{action} {self.provider_label} credential",
                classes="destination-section",
            )
            yield Static(
                "The editor starts empty. This value will be stored as a local "
                "config secret; an environment variable is safer and more portable.",
                classes="settings-detail-row",
                markup=False,
            )
            yield Input(
                id="settings-speech-credential-new-value",
                password=True,
                placeholder="Enter a new credential",
                tooltip=f"New {self.provider_label} credential",
            )
            with Horizontal(classes="settings-action-row"):
                yield Button("Cancel", id="settings-speech-credential-cancel")
                yield Button(
                    action,
                    id="settings-speech-credential-confirm",
                    variant="primary",
                )

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#settings-speech-credential-cancel")
    def handle_cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#settings-speech-credential-confirm")
    def handle_confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(
            self.query_one("#settings-speech-credential-new-value", Input).value
        )


class _CredentialClearModal(ModalScreen[bool]):
    """Explicit confirmation for deleting only a saved local credential."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, provider_label: str) -> None:
        super().__init__()
        self.provider_label = provider_label

    def compose(self) -> ComposeResult:
        with Vertical(classes="settings-speech-credential-modal"):
            yield Static(
                f"Clear saved {self.provider_label} credential?",
                classes="destination-section",
            )
            yield Static(
                "This removes only the local-config value. It cannot change a "
                "process environment variable.",
                classes="settings-detail-row",
                markup=False,
            )
            with Horizontal(classes="settings-action-row"):
                yield Button("Cancel", id="settings-speech-credential-clear-cancel")
                yield Button(
                    "Clear saved credential",
                    id="settings-speech-credential-clear-confirm",
                    variant="warning",
                )

    def action_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#settings-speech-credential-clear-cancel")
    def handle_cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(False)

    @on(Button.Pressed, "#settings-speech-credential-clear-confirm")
    def handle_confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(True)


class _GlobalSpeechTTSLeaveModal(ModalScreen[LeaveChoice]):
    """Protect a dirty global Speech/TTS draft before changing its owner."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def compose(self) -> ComposeResult:
        with Vertical(
            id="global-speech-tts-leave-modal",
            classes="settings-rag-profile-modal",
        ):
            yield Static(
                "Unsaved global Speech & TTS settings",
                classes="destination-section",
            )
            yield Static(
                "Save these application-wide changes before continuing, or "
                "discard them?",
                classes="settings-detail-row",
                markup=False,
            )
            with Horizontal(classes="settings-action-row"):
                yield Button("Cancel", id="global-speech-tts-leave-cancel")
                yield Button(
                    "Discard and continue",
                    id="global-speech-tts-leave-discard",
                )
                yield Button(
                    "Save and continue",
                    id="global-speech-tts-leave-save",
                    variant="primary",
                )

    def action_cancel(self) -> None:
        self.dismiss("cancel")

    @on(Button.Pressed, "#global-speech-tts-leave-cancel")
    def handle_cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss("cancel")

    @on(Button.Pressed, "#global-speech-tts-leave-discard")
    def handle_discard(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss("discard")

    @on(Button.Pressed, "#global-speech-tts-leave-save")
    def handle_save(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss("save")


class SpeechTTSSettingsPanel(Vertical):
    """Edit application-wide Speech/TTS defaults and one provider at a time."""

    class DraftModified(Message):
        """Report whether the panel has an unsaved non-secret draft."""

        def __init__(
            self,
            is_modified: bool,
            state: GlobalSpeechTTSState,
            original_state: GlobalSpeechTTSState,
            configure_provider: str,
        ) -> None:
            self.is_modified = is_modified
            self.state = deepcopy(state)
            self.original_state = deepcopy(original_state)
            self.configure_provider = configure_provider
            super().__init__()

    def __init__(
        self,
        *,
        state: GlobalSpeechTTSState,
        original_state: GlobalSpeechTTSState | None = None,
        configure_provider: str | None = None,
        audio_cpp_observation: TTSNativeCapabilityObservation | None = None,
        audio_cpp_configuration_revision: int | None = None,
        audio_cpp_saved_configuration_revision: int | None = None,
        audio_cpp_applied_configuration_revision: int | None = None,
        provider_configuration_revisions: dict[str, int] | None = None,
        provider_runtime_revisions: dict[str, int] | None = None,
        provider_applied_configuration_revisions: dict[str, int] | None = None,
        runtime_status_store: SpeechTTSRuntimeStatusStore | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.original_state = deepcopy(original_state or state)
        self.state = deepcopy(state)
        self._local_dependencies = speech_local_dependency_availability(refresh=True)
        self._audio_cpp_observation = audio_cpp_observation
        self._runtime_status_store = (
            runtime_status_store or SpeechTTSRuntimeStatusStore()
        )
        self._audio_cpp_runtime_revision = audio_cpp_configuration_revision
        saved_revision = (
            audio_cpp_configuration_revision
            if audio_cpp_saved_configuration_revision is None
            else audio_cpp_saved_configuration_revision
        )
        applied_revision = (
            saved_revision
            if audio_cpp_applied_configuration_revision is None
            else audio_cpp_applied_configuration_revision
        )
        self._provider_configuration_revisions = dict(
            provider_configuration_revisions or {}
        )
        self._provider_runtime_revisions = dict(provider_runtime_revisions or {})
        self._provider_applied_configuration_revisions = dict(
            provider_applied_configuration_revisions or {}
        )
        if saved_revision is not None:
            self._provider_configuration_revisions["audio_cpp"] = saved_revision
        if audio_cpp_configuration_revision is not None:
            self._provider_runtime_revisions["audio_cpp"] = (
                audio_cpp_configuration_revision
            )
        if applied_revision is not None:
            self._provider_applied_configuration_revisions["audio_cpp"] = (
                applied_revision
            )
        self._runtime_statuses: dict[str, SpeechTTSRuntimeStatus] = {}
        self._provider_runtime_request_ids: dict[str, int] = {}
        self._provider_runtime_request_observed_at: dict[tuple[str, int], datetime] = {}
        self.configure_provider = (
            configure_provider
            if configure_provider in BUILT_IN_TTS_PROVIDER_ORDER
            else state.defaults.provider_id
            if state.defaults.provider_id in BUILT_IN_TTS_PROVIDER_ORDER
            else "audio_cpp"
        )
        self.result_text = "No global Speech & TTS changes saved this session."
        self._syncing = False
        self._field_errors: dict[tuple[str, str], str] = {}
        self._next_request_id = 1
        self._latest_request_id: int | None = None
        self._pending_credential_mutation: GlobalSpeechTTSCredentialMutation | None = (
            None
        )
        self._pending_saved_defaults = None
        self._pending_saved_provider_id: str | None = None
        self._pending_saved_provider_values: dict[str, object] | None = None
        self._pending_focus_control_id: str | None = None
        self._pending_displaced_focus_control_id: str | None = None
        self._pending_focus_moved_after_displacement = False
        self._leave_save_waiters: dict[int, asyncio.Future[bool]] = {}
        self._last_focused_control_id: str | None = None
        self._realtime_original = _read_realtime_settings_draft()
        self._realtime_draft = replace(self._realtime_original)

    def on_mount(self) -> None:
        """Apply responsive layout without performing provider work."""

        self._sync_responsive_layout()

    def on_resize(self) -> None:
        """Keep labels and actions inside the available Settings detail width."""

        self._sync_responsive_layout()

    def _sync_responsive_layout(self) -> None:
        """Stack fields and actions when one row cannot fit four cells."""

        self.set_class(
            self.size.width < _GLOBAL_SPEECH_TTS_STACK_WIDTH,
            "settings-speech-stacked",
        )

    @staticmethod
    def _field_dom_id(provider_id: str, field_id: str) -> str:
        return f"settings-speech-{provider_id}-{field_id.replace('_', '-')}"

    @staticmethod
    def _row(
        label: str,
        control: Input | Select | Switch | Static | Vertical | Horizontal,
        *,
        classes: str = "",
        error: Static | None = None,
    ) -> Horizontal:
        if isinstance(control, (Input, Select, Switch)) and not control.tooltip:
            control.tooltip = label
        children: list[Any] = [Static(label, classes="settings-input-label")]
        if error is not None:
            children.append(
                Vertical(
                    control,
                    error,
                    classes="settings-speech-control-stack",
                )
            )
        else:
            children.append(control)
        return Horizontal(
            *children,
            classes=(f"settings-input-row settings-speech-input-row {classes}").strip(),
        )

    def _error(self, provider_id: str, field_id: str) -> Static:
        message = self._field_errors.get((provider_id, field_id), "")
        return Static(
            message,
            id=f"{self._field_dom_id(provider_id, field_id)}-error",
            classes=(
                "settings-status-row settings-speech-field-error"
                + (" settings-speech-field-error-visible" if message else "")
            ),
            markup=False,
        )

    def _default_error(self, field_id: str, control_id: str) -> Static:
        message = self._field_errors.get(("defaults", field_id), "")
        return Static(
            message,
            id=f"{control_id}-error",
            classes=(
                "settings-status-row settings-speech-field-error"
                + (" settings-speech-field-error-visible" if message else "")
            ),
            markup=False,
        )

    def _input(
        self,
        provider_id: str,
        field_id: str,
        label: str,
        *,
        placeholder: str = "",
        disabled: bool = False,
    ) -> Horizontal:
        return self._row(
            label,
            Input(
                value=str(self.state.providers[provider_id].get(field_id, "")),
                id=self._field_dom_id(provider_id, field_id),
                placeholder=placeholder,
                disabled=disabled,
                classes=(
                    "settings-compact-input settings-speech-field "
                    "settings-speech-draft-field"
                ),
            ),
            error=self._error(provider_id, field_id),
        )

    def _select(
        self,
        provider_id: str,
        field_id: str,
        label: str,
        options: list[tuple[str, str]],
    ) -> Horizontal:
        return self._row(
            label,
            Select(
                options,
                value=self.state.providers[provider_id].get(field_id),
                id=self._field_dom_id(provider_id, field_id),
                allow_blank=False,
                compact=True,
                classes=(
                    "settings-compact-select settings-speech-field "
                    "settings-speech-draft-field"
                ),
            ),
            classes="settings-select-row",
            error=self._error(provider_id, field_id),
        )

    def _switch(
        self,
        provider_id: str,
        field_id: str,
        label: str,
    ) -> Horizontal:
        return self._row(
            label,
            Switch(
                value=bool(self.state.providers[provider_id].get(field_id)),
                id=self._field_dom_id(provider_id, field_id),
                classes="settings-speech-field settings-speech-draft-field",
            ),
            error=self._error(provider_id, field_id),
        )

    def _path(
        self,
        provider_id: str,
        field_id: str,
        label: str,
        *,
        placeholder: str,
    ) -> Horizontal:
        dom_id = self._field_dom_id(provider_id, field_id)
        environment_variable = GLOBAL_TTS_PROVIDER_ENVIRONMENT_FIELDS.get(
            provider_id, {}
        ).get(field_id)
        environment_owned = (
            self.state.provider_field_sources.get(provider_id, {}).get(field_id)
            is GlobalSpeechTTSEffectiveSource.ENVIRONMENT
        )
        controls: list[Horizontal | Static] = [
            Horizontal(
                Input(
                    value=str(self.state.providers[provider_id].get(field_id, "")),
                    id=dom_id,
                    placeholder=placeholder,
                    tooltip=label,
                    disabled=environment_owned,
                    classes=(
                        "settings-compact-input settings-speech-field "
                        "settings-speech-draft-field"
                    ),
                ),
                Button(
                    "Browse…",
                    id=f"{dom_id}-browse",
                    compact=True,
                    disabled=environment_owned,
                    classes="settings-speech-path-picker",
                    tooltip=(
                        f"Choose {label.lower()} without initializing a provider."
                    ),
                ),
                classes="settings-speech-path-control",
            )
        ]
        if environment_owned and environment_variable is not None:
            controls.append(
                Static(
                    "Effective source: Environment "
                    f"({environment_variable}, read-only). The displayed value is "
                    "the saved local fallback and does not override the environment.",
                    classes="settings-detail-row",
                    markup=False,
                )
            )
        controls.append(self._error(provider_id, field_id))
        return self._row(
            label,
            Vertical(
                *controls,
                classes="settings-speech-control-stack",
            ),
        )

    def _credential(self, provider_id: str) -> Vertical:
        state = self.state.credentials[provider_id]
        action_label = "Replace credential" if state.local_saved else "Set credential"
        source_copy = f"Effective source: {state.source.value}"
        if state.source.value == "Environment":
            source_copy += f" ({state.environment_variable}, read-only)"
        if state.local_shadowed:
            source_copy += "; saved local fallback is shadowed by the environment"
        controls: list[Button | Static] = [
            Static(source_copy, classes="settings-status-row", markup=False),
            Static(
                "Local credentials are stored as local config secrets. "
                f"Using {state.environment_variable} is the safer portable option.",
                classes="settings-detail-row",
                markup=False,
            ),
            Button(
                action_label,
                id=f"settings-speech-{provider_id}-credential-edit",
                compact=True,
                disabled=self._latest_request_id is not None,
            ),
        ]
        if state.local_saved:
            controls.append(
                Button(
                    "Clear saved credential",
                    id=f"settings-speech-{provider_id}-credential-clear",
                    compact=True,
                    variant="warning",
                    disabled=self._latest_request_id is not None,
                )
            )
        return Vertical(
            *controls,
            id=f"settings-speech-{provider_id}-credential",
            classes="settings-speech-credential",
        )

    def _audio_cpp_choices(self) -> AudioCppGlobalChoices:
        return project_audio_cpp_global_choices(
            self.state.defaults,
            observation=self._audio_cpp_observation,
            current_configuration_revision=self._audio_cpp_runtime_revision,
            saved_configuration_revision=(
                self._provider_configuration_revisions.get("audio_cpp")
            ),
            applied_configuration_revision=(
                self._provider_applied_configuration_revisions.get("audio_cpp")
            ),
        )

    @staticmethod
    def _safe_exact_options(
        options: tuple[tuple[str, str], ...],
    ) -> list[tuple[Text, str]]:
        """Render provider-supplied labels as text, never Textual markup."""
        return [(Text(label, no_wrap=True), value) for label, value in options]

    def _audio_cpp_observation_copy(
        self,
        choices: AudioCppGlobalChoices,
    ) -> str:
        if choices.observed_at is None:
            return "No accepted audio.cpp catalog has been observed in this session."
        observed = choices.observed_at.astimezone(timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
        configuration_revision = (
            str(choices.configuration_revision)
            if choices.configuration_revision is not None
            else "unknown"
        )
        catalog_revision = (
            str(choices.catalog_revision)
            if choices.catalog_revision is not None
            else "none"
        )
        return (
            f"Observed {observed} | configuration revision "
            f"{configuration_revision} | catalog revision {catalog_revision}."
        )

    def _audio_cpp_draft_attribution_copy(self, value: object | None = None) -> str:
        candidate = (
            self.state.providers["audio_cpp"].get("base_url")
            if value is None
            else value
        )
        saved = self.original_state.providers["audio_cpp"].get("base_url")
        if candidate != saved:
            return (
                "Catalog and runtime observations apply to the saved server "
                "configuration, not this unsaved Server URL draft."
            )
        return (
            "Catalog and runtime observations apply to the saved server configuration."
        )

    def _configuration_state(self) -> SpeechTTSConfigurationState:
        """Derive only the selected provider's local configuration state."""

        try:
            changed = global_speech_tts_provider_configuration_changed(
                self.original_state,
                self.state,
                provider_id=self.configure_provider,
            )
        except GlobalSpeechTTSValidationError as error:
            if "required" in str(error).lower():
                return SpeechTTSConfigurationState.INCOMPLETE
            return SpeechTTSConfigurationState.INVALID
        saved_state = global_speech_tts_provider_configuration_state(
            self.state,
            provider_id=self.configure_provider,
        )
        if saved_state in {
            SpeechTTSConfigurationState.INCOMPLETE,
            SpeechTTSConfigurationState.INVALID,
        }:
            return saved_state
        if changed:
            return SpeechTTSConfigurationState.UNSAVED
        return saved_state

    def _model_id_for_provider(self, provider_id: str) -> str | None:
        defaults = self.state.defaults
        if (
            defaults.provider_id == provider_id
            and defaults.model_mode == "exact"
            and defaults.model_id
        ):
            return defaults.model_id
        return None

    def _provider_connection_draft_dirty(self, provider_id: str) -> bool:
        try:
            return global_speech_tts_provider_configuration_changed(
                self.original_state,
                self.state,
                provider_id=provider_id,
            )
        except GlobalSpeechTTSValidationError:
            return (
                self.state.providers[provider_id]
                != self.original_state.providers[provider_id]
            )
        return False

    def _defaults_configuration_state(self) -> SpeechTTSConfigurationState:
        """Return the global-selection draft state independently of setup."""

        try:
            changed = (
                self.state.defaults.snapshot()
                != self.original_state.defaults.snapshot()
            )
        except GlobalSpeechTTSValidationError as error:
            if "required" in str(error).lower():
                return SpeechTTSConfigurationState.INCOMPLETE
            return SpeechTTSConfigurationState.INVALID
        if changed:
            return SpeechTTSConfigurationState.UNSAVED
        if self.state.defaults_source is GlobalSpeechTTSEffectiveSource.DEFAULT:
            return SpeechTTSConfigurationState.DEFAULT
        if self.state.defaults_source is GlobalSpeechTTSEffectiveSource.INHERITED:
            return SpeechTTSConfigurationState.INHERITED
        return SpeechTTSConfigurationState.SAVED

    def _draft_impact_copy(self) -> str:
        """Describe only the global owners an unsaved draft would change."""

        defaults_dirty = self._defaults_configuration_state() in {
            SpeechTTSConfigurationState.UNSAVED,
            SpeechTTSConfigurationState.INCOMPLETE,
            SpeechTTSConfigurationState.INVALID,
        }
        provider_dirty = self._provider_connection_draft_dirty(self.configure_provider)
        if defaults_dirty and provider_dirty:
            impact = "global selection defaults and selected provider setup"
        elif defaults_dirty:
            impact = "global selection defaults"
        elif provider_dirty:
            impact = "selected provider setup"
        else:
            impact = "none"
        return f"Unsaved draft impact: {impact}. Studio preferences are separate."

    def _refresh_configuration_metadata(self) -> None:
        """Refresh source and draft-impact copy without provider work."""

        updates = {
            "#settings-speech-default-source": (
                "Global default selection: "
                f"{self._defaults_configuration_state().value} — effective source "
                f"{self.state.defaults_source.value}."
            ),
            "#settings-speech-provider-source": (
                "Selected provider setup source: "
                f"{self.state.provider_sources[self.configure_provider].value}."
            ),
            "#settings-speech-draft-impact": self._draft_impact_copy(),
        }
        for selector, copy in updates.items():
            try:
                self.query_one(selector, Static).update(copy)
            except QueryError:
                continue

    def _status_projection(self):
        provider_id = self.configure_provider
        runtime_status = self._latest_status(
            self._runtime_statuses.get(provider_id),
            self._runtime_status_store.runtime_status(provider_id),
            catalog_axis=False,
        )
        projection = project_speech_tts_status(
            provider_id=provider_id,
            configuration_state=self._configuration_state(),
            current_configuration_revision=self._provider_configuration_revisions.get(
                provider_id
            ),
            current_runtime_revision=self._provider_runtime_revisions.get(provider_id),
            applied_configuration_revision=(
                self._provider_applied_configuration_revisions.get(provider_id)
            ),
            model_id=self._model_id_for_provider(provider_id),
            observation=(
                self._audio_cpp_observation if provider_id == "audio_cpp" else None
            ),
            local_dependencies=self._local_dependencies,
            runtime_status=runtime_status,
        )
        shared_catalog = self._runtime_status_store.catalog_status(
            provider_id,
            self._model_id_for_provider(provider_id),
        )
        catalog_status = self._latest_status(
            projection.catalog_status,
            shared_catalog,
            catalog_axis=True,
        )
        if catalog_status is not None:
            projection = project_speech_tts_status(
                provider_id=provider_id,
                configuration_state=self._configuration_state(),
                current_configuration_revision=(
                    self._provider_configuration_revisions.get(provider_id)
                ),
                current_runtime_revision=self._provider_runtime_revisions.get(
                    provider_id
                ),
                applied_configuration_revision=(
                    self._provider_applied_configuration_revisions.get(provider_id)
                ),
                model_id=self._model_id_for_provider(provider_id),
                observation=(
                    self._audio_cpp_observation if provider_id == "audio_cpp" else None
                ),
                local_dependencies=self._local_dependencies,
                runtime_status=runtime_status,
                catalog_status=catalog_status,
            )
        if projection.runtime_status is not None:
            self._runtime_status_store.publish_runtime(projection.runtime_status)
        if projection.catalog_status is not None:
            self._runtime_status_store.publish_catalog(projection.catalog_status)
        return projection

    @staticmethod
    def _latest_status(
        first: SpeechTTSRuntimeStatus | None,
        second: SpeechTTSRuntimeStatus | None,
        *,
        catalog_axis: bool,
    ) -> SpeechTTSRuntimeStatus | None:
        """Return the newest revision-bound status without merging payloads."""

        return newest_speech_tts_status(
            first,
            second,
            catalog_axis=catalog_axis,
        )

    def _refresh_status_rows(self) -> None:
        """Refresh mounted status rows without contacting any provider."""

        projection = self._status_projection()
        dirty_connection = self._provider_connection_draft_dirty(
            self.configure_provider
        )
        for row in projection.rows(dirty_draft=dirty_connection):
            try:
                self.query_one(
                    f"#settings-speech-status-{row.row_id}",
                    Static,
                ).update(row.copy)
            except QueryError:
                continue
        self._refresh_configuration_metadata()

    def compose(self) -> ComposeResult:
        yield Static(
            "Speech & TTS", classes="destination-section settings-column-title"
        )
        with Vertical(id="settings-speech-scope-banner", classes="settings-focus-card"):
            yield Static(
                "You are editing application-wide Speech & TTS defaults. "
                "The Speech Studio can keep separate Studio preferences without "
                "changing these values.",
                classes="settings-detail-row",
                markup=False,
            )
            yield Button(
                "Open Speech Lab",
                id="settings-speech-open-lab",
                compact=True,
                tooltip="Open Speech Lab without testing or refreshing a provider.",
            )

        defaults = self.state.defaults
        audio_cpp_selected = defaults.provider_id == "audio_cpp"
        audio_cpp_choices = self._audio_cpp_choices() if audio_cpp_selected else None
        if audio_cpp_choices is not None:
            model_policy_options = [("First available", "first_available")]
            if audio_cpp_choices.model.exact_allowed:
                model_policy_options.append(("Exact", "exact"))
            voice_policy_options = [("Server default", "server_default")]
            if audio_cpp_choices.voice.exact_allowed:
                voice_policy_options.append(("Exact", "exact"))
        else:
            model_policy_options = [
                ("Exact", "exact"),
                ("First available", "first_available"),
            ]
            voice_policy_options = [
                ("Exact", "exact"),
                ("Server default", "server_default"),
            ]
        with Vertical(
            id="settings-speech-global-defaults", classes="settings-focus-card"
        ):
            yield Static("Global defaults", classes="destination-section")
            yield Static(
                "Global default selection: "
                f"{self._defaults_configuration_state().value} — effective source "
                f"{self.state.defaults_source.value}.",
                id="settings-speech-default-source",
                classes="settings-status-row",
                markup=False,
            )
            yield self._row(
                "Default TTS Provider",
                Select(
                    _PROVIDER_OPTIONS,
                    value=defaults.provider_id,
                    id="settings-speech-default-provider",
                    allow_blank=False,
                    compact=True,
                    classes="settings-compact-select settings-speech-draft-field",
                ),
                classes="settings-select-row",
                error=self._default_error(
                    "provider_id",
                    "settings-speech-default-provider",
                ),
            )
            yield self._row(
                "Model policy",
                Select(
                    model_policy_options,
                    value=defaults.model_mode,
                    id="settings-speech-model-policy",
                    allow_blank=False,
                    compact=True,
                    classes="settings-compact-select settings-speech-draft-field",
                ),
                classes="settings-select-row",
                error=self._default_error(
                    "model_mode",
                    "settings-speech-model-policy",
                ),
            )
            if audio_cpp_choices is not None:
                yield self._row(
                    "Model value",
                    Select(
                        self._safe_exact_options(audio_cpp_choices.model.options),
                        value=(
                            defaults.model_id
                            if defaults.model_mode == "exact" and defaults.model_id
                            else Select.NULL
                        ),
                        id="settings-speech-model-value",
                        allow_blank=defaults.model_mode != "exact",
                        compact=True,
                        disabled=defaults.model_mode != "exact",
                        classes=("settings-compact-select settings-speech-draft-field"),
                    ),
                    classes="settings-select-row",
                    error=self._default_error(
                        "default_model",
                        "settings-speech-model-value",
                    ),
                )
            else:
                yield self._row(
                    "Model value",
                    Input(
                        value=defaults.model_id or "",
                        id="settings-speech-model-value",
                        disabled=defaults.model_mode != "exact",
                        placeholder="Exact model ID",
                        classes="settings-compact-input settings-speech-draft-field",
                    ),
                    error=self._default_error(
                        "default_model",
                        "settings-speech-model-value",
                    ),
                )
            yield self._row(
                "Voice policy",
                Select(
                    voice_policy_options,
                    value=defaults.voice_mode,
                    id="settings-speech-voice-policy",
                    allow_blank=False,
                    compact=True,
                    classes="settings-compact-select settings-speech-draft-field",
                ),
                classes="settings-select-row",
                error=self._default_error(
                    "voice_mode",
                    "settings-speech-voice-policy",
                ),
            )
            if audio_cpp_choices is not None:
                yield self._row(
                    "Voice value",
                    Select(
                        self._safe_exact_options(audio_cpp_choices.voice.options),
                        value=(
                            defaults.voice_id
                            if defaults.voice_mode == "exact" and defaults.voice_id
                            else Select.NULL
                        ),
                        id="settings-speech-voice-value",
                        allow_blank=defaults.voice_mode != "exact",
                        compact=True,
                        disabled=defaults.voice_mode != "exact",
                        classes=("settings-compact-select settings-speech-draft-field"),
                    ),
                    classes="settings-select-row",
                    error=self._default_error(
                        "default_voice",
                        "settings-speech-voice-value",
                    ),
                )
                yield Static(
                    f"Model: {audio_cpp_choices.model.state.value} | "
                    f"Voice: {audio_cpp_choices.voice.state.value}",
                    id="settings-speech-audio-cpp-choice-status",
                    classes="settings-status-row",
                    markup=False,
                )
                yield Static(
                    self._audio_cpp_observation_copy(audio_cpp_choices),
                    id="settings-speech-audio-cpp-observation-provenance",
                    classes="settings-detail-row",
                    markup=False,
                )
                yield Static(
                    "Settings reuses accepted in-memory observations only. Open "
                    "Speech Lab to test the server or refresh models and voices.",
                    classes="settings-detail-row",
                    markup=False,
                )
            else:
                yield self._row(
                    "Voice value",
                    Input(
                        value=defaults.voice_id or "",
                        id="settings-speech-voice-value",
                        disabled=defaults.voice_mode != "exact",
                        placeholder="Exact voice ID",
                        classes="settings-compact-input settings-speech-draft-field",
                    ),
                    error=self._default_error(
                        "default_voice",
                        "settings-speech-voice-value",
                    ),
                )
            yield self._row(
                "Output format",
                Select(
                    [
                        ("MP3", "mp3"),
                        ("Opus", "opus"),
                        ("AAC", "aac"),
                        ("FLAC", "flac"),
                        ("WAV", "wav"),
                    ],
                    value=defaults.response_format,
                    id="settings-speech-output-format",
                    allow_blank=False,
                    compact=True,
                    disabled=audio_cpp_selected,
                    classes="settings-compact-select settings-speech-draft-field",
                ),
                classes="settings-select-row",
                error=self._default_error(
                    "response_format",
                    "settings-speech-output-format",
                ),
            )
            yield self._row(
                "Speed",
                Input(
                    value=str(defaults.speed),
                    id="settings-speech-speed",
                    disabled=audio_cpp_selected,
                    placeholder="0.25 - 4.0",
                    classes="settings-compact-input settings-speech-draft-field",
                ),
                error=self._default_error(
                    "default_speed",
                    "settings-speech-speed",
                ),
            )
            yield Static(
                "audio.cpp requires WAV output and speed 1.0."
                if audio_cpp_selected
                else "Provider capability constraints are validated before Save.",
                id="settings-speech-default-constraints",
                classes="settings-status-row",
                markup=False,
            )

        with Vertical(
            id="settings-speech-provider-setup", classes="settings-focus-card"
        ):
            yield Static("Provider setup", classes="destination-section")
            yield Static(
                "Configure Provider does not change the Default TTS Provider.",
                classes="settings-detail-row",
                markup=False,
            )
            yield self._row(
                "Configure Provider",
                Select(
                    _PROVIDER_OPTIONS,
                    value=self.configure_provider,
                    id="settings-speech-configure-provider",
                    allow_blank=False,
                    compact=True,
                    classes="settings-compact-select",
                ),
                classes="settings-select-row",
            )
            yield from self._compose_provider_form(self.configure_provider)

        yield from self._compose_realtime_section()

        with Vertical(id="settings-speech-inspector", classes="settings-focus-card"):
            yield Static("Configuration inspector", classes="destination-section")
            yield Static(
                f"Selected setup: {TTS_PROVIDER_LABELS[self.configure_provider]}",
                id="settings-speech-inspector-summary",
                classes="settings-status-row",
                markup=False,
            )
            yield Static(
                "Selected provider setup source: "
                f"{self.state.provider_sources[self.configure_provider].value}.",
                id="settings-speech-provider-source",
                classes="settings-status-row",
                markup=False,
            )
            yield Static(
                self._draft_impact_copy(),
                id="settings-speech-draft-impact",
                classes="settings-status-row",
                markup=False,
            )
            projection = self._status_projection()
            dirty_connection = self._provider_connection_draft_dirty(
                self.configure_provider
            )
            for row in projection.rows(dirty_draft=dirty_connection):
                yield Static(
                    row.copy,
                    id=f"settings-speech-status-{row.row_id}",
                    classes="settings-status-row",
                    markup=False,
                )
            yield Static(
                "Ordinary Save validates and persists locally. Use Speech Lab for "
                "connection tests, discovery, generation, and playback.",
                classes="settings-detail-row",
                markup=False,
            )

        with Horizontal(id="settings-speech-actions", classes="settings-action-row"):
            yield Button(
                "Save",
                id="settings-speech-save",
                variant="primary",
                disabled=self._latest_request_id is not None,
            )
            yield Button("Revert", id="settings-speech-revert")
            yield Button(
                "Restore Non-secret Defaults",
                id="settings-speech-restore-defaults",
            )
            yield Button("Open Speech Lab", id="settings-speech-open-lab-bottom")
        yield Static(
            self.result_text,
            id="settings-speech-save-result",
            classes="settings-status-row",
            markup=False,
        )

    def _compose_realtime_section(self) -> ComposeResult:
        """Build the Realtime engine block: config keys owned by task 6.

        Deliberately not part of the per-TTS-provider form above -- `realtime`
        and `dictation` are plain top-level config sections with no adapter to
        reconfigure, so this reuses only the panel's row/error/draft-field
        conventions, not `GlobalSpeechTTSState`.
        """
        draft = self._realtime_draft
        with Vertical(
            id="settings-speech-realtime", classes="settings-focus-card"
        ):
            yield Static("Realtime engine", classes="destination-section")
            yield Static(
                "Optional low-latency voice engine for the Console's hands-free "
                "loop (Ctrl+Shift+H). When off, the pipeline engine (record, "
                "transcribe, reply, speak) is used, as before.",
                classes="settings-detail-row",
                markup=False,
            )
            yield self._row(
                "Enable realtime voice engine",
                Switch(
                    value=draft.enabled,
                    id="settings-speech-realtime-enabled",
                    classes="settings-speech-field settings-speech-draft-field",
                ),
                error=self._error("realtime", "enabled"),
            )
            yield self._row(
                "Provider",
                Select(
                    _REALTIME_PROVIDER_OPTIONS,
                    value=DEFAULT_REALTIME_PROVIDER,
                    id="settings-speech-realtime-provider",
                    allow_blank=False,
                    compact=True,
                    classes="settings-compact-select settings-speech-draft-field",
                ),
                classes="settings-select-row",
                error=self._error("realtime", "provider"),
            )
            yield self._row(
                "Model",
                Input(
                    value=draft.model,
                    id="settings-speech-realtime-model",
                    placeholder=DEFAULT_REALTIME_MODEL,
                    classes=(
                        "settings-compact-input settings-speech-draft-field"
                    ),
                ),
                error=self._error("realtime", "model"),
            )
            yield self._row(
                "Voice (optional)",
                Input(
                    value=draft.voice,
                    id="settings-speech-realtime-voice",
                    placeholder="Provider default",
                    classes=(
                        "settings-compact-input settings-speech-draft-field"
                    ),
                ),
                error=self._error("realtime", "voice"),
            )
            yield self._row(
                "Idle timeout (minutes)",
                Input(
                    value=draft.idle_timeout_minutes,
                    id="settings-speech-realtime-idle-timeout-minutes",
                    placeholder=str(DEFAULT_REALTIME_IDLE_TIMEOUT_MINUTES),
                    classes=(
                        "settings-compact-input settings-speech-draft-field"
                    ),
                ),
                error=self._error("realtime", "idle_timeout_minutes"),
            )
            yield self._row(
                "Hands-free engine",
                Select(
                    _REALTIME_HANDSFREE_ENGINE_OPTIONS,
                    value=draft.handsfree_engine,
                    id="settings-speech-realtime-handsfree-engine",
                    allow_blank=False,
                    compact=True,
                    classes="settings-compact-select settings-speech-draft-field",
                ),
                classes="settings-select-row",
                error=self._error("realtime", "handsfree_engine"),
            )
            yield Static(
                "Spoken commands do not work inside realtime mode -- there is "
                "no client-side speech-to-text running, so phrases like "
                '"Console, stop." are never heard. Exit with Esc, the mic '
                "button, or Ctrl+Shift+H.",
                classes="settings-detail-row",
                markup=False,
            )
            yield Static(
                "Privacy: while a realtime session is live, microphone audio "
                "streams continuously to the provider for the whole session "
                "(subject to barge-in gating) -- not just after a pause, as "
                "the pipeline engine does.",
                classes="settings-detail-row",
                markup=False,
            )
            yield Static(
                "Cost: realtime sessions are billed by the provider per "
                "connected minute. The idle timeout above ends an unattended "
                "session automatically (never while a reply is being spoken); "
                "an unexpected connection drop is retried once.",
                classes="settings-detail-row",
                markup=False,
            )

    def _compose_provider_form(self, provider_id: str) -> ComposeResult:
        with Vertical(
            id=f"settings-speech-provider-{provider_id}",
            classes="settings-speech-provider-form",
        ):
            yield Static(
                TTS_PROVIDER_LABELS[provider_id], classes="destination-section"
            )
            if provider_id == "audio_cpp":
                yield Static(
                    "External server. Chatbook connects to a server that you start and own.",
                    id="settings-speech-audio_cpp-external-mode",
                    classes="settings-detail-row",
                    markup=False,
                )
                yield self._input(
                    provider_id,
                    "base_url",
                    "Server URL (HTTP/HTTPS origin only)",
                    placeholder="http://127.0.0.1:8080",
                )
                yield Static(
                    audio_cpp_transport_warning(
                        self.state.providers[provider_id].get("base_url")
                    )
                    or "",
                    id="settings-speech-audio-cpp-transport-warning",
                    classes="settings-status-row",
                    markup=False,
                )
                yield self._input(
                    provider_id,
                    "connect_timeout_seconds",
                    "Connect timeout (seconds)",
                )
                yield self._input(
                    provider_id,
                    "synthesis_timeout_seconds",
                    "Synthesis timeout (seconds)",
                )
                yield Static(
                    "Generation sends submitted text to this configured server.",
                    id="settings-speech-audio_cpp-privacy-notice",
                    classes="settings-status-row",
                    markup=False,
                )
                yield Static(
                    self._audio_cpp_draft_attribution_copy(),
                    id="settings-speech-audio-cpp-draft-attribution",
                    classes="settings-detail-row",
                    markup=False,
                )
                yield Button(
                    "Open Speech Lab to test or refresh",
                    id="settings-speech-audio-cpp-open-lab",
                    compact=True,
                    tooltip=(
                        "Open the TTS Playground; Settings itself never contacts "
                        "the configured server."
                    ),
                )
                with Collapsible(
                    title="Advanced safety limits",
                    collapsed=True,
                    id="settings-speech-audio-cpp-advanced",
                ):
                    yield self._input(
                        provider_id, "max_input_characters", "Max input characters"
                    )
                    yield self._input(
                        provider_id, "max_response_bytes", "Max response bytes"
                    )
                    yield self._input(
                        provider_id, "max_metadata_bytes", "Max metadata bytes"
                    )
                    yield self._input(
                        provider_id, "max_catalog_models", "Max catalog models"
                    )
                    yield self._input(
                        provider_id,
                        "max_voices_per_model",
                        "Max voices per model",
                    )
                    yield self._input(
                        provider_id,
                        "max_identifier_characters",
                        "Max identifier characters",
                    )
                return

            if provider_id == "openai":
                yield self._credential(provider_id)
                yield self._input(provider_id, "base_url", "Base URL")
                yield self._input(
                    provider_id,
                    "organization_id",
                    "Organization ID",
                    placeholder="Optional",
                )
                return

            if provider_id == "elevenlabs":
                yield self._credential(provider_id)
                yield self._select(
                    provider_id,
                    "output_format",
                    "Output format",
                    [
                        ("MP3 192kbps", "mp3_44100_192"),
                        ("MP3 128kbps", "mp3_44100_128"),
                        ("MP3 96kbps", "mp3_44100_96"),
                        ("MP3 64kbps", "mp3_44100_64"),
                        ("MP3 32kbps", "mp3_44100_32"),
                        ("PCM 44.1kHz", "pcm_44100"),
                        ("PCM 24kHz", "pcm_24000"),
                        ("PCM 16kHz", "pcm_16000"),
                        ("μ-law 8kHz", "ulaw_8000"),
                    ],
                )
                yield self._input(provider_id, "stability", "Voice stability")
                yield self._input(provider_id, "similarity_boost", "Similarity boost")
                yield self._input(provider_id, "style", "Style")
                yield self._switch(provider_id, "speaker_boost", "Speaker boost")
                return

            if provider_id == "kokoro":
                yield self._select(
                    provider_id,
                    "device",
                    "Device",
                    [("CPU", "cpu"), ("CUDA", "cuda"), ("Apple MPS", "mps")],
                )
                yield self._switch(provider_id, "use_onnx", "Use ONNX")
                yield self._path(
                    provider_id,
                    "onnx_model_path",
                    "ONNX model file",
                    placeholder="Path to model file",
                )
                yield self._path(
                    provider_id,
                    "voices_json_path",
                    "Voices JSON file",
                    placeholder="Path to voices.json",
                )
                yield self._input(provider_id, "max_tokens", "Max tokens")
                yield self._switch(provider_id, "voice_mixing", "Voice mixing")
                yield self._switch(
                    provider_id, "track_performance", "Performance tracking"
                )
                return

            if provider_id == "chatterbox":
                yield self._select(
                    provider_id,
                    "device",
                    "Device",
                    [("CPU", "cpu"), ("CUDA", "cuda")],
                )
                yield self._path(
                    provider_id,
                    "voice_resource_directory",
                    "Voice resource directory",
                    placeholder="Path to voice resources",
                )
                yield self._input(provider_id, "temperature", "Temperature")
                yield self._input(provider_id, "chunk_size", "Chunk size")
                yield self._input(
                    provider_id, "random_seed", "Random seed", placeholder="Optional"
                )
                yield self._input(provider_id, "candidates", "Candidates")
                yield self._switch(
                    provider_id, "validate_whisper", "Whisper validation"
                )
                yield self._switch(provider_id, "preprocess_text", "Text preprocessing")
                yield self._switch(
                    provider_id, "normalize_audio", "Audio normalization"
                )
                yield self._input(provider_id, "target_db", "Target dB")
                yield self._input(provider_id, "max_chunk_size", "Max text chunk")
                yield self._switch(provider_id, "streaming", "Streaming")
                yield self._input(provider_id, "stream_chunk_size", "Stream chunk size")
                yield self._switch(provider_id, "crossfade", "Crossfade")
                yield self._input(
                    provider_id, "crossfade_ms", "Crossfade duration (ms)"
                )
                return

            if provider_id == "higgs":
                yield self._path(
                    provider_id,
                    "model_path",
                    "Model path",
                    placeholder="Local path or Hugging Face model ID",
                )
                yield self._path(
                    provider_id,
                    "voice_resource_directory",
                    "Voice resource directory",
                    placeholder="Path to voice samples",
                )
                yield self._select(
                    provider_id,
                    "device",
                    "Device",
                    [
                        ("Auto", "auto"),
                        ("CPU", "cpu"),
                        ("CUDA", "cuda"),
                        ("CUDA 0", "cuda:0"),
                        ("CUDA 1", "cuda:1"),
                        ("Apple MPS", "mps"),
                    ],
                )
                yield self._switch(
                    provider_id,
                    "enable_flash_attention",
                    "Enable flash attention",
                )
                yield self._select(
                    provider_id,
                    "dtype",
                    "Data type",
                    [
                        ("Float32", "float32"),
                        ("Float16", "float16"),
                        ("BFloat16", "bfloat16"),
                    ],
                )
                yield self._input(
                    provider_id,
                    "max_reference_duration",
                    "Max reference duration",
                )
                yield self._select(
                    provider_id, "language", "Default language", _LANGUAGE_OPTIONS
                )
                yield self._switch(provider_id, "voice_cloning", "Voice cloning")
                yield self._switch(provider_id, "multi_speaker", "Multi-speaker")
                yield self._input(provider_id, "speaker_delimiter", "Speaker delimiter")
                yield self._switch(
                    provider_id, "track_performance", "Performance tracking"
                )
                yield self._input(provider_id, "max_new_tokens", "Max new tokens")
                yield self._input(provider_id, "temperature", "Temperature")
                yield self._input(provider_id, "top_p", "Top P")
                yield self._input(
                    provider_id, "repetition_penalty", "Repetition penalty"
                )
                return

            if provider_id == "alltalk":
                yield self._input(
                    provider_id,
                    "server_url",
                    "Server URL",
                    placeholder="http://127.0.0.1:7851",
                )
                yield self._select(
                    provider_id, "language", "Default language", _LANGUAGE_OPTIONS
                )

    def _collect_visible_state(self) -> None:
        """Copy mounted widget values into the in-memory draft."""
        try:
            provider = self.query_one("#settings-speech-default-provider", Select).value
            model_mode = self.query_one("#settings-speech-model-policy", Select).value
            voice_mode = self.query_one("#settings-speech-voice-policy", Select).value
            response_format = self.query_one(
                "#settings-speech-output-format", Select
            ).value
            if isinstance(provider, str):
                self.state.defaults.provider_id = provider
            if isinstance(model_mode, str):
                self.state.defaults.model_mode = model_mode
            model_control = self.query_one("#settings-speech-model-value")
            model_value = (
                model_control.value
                if isinstance(model_control, (Input, Select))
                else None
            )
            self.state.defaults.model_id = (
                model_value
                if self.state.defaults.model_mode == "exact"
                and isinstance(model_value, str)
                else None
            )
            if isinstance(voice_mode, str):
                self.state.defaults.voice_mode = voice_mode
            voice_control = self.query_one("#settings-speech-voice-value")
            voice_value = (
                voice_control.value
                if isinstance(voice_control, (Input, Select))
                else None
            )
            self.state.defaults.voice_id = (
                voice_value
                if self.state.defaults.voice_mode == "exact"
                and isinstance(voice_value, str)
                else None
            )
            if isinstance(response_format, str):
                self.state.defaults.response_format = response_format
            speed = self.query_one("#settings-speech-speed", Input).value
            try:
                self.state.defaults.speed = float(speed)
            except ValueError:
                self.state.defaults.speed = speed  # type: ignore[assignment]
        except QueryError:
            pass

        values = self.state.providers[self.configure_provider]
        for field_id in GLOBAL_TTS_PROVIDER_FIELD_IDS[self.configure_provider]:
            if field_id == "credential":
                continue
            selector = f"#{self._field_dom_id(self.configure_provider, field_id)}"
            try:
                widget = self.query_one(selector)
            except QueryError:
                continue
            if isinstance(widget, Input):
                values[field_id] = widget.value
            elif isinstance(widget, Select):
                if isinstance(widget.value, str):
                    values[field_id] = widget.value
            elif isinstance(widget, Switch):
                values[field_id] = widget.value

        self._collect_realtime_visible_state()

    def _collect_realtime_visible_state(self) -> None:
        """Copy the Realtime block's mounted widget values into its draft."""
        try:
            enabled_widget = self.query_one(
                "#settings-speech-realtime-enabled", Switch
            )
            provider_widget = self.query_one(
                "#settings-speech-realtime-provider", Select
            )
            model_widget = self.query_one("#settings-speech-realtime-model", Input)
            voice_widget = self.query_one("#settings-speech-realtime-voice", Input)
            idle_widget = self.query_one(
                "#settings-speech-realtime-idle-timeout-minutes", Input
            )
            engine_widget = self.query_one(
                "#settings-speech-realtime-handsfree-engine", Select
            )
        except QueryError:
            return
        self._realtime_draft.enabled = bool(enabled_widget.value)
        if isinstance(provider_widget.value, str):
            self._realtime_draft.provider = provider_widget.value
        self._realtime_draft.model = model_widget.value
        self._realtime_draft.voice = voice_widget.value
        self._realtime_draft.idle_timeout_minutes = idle_widget.value
        if isinstance(engine_widget.value, str):
            self._realtime_draft.handsfree_engine = engine_widget.value

    def has_unsaved_changes(self) -> bool:
        """Return whether any non-secret global value differs from its baseline."""
        self._collect_visible_state()
        try:
            if (
                self.state.defaults.snapshot()
                != self.original_state.defaults.snapshot()
            ):
                return True
        except GlobalSpeechTTSValidationError:
            if self.state.defaults != self.original_state.defaults:
                return True

        for provider_id in BUILT_IN_TTS_PROVIDER_ORDER:
            try:
                proposal = build_global_speech_tts_save_proposal(
                    self.original_state,
                    self.state,
                    configure_provider=provider_id,
                )
            except GlobalSpeechTTSValidationError:
                if (
                    self.state.providers[provider_id]
                    != self.original_state.providers[provider_id]
                ):
                    return True
                continue
            if proposal.settings or proposal.delete_setting_keys:
                return True
        return self._realtime_draft.snapshot() != self._realtime_original.snapshot()

    def _announce_draft_state(self) -> None:
        """Publish the latest safe draft snapshot to the Settings shell."""
        is_modified = self.has_unsaved_changes()
        self._refresh_status_rows()
        self.post_message(
            self.DraftModified(
                is_modified,
                self.state,
                self.original_state,
                self.configure_provider,
            )
        )

    def _set_result(self, text: str, *, severity: str | None = None) -> None:
        self.result_text = text
        try:
            self.query_one("#settings-speech-save-result", Static).update(text)
        except QueryError:
            pass
        if severity is not None:
            self.app.notify(text, severity=severity)

    def _set_save_pending(self, pending: bool) -> None:
        """Prevent overlapping persistence requests while keeping draft edits live."""
        for selector in (
            "#settings-speech-save",
            "#settings-speech-openai-credential-edit",
            "#settings-speech-openai-credential-clear",
            "#settings-speech-elevenlabs-credential-edit",
            "#settings-speech-elevenlabs-credential-clear",
        ):
            try:
                self.query_one(selector, Button).disabled = pending
            except QueryError:
                continue

    def _clear_validation_errors(self) -> None:
        self._field_errors.clear()
        for error in self.query(".settings-speech-field-error"):
            if isinstance(error, Static):
                error.update("")
                error.remove_class("settings-speech-field-error-visible")

    def _show_validation_error(self, error: GlobalSpeechTTSValidationError) -> None:
        key = (error.provider_id, error.field_id)
        self._field_errors[key] = str(error)
        selector = (
            f"#{self._field_dom_id(error.provider_id, error.field_id)}"
            if error.provider_id != "defaults"
            else {
                "provider_id": "#settings-speech-default-provider",
                "model_mode": "#settings-speech-model-policy",
                "default_model": "#settings-speech-model-value",
                "voice_mode": "#settings-speech-voice-policy",
                "default_voice": "#settings-speech-voice-value",
                "response_format": "#settings-speech-output-format",
                "default_speed": "#settings-speech-speed",
            }.get(error.field_id, "#settings-speech-default-provider")
        )
        try:
            error_widget = self.query_one(f"{selector}-error", Static)
            error_widget.update(str(error))
            error_widget.add_class("settings-speech-field-error-visible")
        except QueryError:
            pass
        try:
            self.query_one(selector).focus()
        except QueryError:
            pass
        self._set_result(str(error), severity="error")

    def request_save(self) -> int | None:
        """Validate locally and post one atomic ordinary-save proposal."""
        self._collect_visible_state()
        if self._latest_request_id is not None:
            self._set_result("A global Speech & TTS save is already in progress.")
            return None
        self._clear_validation_errors()
        try:
            proposal = build_global_speech_tts_save_proposal(
                self.original_state,
                self.state,
                configure_provider=self.configure_provider,
            )
            defaults_changed = (
                proposal.preferences != self.original_state.defaults.snapshot()
            )
        except GlobalSpeechTTSValidationError as error:
            self._show_validation_error(error)
            self._refresh_status_rows()
            return None

        try:
            realtime_payload = self._validated_realtime_payload()
        except GlobalSpeechTTSValidationError as error:
            self._show_validation_error(error)
            self._refresh_status_rows()
            return None
        realtime_changed = realtime_payload is not None

        if (
            not defaults_changed
            and not proposal.settings
            and not proposal.delete_setting_keys
            and not realtime_changed
        ):
            self._set_result("No global Speech & TTS changes to save.")
            return None

        if realtime_payload is not None:
            if not self._persist_realtime_draft(realtime_payload):
                self._set_result(
                    "Realtime engine settings were not saved.",
                    severity="error",
                )
                return None
            self._realtime_original = replace(self._realtime_draft)

        if (
            not defaults_changed
            and not proposal.settings
            and not proposal.delete_setting_keys
        ):
            # Only the realtime/dictation block changed; already persisted
            # locally above through the same atomic config writer -- no TTS
            # provider adapter round trip needed.
            self._set_result("Saved locally. Realtime engine settings updated.")
            self._announce_draft_state()
            return None

        request_id = self._next_request_id
        self._next_request_id += 1
        self._latest_request_id = request_id
        self._pending_focus_control_id = self._focused_id()
        self._pending_displaced_focus_control_id = None
        self._pending_focus_moved_after_displacement = False
        self._set_save_pending(True)
        self._pending_credential_mutation = None
        self._pending_saved_defaults = (
            deepcopy(self.state.defaults) if defaults_changed else None
        )
        self._pending_saved_provider_id = (
            self.configure_provider
            if proposal.settings or proposal.delete_setting_keys
            else None
        )
        self._pending_saved_provider_values = (
            deepcopy(self.state.providers[self.configure_provider])
            if proposal.settings or proposal.delete_setting_keys
            else None
        )
        self._set_result(
            "Saving global Speech & TTS settings locally…",
            severity="information",
        )
        self.app.post_message(
            STTSSettingsSaveEvent(
                proposal.settings,
                delete_setting_keys=proposal.delete_setting_keys,
                preferences=proposal.preferences,
                request_id=request_id,
                reply_to=self,
            )
        )
        return request_id

    def _validated_realtime_payload(self) -> _RealtimeSavePayload | None:
        """Validate the Realtime block draft; ``None`` when unchanged.

        Sibling validation shape to the global defaults' speed field: an
        invalid value raises `GlobalSpeechTTSValidationError` so it reuses
        the same inline-error display and refuses the *entire* Save (both
        this block and the TTS proposal), never a partial write.
        """
        if self._realtime_draft.snapshot() == self._realtime_original.snapshot():
            return None
        raw_idle = self._realtime_draft.idle_timeout_minutes
        try:
            idle_minutes = float(raw_idle)
        except (TypeError, ValueError):
            idle_minutes = None
        if idle_minutes is None or idle_minutes <= 0:
            raise GlobalSpeechTTSValidationError(
                "realtime",
                "idle_timeout_minutes",
                "Idle timeout must be a positive number of minutes.",
            )
        model = self._realtime_draft.model.strip()
        if not model:
            raise GlobalSpeechTTSValidationError(
                "realtime",
                "model",
                "The realtime model is required.",
            )
        realtime_section: dict[str, Any] = {
            "enabled": self._realtime_draft.enabled,
            "provider": self._realtime_draft.provider,
            "model": model,
            "idle_timeout_minutes": idle_minutes,
        }
        delete_keys: dict[str, tuple[str, ...]] = {}
        voice = self._realtime_draft.voice.strip()
        if voice:
            realtime_section["voice"] = voice
        else:
            delete_keys["realtime"] = ("voice",)
        dictation_section = {"handsfree_engine": self._realtime_draft.handsfree_engine}
        return _RealtimeSavePayload(
            section_values={"realtime": realtime_section, "dictation": dictation_section},
            delete_keys=delete_keys,
        )

    @staticmethod
    def _persist_realtime_draft(payload: _RealtimeSavePayload) -> bool:
        """Write the realtime/dictation draft through the shared config writer."""
        try:
            return bool(
                save_settings_to_cli_config(
                    payload.section_values,
                    delete_keys=payload.delete_keys,
                )
            )
        except Exception:
            logger.exception("Failed to save realtime engine settings")
            return False

    def submit_credential_mutation(
        self,
        provider_id: str,
        intent: CredentialIntent,
        value: str | None,
    ) -> None:
        """Post one separately confirmed Set/Replace/Clear mutation."""
        if self._latest_request_id is not None:
            raise ValueError("Wait for the current Settings save to finish")
        if provider_id not in self.state.credentials:
            raise ValueError("This provider has no Settings credential operation")
        mutation = build_credential_mutation(
            self.state.credentials[provider_id],
            intent,
            value,
        )
        request_id = self._next_request_id
        self._next_request_id += 1
        self._latest_request_id = request_id
        self._pending_focus_control_id = self._focused_id()
        self._pending_displaced_focus_control_id = None
        self._pending_focus_moved_after_displacement = False
        self._set_save_pending(True)
        self._pending_credential_mutation = mutation
        self._pending_saved_defaults = None
        self._pending_saved_provider_id = None
        self._pending_saved_provider_values = None
        settings = {} if mutation.delete else {mutation.setting_key: mutation.value}
        delete_keys = (mutation.setting_key,) if mutation.delete else ()
        self._set_result(
            f"Applying explicit {intent.value} credential operation locally…"
        )
        self.app.post_message(
            STTSSettingsSaveEvent(
                settings,
                delete_setting_keys=delete_keys,
                request_id=request_id,
                reply_to=self,
            )
        )

    def _record_save_runtime_statuses(
        self,
        result: STTSSettingsSaveResult,
        *,
        saved_provider_id: str | None,
    ) -> None:
        """Record bounded publication facts without claiming reachability."""

        providers = set(result.provider_statuses)
        if result.failure_phase == "cache_reload" and saved_provider_id is not None:
            providers.add(saved_provider_id)
        for provider_id in providers:
            observation_key = (provider_id, result.request_id)
            observed_at = self._provider_runtime_request_observed_at.setdefault(
                observation_key,
                datetime.now(timezone.utc),
            )
            configuration_revision = result.provider_configuration_revisions.get(
                provider_id
            )
            if configuration_revision is None:
                self._provider_configuration_revisions.pop(provider_id, None)
                self._provider_runtime_revisions.pop(provider_id, None)
                self._runtime_statuses.pop(provider_id, None)
                continue
            self._provider_configuration_revisions[provider_id] = configuration_revision
            runtime_revision = result.provider_runtime_revisions.get(provider_id)
            if runtime_revision is None:
                self._provider_runtime_revisions.pop(provider_id, None)
            else:
                self._provider_runtime_revisions[provider_id] = runtime_revision
            publication_state = result.provider_statuses.get(provider_id)
            if result.failure_phase == "cache_reload":
                runtime_state = SpeechTTSRuntimeState.UNAVAILABLE
                category = SpeechTTSDiagnosticCategory.CONFIGURATION
                recovery = SpeechTTSNavigationIntent.CONFIGURE
            elif publication_state == "pending":
                runtime_state = SpeechTTSRuntimeState.RECONFIGURING
                category = SpeechTTSDiagnosticCategory.CONFIGURATION
                recovery = SpeechTTSNavigationIntent.TEST
            elif publication_state == "unavailable":
                runtime_state = SpeechTTSRuntimeState.UNAVAILABLE
                category = SpeechTTSDiagnosticCategory.CONNECTION
                recovery = SpeechTTSNavigationIntent.TEST
            else:
                # Applying configuration is not itself a reachability check.
                runtime_state = SpeechTTSRuntimeState.NOT_CHECKED
                category = None
                recovery = SpeechTTSNavigationIntent.TEST
            if publication_state in {"applied", "unchanged"}:
                self._provider_applied_configuration_revisions[provider_id] = (
                    configuration_revision
                )
            self._runtime_statuses[provider_id] = SpeechTTSRuntimeStatus(
                provider_id=provider_id,
                saved_configuration_revision=configuration_revision,
                runtime_revision=runtime_revision,
                catalog_revision=None,
                model_scope=None,
                runtime_state=runtime_state,
                observed_at=observed_at,
                freshness=SpeechTTSStatusFreshness.FRESH,
                diagnostic_category=category,
                recovery_action=recovery,
            )
            if publication_state != "pending":
                self._provider_runtime_request_observed_at.pop(
                    observation_key,
                    None,
                )

    def receive_stts_settings_save_result(
        self,
        result: STTSSettingsSaveResult,
    ) -> None:
        """Apply only the latest bounded persistence/reconfiguration result."""
        if result.request_id != self._latest_request_id:
            return
        focus_id = self._completion_focus_id(self._pending_focus_control_id)
        leave_waiter = self._leave_save_waiters.pop(result.request_id, None)
        self._latest_request_id = None
        self._pending_focus_control_id = None
        self._pending_displaced_focus_control_id = None
        self._pending_focus_moved_after_displacement = False
        self._set_save_pending(False)
        mutation = self._pending_credential_mutation
        saved_defaults = self._pending_saved_defaults
        saved_provider_id = self._pending_saved_provider_id
        saved_provider_values = self._pending_saved_provider_values
        self._pending_credential_mutation = None
        self._pending_saved_defaults = None
        self._pending_saved_provider_id = None
        self._pending_saved_provider_values = None
        if not result.persisted:
            failure = (
                " before replacing the config file"
                if result.failure_phase == "before_replace"
                else " while reloading config caches"
                if result.failure_phase == "cache_reload"
                else ""
            )
            self._set_result(
                f"Global Speech & TTS settings were not saved{failure}.",
                severity="error",
            )
            self._refresh_status_rows()
            if leave_waiter is not None and not leave_waiter.done():
                leave_waiter.set_result(False)
            self.call_later(self._restore_focus, focus_id)
            return

        if mutation is not None:
            previous = self.state.credentials[mutation.provider_id]
            local_saved = not mutation.delete
            environment_effective = previous.source is CredentialSource.ENVIRONMENT
            source = (
                CredentialSource.ENVIRONMENT
                if environment_effective
                else CredentialSource.SAVED_LOCAL
                if local_saved
                else CredentialSource.MISSING
            )
            updated = replace(
                previous,
                source=source,
                local_saved=local_saved,
                local_shadowed=environment_effective and local_saved,
            )
            self.state.credentials[mutation.provider_id] = updated
            self.original_state.credentials[mutation.provider_id] = updated
            if environment_effective or local_saved:
                provider_source = (
                    GlobalSpeechTTSEffectiveSource.ENVIRONMENT
                    if environment_effective
                    else GlobalSpeechTTSEffectiveSource.SAVED_LOCAL
                )
                self.state.provider_sources[mutation.provider_id] = provider_source
                self.original_state.provider_sources[mutation.provider_id] = (
                    provider_source
                )
        else:
            if saved_defaults is not None:
                self.original_state.defaults = saved_defaults
                self.state.defaults_source = GlobalSpeechTTSEffectiveSource.SAVED_LOCAL
                self.original_state.defaults_source = (
                    GlobalSpeechTTSEffectiveSource.SAVED_LOCAL
                )
            if saved_provider_id is not None and saved_provider_values is not None:
                self.original_state.providers[saved_provider_id] = saved_provider_values
                if (
                    self.state.provider_sources[saved_provider_id]
                    is not GlobalSpeechTTSEffectiveSource.ENVIRONMENT
                ):
                    self.state.provider_sources[saved_provider_id] = (
                        GlobalSpeechTTSEffectiveSource.SAVED_LOCAL
                    )
                    self.original_state.provider_sources[saved_provider_id] = (
                        GlobalSpeechTTSEffectiveSource.SAVED_LOCAL
                    )

        self._record_save_runtime_statuses(
            result,
            saved_provider_id=saved_provider_id,
        )
        for provider_id in result.provider_statuses:
            self._provider_runtime_request_ids[provider_id] = result.request_id
        if saved_provider_id == "audio_cpp":
            self._audio_cpp_runtime_revision = self._provider_runtime_revisions.get(
                "audio_cpp"
            )

        cache_reload_failed = result.failure_phase == "cache_reload"
        if result.provider_statuses:
            handoff = ", ".join(
                f"{TTS_PROVIDER_LABELS[provider]}: {status}"
                for provider, status in result.provider_statuses.items()
            )
        else:
            handoff = "no provider adapter recreation needed"
        if cache_reload_failed:
            result_copy = (
                "Saved locally, but the runtime configuration cache reload failed; "
                f"restart or retry. Provider handoff: {handoff}."
            )
        else:
            result_copy = f"Saved locally. Runtime reconfiguration: {handoff}."
        self._set_result(
            result_copy,
            severity=(
                "warning"
                if cache_reload_failed
                or "unavailable" in result.provider_statuses.values()
                else "information"
            ),
        )
        self._announce_draft_state()
        if leave_waiter is not None and not leave_waiter.done():
            leave_waiter.set_result(result.failure_phase is None)
        if mutation is not None or saved_provider_id == "audio_cpp":
            # Credential controls derive their affordances from safe metadata,
            # while a saved audio.cpp connection invalidates prior capability
            # provenance. Repaint either bounded presentation state.
            self.call_later(
                self._recompose_and_restore_focus,
                focus_id,
            )
        else:
            self.call_later(self._restore_focus, focus_id)

    def receive_stts_settings_runtime_result(
        self,
        result: STTSSettingsSaveResult,
    ) -> None:
        """Land only a final handoff matching each provider's newest save."""

        matching_statuses = {
            provider_id: status
            for provider_id, status in result.provider_statuses.items()
            if (
                self._provider_runtime_request_ids.get(provider_id) == result.request_id
                and result.provider_configuration_revisions.get(provider_id)
                == self._provider_configuration_revisions.get(provider_id)
            )
        }
        if not result.persisted or not matching_statuses:
            return
        matching_configuration_revisions = {
            provider_id: revision
            for provider_id, revision in (
                result.provider_configuration_revisions.items()
            )
            if provider_id in matching_statuses
        }
        matching_runtime_revisions = {
            provider_id: revision
            for provider_id, revision in result.provider_runtime_revisions.items()
            if provider_id in matching_statuses
        }
        bounded_result = STTSSettingsSaveResult(
            request_id=result.request_id,
            persisted=True,
            provider_statuses=matching_statuses,
            failure_phase=result.failure_phase,
            provider_configuration_revisions=(matching_configuration_revisions),
            provider_runtime_revisions=matching_runtime_revisions,
        )
        self._record_save_runtime_statuses(
            bounded_result,
            saved_provider_id=None,
        )
        handoff = ", ".join(
            f"{TTS_PROVIDER_LABELS[provider_id]}: {status}"
            for provider_id, status in matching_statuses.items()
        )
        self._set_result(
            f"Saved locally. Runtime reconfiguration completed: {handoff}.",
            severity=(
                "error"
                if "unavailable" in matching_statuses.values()
                else "information"
            ),
        )
        self._refresh_status_rows()

    def _credential_editor_result(
        self,
        provider_id: str,
        intent: CredentialIntent,
        value: str | None,
    ) -> None:
        if value is None:
            return
        try:
            self.submit_credential_mutation(provider_id, intent, value)
        except (TypeError, ValueError) as error:
            self._set_result(str(error), severity="error")

    def _credential_clear_result(self, provider_id: str, confirmed: bool) -> None:
        if not confirmed:
            return
        try:
            self.submit_credential_mutation(
                provider_id,
                CredentialIntent.CLEAR,
                None,
            )
        except (TypeError, ValueError) as error:
            self._set_result(str(error), severity="error")

    def _path_picker_result(self, target_selector: str, path: Path | None) -> None:
        if path is None:
            return
        try:
            self.query_one(target_selector, Input).value = str(path)
        except QueryError:
            return

    async def _ask_leave_choice(self) -> LeaveChoice:
        return await self.app.push_screen_wait(_GlobalSpeechTTSLeaveModal())

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        """Remember the draft control even if a navigation click takes focus."""

        control = event.control
        focus_token = self._focus_token(control)
        if focus_token is not None:
            if (
                self._pending_focus_control_id is not None
                and focus_token != self._pending_focus_control_id
                and self._pending_invoking_control_is_disabled()
            ):
                if self._pending_displaced_focus_control_id is None:
                    # Disabling the invoking action blurs it; Textual then
                    # advances to the next focusable control. Remember that
                    # automatic landing separately from a later user move.
                    self._pending_displaced_focus_control_id = focus_token
                elif focus_token != self._pending_displaced_focus_control_id:
                    self._pending_focus_moved_after_displacement = True
            self._last_focused_control_id = focus_token

    @staticmethod
    def _focus_token(control: Widget | None) -> str | None:
        """Return a stable token for ID'd controls or a Collapsible title."""

        if control is None:
            return None
        if control.id:
            return control.id
        parent = control.parent
        if isinstance(parent, Collapsible) and parent.id:
            return f"{parent.id}{_COLLAPSIBLE_TITLE_FOCUS_SUFFIX}"
        return None

    def _resolve_focus_token(self, focus_token: str) -> Widget | None:
        """Resolve a stable focus token after an optional panel recompose."""

        try:
            if focus_token.endswith(_COLLAPSIBLE_TITLE_FOCUS_SUFFIX):
                collapsible_id = focus_token.removesuffix(
                    _COLLAPSIBLE_TITLE_FOCUS_SUFFIX
                )
                return self.query_one(f"#{collapsible_id}", Collapsible).query_one(
                    "CollapsibleTitle"
                )
            return self.query_one(f"#{focus_token}")
        except QueryError:
            return None

    def _pending_invoking_control_is_disabled(self) -> bool:
        """Return whether pending work disabled the control that invoked it."""

        if self._pending_focus_control_id is None:
            return False
        control = self._resolve_focus_token(self._pending_focus_control_id)
        return control is not None and control.disabled

    def _focused_id(self) -> str | None:
        focused = self.app.focused
        focus_token = self._focus_token(focused)
        if (
            focused is not None
            and focus_token is not None
            and self in focused.ancestors_with_self
        ):
            self._last_focused_control_id = focus_token
        return self._last_focused_control_id

    def _restore_focus(self, control_id: str | None) -> None:
        if not control_id or not self.is_mounted:
            return
        control = self._resolve_focus_token(control_id)
        if control is not None:
            control.screen.set_focus(control)

    def _completion_focus_id(self, fallback_id: str | None) -> str | None:
        """Keep newer focus, falling back only when the invoking focus was lost."""

        focused = self.app.focused
        if focused is None or not focused.is_mounted:
            return fallback_id
        if self not in focused.ancestors_with_self:
            return None
        focus_token = self._focus_token(focused)
        if focus_token is not None:
            if (
                not self._pending_focus_moved_after_displacement
                and focus_token == self._pending_displaced_focus_control_id
            ):
                return fallback_id
            self._last_focused_control_id = focus_token
            return focus_token
        return fallback_id

    async def _recompose_and_restore_focus(self, control_id: str | None) -> None:
        """Repaint derived controls while retaining the invoking keyboard target."""

        await self.recompose()
        self._restore_focus(control_id)

    async def confirm_leave(self) -> bool:
        """Resolve the global draft before its owner or surface changes."""

        focus_id = self._focused_id()
        if self._latest_request_id is not None:
            request_id = self._latest_request_id
            waiter = self._leave_save_waiters.get(request_id)
            if waiter is None:
                waiter = asyncio.get_running_loop().create_future()
                self._leave_save_waiters[request_id] = waiter
            saved = await waiter
            if not saved:
                self.call_later(self._restore_focus, focus_id)
                return False
        if not self.has_unsaved_changes():
            return True
        choice = await self._ask_leave_choice()
        if choice == "discard":
            self._reset_to_saved()
            self.post_message(
                self.DraftModified(
                    False,
                    self.state,
                    self.original_state,
                    self.configure_provider,
                )
            )
            return True
        if choice == "save":
            request_id = self.request_save()
            if request_id is None:
                self._restore_focus(focus_id)
                return False
            waiter: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
            self._leave_save_waiters[request_id] = waiter
            saved = await waiter
            if not saved:
                self.call_later(self._restore_focus, focus_id)
            return saved
        self._restore_focus(focus_id)
        return False

    @on(Select.Changed, "#settings-speech-configure-provider")
    async def handle_configure_provider_changed(self, event: Select.Changed) -> None:
        if self._syncing or not isinstance(event.value, str):
            return
        if event.value != event.select.value:
            return
        if event.value == self.configure_provider:
            return
        self._collect_visible_state()
        requested_provider = event.value
        draft_dirty = self.has_unsaved_changes()
        if not draft_dirty:
            await self._apply_configure_provider(requested_provider)
            return
        try:
            selector = self.query_one(
                "#settings-speech-configure-provider",
                Select,
            )
            self._syncing = True
            selector.value = self.configure_provider
        finally:
            self._syncing = False
        self.run_worker(
            self._change_configure_provider(
                requested_provider,
                confirm_dirty=draft_dirty,
            ),
            group="settings-speech-provider-leave",
            exclusive=True,
            exit_on_error=False,
        )

    async def _apply_configure_provider(self, requested_provider: str) -> None:
        self.configure_provider = requested_provider
        await self.recompose()
        self._announce_draft_state()

    async def _change_configure_provider(
        self,
        requested_provider: str,
        *,
        confirm_dirty: bool,
    ) -> None:
        if confirm_dirty and not await self.confirm_leave():
            return
        await self._apply_configure_provider(requested_provider)

    @on(Select.Changed, "#settings-speech-default-provider")
    async def handle_default_provider_changed(self, event: Select.Changed) -> None:
        if self._syncing or not isinstance(event.value, str):
            return
        if event.value == self.state.defaults.provider_id:
            return
        self._collect_visible_state()
        if event.value == "audio_cpp":
            persisted = self.original_state.defaults
            if persisted.provider_id == "audio_cpp":
                self.state.defaults.model_mode = persisted.model_mode
                self.state.defaults.model_id = persisted.model_id
                self.state.defaults.voice_mode = persisted.voice_mode
                self.state.defaults.voice_id = persisted.voice_id
            else:
                self.state.defaults.model_mode = "first_available"
                self.state.defaults.model_id = None
                self.state.defaults.voice_mode = "server_default"
                self.state.defaults.voice_id = None
            self.state.defaults.response_format = "wav"
            self.state.defaults.speed = 1.0
        await self.recompose()
        self._announce_draft_state()

    @on(Select.Changed, "#settings-speech-model-policy")
    async def handle_model_policy_changed(self, event: Select.Changed) -> None:
        if not isinstance(event.value, str):
            return
        if event.value == self.state.defaults.model_mode:
            return
        self._collect_visible_state()
        if event.value != "exact":
            self.state.defaults.model_id = None
        elif self.state.defaults.provider_id == "audio_cpp":
            choices = self._audio_cpp_choices().model.options
            if choices and not self.state.defaults.model_id:
                self.state.defaults.model_id = choices[0][1]
        await self.recompose()
        self._announce_draft_state()

    @on(Select.Changed, "#settings-speech-voice-policy")
    async def handle_voice_policy_changed(self, event: Select.Changed) -> None:
        if not isinstance(event.value, str):
            return
        if event.value == self.state.defaults.voice_mode:
            return
        self._collect_visible_state()
        if event.value != "exact":
            self.state.defaults.voice_id = None
        elif self.state.defaults.provider_id == "audio_cpp":
            choices = self._audio_cpp_choices().voice.options
            if choices and not self.state.defaults.voice_id:
                self.state.defaults.voice_id = choices[0][1]
        await self.recompose()
        self._announce_draft_state()

    @on(Select.Changed, "#settings-speech-model-value")
    async def handle_audio_cpp_model_changed(self, event: Select.Changed) -> None:
        if (
            self._syncing
            or self.state.defaults.provider_id != "audio_cpp"
            or self.state.defaults.model_mode != "exact"
            or not isinstance(event.value, str)
            or event.value == self.state.defaults.model_id
        ):
            return
        self._collect_visible_state()
        await self.recompose()
        self._announce_draft_state()

    @on(Input.Changed, "#settings-speech-audio_cpp-base-url")
    def handle_audio_cpp_base_url_changed(self, event: Input.Changed) -> None:
        try:
            warning = self.query_one(
                "#settings-speech-audio-cpp-transport-warning",
                Static,
            )
            attribution = self.query_one(
                "#settings-speech-audio-cpp-draft-attribution",
                Static,
            )
        except QueryError:
            return
        warning.update(audio_cpp_transport_warning(event.value) or "")
        attribution.update(self._audio_cpp_draft_attribution_copy(event.value))

    @on(Input.Changed, ".settings-speech-draft-field")
    @on(Select.Changed, ".settings-speech-draft-field")
    @on(Switch.Changed, ".settings-speech-draft-field")
    def handle_draft_field_changed(
        self,
        _event: Input.Changed | Select.Changed | Switch.Changed,
    ) -> None:
        if not self._syncing:
            self._announce_draft_state()

    def _reset_to_saved(self) -> None:
        """Reset the draft model without starting a competing recompose."""

        self._clear_validation_errors()
        self.state = deepcopy(self.original_state)
        self._realtime_draft = replace(self._realtime_original)
        self.result_text = "Reverted to the last successfully loaded global values."

    async def revert_to_saved(self) -> None:
        """Restore the last successfully loaded or published snapshot."""

        self._reset_to_saved()
        try:
            get_current_worker()
        except NoActiveWorker:
            await self.recompose()
            self._announce_draft_state()
        else:
            # A worker awaiting recompose can deadlock against the same
            # message pump that owns guarded navigation. Schedule that path.
            self.refresh(recompose=True)
            self.call_after_refresh(self._announce_draft_state)

    @on(Button.Pressed, "#settings-speech-revert")
    async def handle_revert(self, event: Button.Pressed) -> None:
        event.stop()
        await self.revert_to_saved()

    @on(Button.Pressed, "#settings-speech-save")
    def handle_save(self, event: Button.Pressed) -> None:
        event.stop()
        self.request_save()

    @on(Button.Pressed, "#settings-speech-restore-defaults")
    async def handle_restore_defaults(self, event: Button.Pressed) -> None:
        event.stop()
        self._collect_visible_state()
        self._clear_validation_errors()
        self.state = restore_non_secret_defaults(
            self.state,
            configure_provider=self.configure_provider,
        )
        self.result_text = (
            "Non-secret defaults restored in the draft; choose Save to persist them."
        )
        await self.recompose()
        self._announce_draft_state()

    @on(Button.Pressed, "#settings-speech-open-lab")
    @on(Button.Pressed, "#settings-speech-open-lab-bottom")
    @on(Button.Pressed, "#settings-speech-audio-cpp-open-lab")
    def handle_open_lab(self, event: Button.Pressed) -> None:
        event.stop()
        intent = (
            SpeechTTSNavigationIntent.REFRESH_MODELS
            if event.button.id == "settings-speech-audio-cpp-open-lab"
            else SpeechTTSNavigationIntent.TEST
        )
        provider_id = (
            "audio_cpp"
            if event.button.id == "settings-speech-audio-cpp-open-lab"
            else self.configure_provider
        )
        self.run_worker(
            self._open_lab(
                SpeechTTSNavigationTarget(provider_id, intent),
            ),
            group="settings-speech-open-lab",
            exclusive=True,
            exit_on_error=False,
        )

    async def _open_lab(self, target: SpeechTTSNavigationTarget) -> None:
        context = {
            "view": "playground",
            **speech_tts_navigation_context(target),
        }
        self.app.post_message(NavigateToScreen("stts", context))

    @on(Button.Pressed, ".settings-speech-credential Button")
    def handle_credential_action(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = event.button.id or ""
        provider_id = next(
            (
                provider
                for provider in self.state.credentials
                if button_id.startswith(f"settings-speech-{provider}-credential-")
            ),
            None,
        )
        if provider_id is None:
            return
        credential_state = self.state.credentials[provider_id]
        if button_id.endswith("-edit"):
            intent = (
                CredentialIntent.REPLACE
                if credential_state.local_saved
                else CredentialIntent.SET
            )
            self.app.push_screen(
                _CredentialEditorModal(TTS_PROVIDER_LABELS[provider_id], intent),
                lambda value: self._credential_editor_result(
                    provider_id,
                    intent,
                    value,
                ),
            )
            return
        if button_id.endswith("-clear"):
            self.app.push_screen(
                _CredentialClearModal(TTS_PROVIDER_LABELS[provider_id]),
                lambda confirmed: self._credential_clear_result(
                    provider_id,
                    confirmed,
                ),
            )

    @on(Button.Pressed, ".settings-speech-path-picker")
    def handle_path_picker(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = event.button.id or ""
        if not button_id.endswith("-browse"):
            return
        target_selector = f"#{button_id.removesuffix('-browse')}"
        directory_fields = {
            "#settings-speech-chatterbox-voice-resource-directory",
            "#settings-speech-higgs-voice-resource-directory",
        }
        if target_selector in directory_fields:
            picker = SelectDirectory(title="Choose voice resource directory")
        else:
            filters = Filters(("Model and resource files", lambda _path: True))
            picker = FileOpen(
                title="Choose Speech & TTS resource",
                filters=filters,
            )
        self.app.push_screen(
            picker,
            lambda path: self._path_picker_result(target_selector, path),
        )
