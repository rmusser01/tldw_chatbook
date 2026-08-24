"""Global Speech & TTS Settings panel.

This is a bounded built-in surface.  Provider forms are deliberately explicit
so moving a field between global and Studio ownership is a reviewed code
change rather than a dynamic schema side effect.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Literal

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
from textual.widgets import Button, Collapsible, Input, Select, Static, Switch, TextArea
from textual.worker import NoActiveWorker, get_current_worker

from tldw_chatbook.Chat.console_voice_input import (
    DEFAULT_REALTIME_IDLE_TIMEOUT_MINUTES,
    DEFAULT_REALTIME_MODEL,
    DEFAULT_REALTIME_PROVIDER,
    handsfree_engine as _read_handsfree_engine,
    realtime_enabled as _read_realtime_enabled,
    realtime_model as _read_realtime_model,
    realtime_provider as _read_realtime_provider,
    realtime_turn_detection as _read_realtime_turn_detection,
    realtime_vad_silence_ms as _read_realtime_vad_silence_ms,
    realtime_vad_threshold as _read_realtime_vad_threshold,
    realtime_voice as _read_realtime_voice,
)
from tldw_chatbook.config import get_cli_setting, save_settings_to_cli_config
from tldw_chatbook.Model_Artifacts.service import ArtifactRef
from tldw_chatbook.Model_Artifacts.store import managed_service
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
    STTSSettingsSaveResult,
)
from tldw_chatbook.TTS.adapter_types import TTSNativeCapabilityObservation
from tldw_chatbook.TTS.audio_cpp_guided_config import (
    AudioCppAcceptedPackage,
    AudioCppSettingsConfig,
)
from tldw_chatbook.TTS.audio_cpp_guided_launch import (
    AudioCppGuidedLaunchError,
    revalidate_audio_cpp_guided_packages,
)
from tldw_chatbook.TTS.audio_cpp_package_scanner import (
    AudioCppPackageScanError,
    AudioCppScanOutcome,
    scan_audio_cpp_package_root_async,
)
from tldw_chatbook.TTS.audio_cpp_recipes import (
    AUDIO_CPP_RECIPE_REGISTRY,
    AUDIO_CPP_PINNED_RELEASE,
    AudioCppMatchState,
    AudioCppReferenceRequirement,
)
from tldw_chatbook.Third_Party.textual_fspicker import (
    FileOpen,
    Filters,
    SelectDirectory,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Navigation.audio_cpp_model_handoff import (
    AudioCppManagedLeaseHold,
    AudioCppManagedLeasePublication,
    AudioCppModelInstallOwner,
)
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
    SpeechTTSConnectionState,
    SpeechTTSDiagnosticCategory,
    SpeechTTSNavigationIntent,
    SpeechTTSNavigationTarget,
    SpeechTTSRuntimeState,
    SpeechTTSRuntimeStatus,
    SpeechTTSStatusFreshness,
    combine_tts_readiness,
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
    GlobalSpeechTTSSaveProposal,
    GlobalSpeechTTSState,
    GlobalSpeechTTSValidationError,
    OpenAIPlaintextConfirmation,
    ProcessProviderTestEvidenceStore,
    audio_cpp_transport_warning,
    build_credential_mutation,
    build_global_speech_tts_save_proposal,
    build_provider_test_fingerprint,
    detect_audio_cpp_server_binary,
    global_speech_tts_provider_configuration_changed,
    global_speech_tts_provider_configuration_state,
    project_audio_cpp_global_choices,
    required_openai_plaintext_confirmation_fingerprint,
    restore_non_secret_defaults,
    validate_audio_cpp_managed_settings,
)

# TASK-21108: the draft-validation cluster below (bounds, the realtime
# sibling draft, the detach/validate helpers, and the
# ``SpeechTTSPanelDraftSnapshot`` payload) lives in the pure
# ``speech_tts_panel_types`` module so ``app.py`` can read the payload class
# without importing this 5,600-line widget module. These are re-exports, not
# copies: ``type(x) is SpeechTTSPanelDraftSnapshot`` stays true across both
# import paths.
from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_panel_types import (
    _MAX_DRAFT_REVISION,
    _RealtimeSettingsDraft,
    SpeechTTSPanelDraftSnapshot,
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
_AUDIO_CPP_MANAGED_UI_SUPPORTED = os.name != "nt"
_AUDIO_CPP_MANAGED_FIELD_IDS = frozenset(
    {
        "managed_binary_path",
        "managed_server_json_path",
        "managed_startup_timeout_seconds",
        "managed_health_check_interval_seconds",
        "managed_termination_grace_seconds",
    }
)
_AUDIO_CPP_MANUAL_FIELD_IDS = frozenset(
    {
        "managed_binary_path",
        "managed_server_json_path",
    }
)
_AUDIO_CPP_GUIDED_FIELD_IDS = frozenset(
    {
        "guided_binary_path",
        "guided_binary_source",
        "guided_packages",
        "guided_default_model_id",
        "guided_backend_preference",
        "guided_device",
        "guided_threads",
        "guided_max_request_body_bytes",
        "guided_busy_timeout_ms",
    }
)
_AUDIO_CPP_MANAGED_DEFAULTS = AudioCppSettingsConfig(mode="managed").to_mapping()
_AUDIO_CPP_PACKAGE_SCAN_GROUP = "settings-audio-cpp-package-scan"
_AUDIO_CPP_PACKAGE_SAVE_VALIDATION_GROUP = "settings-audio-cpp-package-save-validation"
_SEMANTIC_DRAFT_CONTROL_IDS = frozenset(
    {
        "settings-speech-configure-provider",
        "settings-speech-default-provider",
        "settings-speech-model-policy",
        "settings-speech-voice-policy",
        "settings-speech-model-value",
        "settings-speech-audio_cpp-mode",
        "settings-speech-audio_cpp-managed-setup-source",
    }
)
# The panel's own blank sentinel for "no default voice profile chosen" in the
# `#settings-speech-default-profile` Select. Never a real saved value: Task 2's
# loader normalizes an absent/blank/whitespace-only `default_profile_id` to
# `None`, so an empty string can never be a genuine saved profile id.
_NO_DEFAULT_PROFILE_ID = ""

_REALTIME_PROVIDER_OPTIONS = [("OpenAI", DEFAULT_REALTIME_PROVIDER)]


def _realtime_provider_options(configured: str) -> list[tuple[str, str]]:
    """Provider choices, including an unsupported configured value.

    A `Select` cannot hold a value that is not one of its options, so a
    config naming a provider this release does not implement (a typo, or
    an aspirational edit) has to appear as an explicit unsupported entry.
    Composing with the DEFAULT instead -- what this did before -- made the
    panel report unsaved changes the moment it opened and silently
    rewrote the user's value to "openai" on Save, without anyone touching
    the field (final review M5).

    Args:
        configured: The provider currently in config.

    Returns:
        The supported options, plus `configured` labelled unsupported when
        it is not among them.
    """
    value = str(configured or "").strip()
    if not value or any(
        value == option for _label, option in _REALTIME_PROVIDER_OPTIONS
    ):
        return list(_REALTIME_PROVIDER_OPTIONS)
    return [*_REALTIME_PROVIDER_OPTIONS, (f"{value} (not supported)", value)]


_REALTIME_TURN_DETECTION_OPTIONS = [
    ("Semantic (model decides when you finished)", "semantic_vad"),
    ("Server VAD (silence-gated)", "server_vad"),
]
_REALTIME_HANDSFREE_ENGINE_OPTIONS = [
    ("Auto (realtime when enabled)", "auto"),
    ("Pipeline (record / transcribe / reply / speak)", "pipeline"),
    ("Realtime (forced)", "realtime"),
]


@dataclass(frozen=True)
class _RealtimeSavePayload:
    """One validated realtime/dictation mutation, ready for the shared writer."""

    section_values: dict[str, dict[str, Any]]
    delete_keys: dict[str, tuple[str, ...]]
    persisted_draft: _RealtimeSettingsDraft


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
        turn_detection=_read_realtime_turn_detection(),
        # Empty string means "unset" all the way through: the readers
        # return None for an unset knob, and Save deletes the key rather
        # than writing a number the user never chose.
        vad_threshold=_format_optional_number(_read_realtime_vad_threshold()),
        vad_silence_ms=_format_optional_number(_read_realtime_vad_silence_ms()),
    )


def _format_optional_number(value: float | int | None) -> str:
    """Render an optional numeric setting for an Input, blank when unset."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


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


class _OpenAINoneHTTPConfirmationModal(ModalScreen[bool]):
    """Consent before sending speech text over unauthenticated plaintext HTTP."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Cancel", show=False)
    ]

    def compose(self) -> ComposeResult:
        with Vertical(classes="settings-speech-credential-modal"):
            yield Static(
                "Confirm unauthenticated HTTP",
                classes="destination-section",
            )
            yield Static(
                "Speech text will be sent without encryption or authentication. "
                "Confirm only if you trust the configured server and network.",
                classes="settings-detail-row",
                markup=False,
            )
            with Horizontal(classes="settings-action-row"):
                yield Button(
                    "Cancel",
                    id="settings-speech-openai-none-http-cancel",
                )
                yield Button(
                    "Confirm and save",
                    id="settings-speech-openai-none-http-confirm",
                    variant="warning",
                )

    def action_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#settings-speech-openai-none-http-cancel")
    def handle_cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(False)

    @on(Button.Pressed, "#settings-speech-openai-none-http-confirm")
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


class _SpeechSettingsCard(Vertical):
    """One Speech settings focus card, recomposable on its own (task-15475).

    A plain ``Vertical`` whose children come from a builder the owning panel
    supplies. The indirection buys exactly one thing: ``await card.recompose()``
    rebuilds THIS card -- Textual only re-runs ``compose()`` on the widget that
    owns it, so the card's contents have to be reachable from a ``compose()``
    of its own rather than inline in the panel's.

    Still a ``Vertical`` subclass, so every ``Vertical`` and
    ``.settings-focus-card`` CSS rule that styled these cards before keeps
    matching (Textual type selectors match base classes too).
    """

    def __init__(self, builder: Callable[[], ComposeResult], **kwargs: Any) -> None:
        """Store the panel-side builder for this card's children.

        Args:
            builder: Zero-argument callable yielding the card's children. A
                bound method of the owning panel: the panel outlives its
                cards (a card recompose never replaces the panel), and a
                panel recompose mints fresh cards bound to the fresh panel --
                so the reference can never go stale.
            **kwargs: Forwarded to ``Vertical`` (id, classes, ...).
        """
        super().__init__(**kwargs)
        self._builder = builder

    def compose(self) -> ComposeResult:
        """Yield this card's children from the panel's builder."""
        yield from self._builder()


class _SpeechTTSCleanupActionButton(Button):
    """Derive a remounted action's fence from its current panel owner."""

    def on_mount(self) -> None:
        owner = self._cleanup_owner()
        if owner is not None:
            callback = getattr(owner, "_audio_cpp_result_cleanup_pending", None)
            fence_owner = getattr(callback, "__self__", None)
            if fence_owner is not None and hasattr(
                fence_owner, "audio_cpp_result_cleanup_fenced"
            ):
                self.watch(
                    fence_owner,
                    "audio_cpp_result_cleanup_fenced",
                    self._sync_cleanup_state,
                    init=True,
                )
            if self.id == "settings-speech-save":
                owner.audio_cpp_result_cleanup_action_mounted()  # type: ignore[attr-defined]
        self.call_later(self._sync_cleanup_state)

    def _cleanup_owner(self) -> Widget | None:
        """Return the mounted panel exposing the result-cleanup contract."""

        owner: Widget | None = self.parent
        while owner is not None and not hasattr(
            owner, "audio_cpp_result_cleanup_pending"
        ):
            owner = owner.parent
        return owner

    def _sync_cleanup_state(self, *_changes: object) -> None:
        """Read the owner after both Mount and reactive-update turns."""

        owner = self._cleanup_owner()
        if owner is None:
            return
        pending = bool(owner.audio_cpp_result_cleanup_pending())  # type: ignore[attr-defined]
        save_pending = self.id == "settings-speech-save" and bool(
            getattr(owner, "_latest_request_id", None)
        )
        self.disabled = pending or save_pending


class SpeechTTSSettingsPanel(Vertical):
    """Edit application-wide Speech/TTS defaults and one provider at a time."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("s", "save", "Save Speech & TTS", show=False),
        Binding("r", "revert", "Revert Speech & TTS", show=False),
    ]

    class DraftModified(Message):
        """Report whether the panel has an unsaved non-secret draft."""

        def __init__(
            self,
            is_modified: bool,
            state: GlobalSpeechTTSState,
            original_state: GlobalSpeechTTSState,
            configure_provider: str,
            snapshot: SpeechTTSPanelDraftSnapshot,
        ) -> None:
            self.is_modified = is_modified
            self.state = deepcopy(state)
            self.original_state = deepcopy(original_state)
            self.configure_provider = configure_provider
            self.snapshot = snapshot
            super().__init__()

    def __init__(
        self,
        *,
        state: GlobalSpeechTTSState,
        original_state: GlobalSpeechTTSState | None = None,
        configure_provider: str | None = None,
        profiles: Sequence[tuple[str, str]] | None = None,
        profiles_unavailable: bool = False,
        audio_cpp_observation: TTSNativeCapabilityObservation | None = None,
        audio_cpp_configuration_revision: int | None = None,
        audio_cpp_saved_configuration_revision: int | None = None,
        audio_cpp_applied_configuration_revision: int | None = None,
        provider_configuration_revisions: dict[str, int] | None = None,
        provider_runtime_revisions: dict[str, int] | None = None,
        provider_applied_configuration_revisions: dict[str, int] | None = None,
        runtime_status_store: SpeechTTSRuntimeStatusStore | None = None,
        provider_test_evidence: ProcessProviderTestEvidenceStore | None = None,
        draft_snapshot: SpeechTTSPanelDraftSnapshot | None = None,
        audio_cpp_result_cleanup_pending: Callable[[], bool] | None = None,
        audio_cpp_result_cleanup_mounted: (
            Callable[[SpeechTTSSettingsPanel], None] | None
        ) = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if draft_snapshot is not None:
            if type(draft_snapshot) is not SpeechTTSPanelDraftSnapshot:
                raise TypeError("Speech/TTS panel draft snapshot is invalid")
            restored = SpeechTTSPanelDraftSnapshot(
                state=draft_snapshot.state,
                original_state=draft_snapshot.original_state,
                realtime_draft=draft_snapshot.realtime_draft,
                realtime_original=draft_snapshot.realtime_original,
                configure_provider=draft_snapshot.configure_provider,
                draft_revision=draft_snapshot.draft_revision,
            )
            self.original_state = deepcopy(restored.original_state)
            self.state = deepcopy(restored.state)
            metadata_original = original_state or state
            for target, metadata in (
                (self.state, state),
                (self.original_state, metadata_original),
            ):
                target.credentials = deepcopy(metadata.credentials)
                target.defaults_source = metadata.defaults_source
                target.provider_sources = deepcopy(metadata.provider_sources)
                target.provider_field_sources = deepcopy(
                    metadata.provider_field_sources
                )
        else:
            restored = None
            self.original_state = deepcopy(original_state or state)
            self.state = deepcopy(state)
        self._local_dependencies = speech_local_dependency_availability(refresh=True)
        self._audio_cpp_observation = audio_cpp_observation
        self._runtime_status_store = (
            runtime_status_store or SpeechTTSRuntimeStatusStore()
        )
        self._provider_test_evidence = (
            provider_test_evidence or ProcessProviderTestEvidenceStore()
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
        restored_provider = (
            restored.configure_provider if restored is not None else configure_provider
        )
        self.configure_provider: str = (
            restored_provider
            if isinstance(restored_provider, str)
            and restored_provider in BUILT_IN_TTS_PROVIDER_ORDER
            else state.defaults.provider_id
            if state.defaults.provider_id in BUILT_IN_TTS_PROVIDER_ORDER
            else "audio_cpp"
        )
        # `None` means the profile store hasn't answered yet, or answered
        # with failure -- `profiles_unavailable` distinguishes those two
        # (only meaningful when `profiles is None`): a genuine third state,
        # "loading", that must never be conflated with "unavailable" -- that
        # collapse is exactly the defect this comment guards against (task 3
        # review round 1). An empty sequence is a distinct fourth state: the
        # store answered successfully with zero profiles. Never loaded here:
        # this widget must stay pure of profile-store I/O; the impure screen
        # both supplies the initial value and later calls
        # `apply_profile_choices` with a narrow, focus-safe live update once
        # its async fetch resolves.
        self._profile_choices: list[tuple[str, str]] | None = (
            list(profiles) if profiles is not None else None
        )
        self._profile_choices_unavailable: bool = profiles is None and bool(
            profiles_unavailable
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
        self._pending_commit_defaults_after_handoff = False
        self._deferred_default_activation_drafts: dict[
            int,
            tuple[object, str | None, dict[str, object] | None],
        ] = {}
        self._pending_saved_openai_confirmation: OpenAIPlaintextConfirmation | None = (
            None
        )
        self._pending_saved_openai_confirmation_cleanup_needed: bool | None = None
        self._pending_focus_control_id: str | None = None
        self._pending_displaced_focus_control_id: str | None = None
        self._pending_focus_moved_after_displacement = False
        self._leave_save_waiters: dict[int, asyncio.Future[bool]] = {}
        self._managed_lease_hold: AudioCppManagedLeaseHold | None = None
        self._last_focused_control_id: str | None = None
        self._audio_cpp_scan_revision = 0
        self._audio_cpp_result_cleanup_pending = audio_cpp_result_cleanup_pending or (
            lambda: False
        )
        self._audio_cpp_result_cleanup_mounted = audio_cpp_result_cleanup_mounted
        self._audio_cpp_cleanup_action_mounted = False
        if restored is None:
            self._realtime_original = _read_realtime_settings_draft()
            self._realtime_draft = replace(self._realtime_original)
            self._draft_revision = 0
        else:
            self._realtime_original = replace(restored.realtime_original)
            self._realtime_draft = replace(restored.realtime_draft)
            self._draft_revision = restored.draft_revision
        self._draft_revision_basis = self._draft_revision_values()

    def audio_cpp_result_cleanup_pending(self) -> bool:
        """Return whether package-review cleanup currently fences this draft."""

        try:
            owner = self._managed_cleanup_owner()
            return bool(
                self._audio_cpp_result_cleanup_pending()
                or self._managed_lease_hold is not None
                or (owner is not None and owner.cleanup_pending)
            )
        except Exception:
            return True

    def audio_cpp_result_cleanup_action_mounted(self) -> None:
        """Notify the shell that this panel's current Save action mounted."""

        self._audio_cpp_cleanup_action_mounted = True
        callback = self._audio_cpp_result_cleanup_mounted
        if callback is not None:
            callback(self)

    def audio_cpp_result_cleanup_action_is_mounted(self) -> bool:
        """Return whether this panel's current Save action has mounted."""

        return self._audio_cpp_cleanup_action_mounted

    def _managed_cleanup_owner(self) -> AudioCppModelInstallOwner | None:
        try:
            owner = getattr(self.app, "audio_cpp_model_install_owner", None)
        except Exception:
            return None
        return owner if isinstance(owner, AudioCppModelInstallOwner) else None

    def _fence_audio_cpp_result_cleanup(self) -> bool:
        if not self.audio_cpp_result_cleanup_pending():
            return False
        self._set_result(
            "Finishing installed package review…",
            severity="information",
        )
        return True

    def refresh_audio_cpp_result_cleanup_state(self) -> None:
        """Update only controls governed by the short review transaction."""

        pending = self.audio_cpp_result_cleanup_pending()
        selectors = (
            "#settings-speech-save",
            "#settings-speech-revert",
            "#settings-speech-restore-defaults",
            "#settings-speech-audio-cpp-guided-add-package",
            "#settings-speech-audio-cpp-open-model-library",
            "#settings-speech-audio_cpp-guided-default-model-id",
        )
        for selector in selectors:
            try:
                self.query_one(selector).disabled = pending
            except QueryError:
                pass
        for button in self.query(".settings-speech-audio-cpp-package-remove").results(
            Button
        ):
            button.disabled = pending
        if pending:
            self._fence_audio_cpp_result_cleanup()

    def _draft_revision_values(self) -> tuple[object, ...]:
        """Return detached values whose actual changes advance the revision."""

        return (
            deepcopy(self.state),
            deepcopy(self.original_state),
            replace(self._realtime_draft),
            replace(self._realtime_original),
            self.configure_provider,
        )

    def _synchronize_draft_revision(self) -> None:
        """Advance once when any complete panel draft value actually changed."""

        current = self._draft_revision_values()
        if current == self._draft_revision_basis:
            return
        if self._draft_revision >= _MAX_DRAFT_REVISION:
            raise ValueError("Speech/TTS draft revision is exhausted")
        self._draft_revision += 1
        self._draft_revision_basis = current

    def draft_snapshot(self) -> SpeechTTSPanelDraftSnapshot:
        """Collect mounted values and return one detached complete snapshot."""

        self._collect_visible_state()
        self._synchronize_draft_revision()
        return SpeechTTSPanelDraftSnapshot(
            state=self.state,
            original_state=self.original_state,
            realtime_draft=self._realtime_draft,
            realtime_original=self._realtime_original,
            configure_provider=self.configure_provider,
            draft_revision=self._draft_revision,
        )

    def restore_draft_snapshot(
        self,
        snapshot: SpeechTTSPanelDraftSnapshot,
        *,
        result_text: str | None = None,
    ) -> None:
        """Restore one validated process-local snapshot without advancing it."""

        if type(snapshot) is not SpeechTTSPanelDraftSnapshot:
            raise TypeError("Speech/TTS panel draft snapshot is invalid")
        restored = SpeechTTSPanelDraftSnapshot(
            state=snapshot.state,
            original_state=snapshot.original_state,
            realtime_draft=snapshot.realtime_draft,
            realtime_original=snapshot.realtime_original,
            configure_provider=snapshot.configure_provider,
            draft_revision=snapshot.draft_revision,
        )
        focus_id = self._focused_id() if self.is_mounted else None
        live_metadata = self.state
        original_metadata = self.original_state
        self.state = deepcopy(restored.state)
        self.original_state = deepcopy(restored.original_state)
        for target, metadata in (
            (self.state, live_metadata),
            (self.original_state, original_metadata),
        ):
            target.credentials = deepcopy(metadata.credentials)
            target.defaults_source = metadata.defaults_source
            target.provider_sources = deepcopy(metadata.provider_sources)
            target.provider_field_sources = deepcopy(metadata.provider_field_sources)
        self._realtime_draft = replace(restored.realtime_draft)
        self._realtime_original = replace(restored.realtime_original)
        self.configure_provider = restored.configure_provider
        self._draft_revision = restored.draft_revision
        self._draft_revision_basis = self._draft_revision_values()
        if result_text is not None:
            self.result_text = result_text
        self.refresh(recompose=True)
        self.call_after_refresh(self._restore_focus, focus_id)

    def merge_managed_audio_cpp_package(
        self,
        package: AudioCppAcceptedPackage,
        *,
        expected_revision: int,
    ) -> SpeechTTSPanelDraftSnapshot | None:
        """Merge exactly one reviewed managed package into the current draft."""

        if type(package) is not AudioCppAcceptedPackage:
            return None
        current = self.draft_snapshot()
        if (
            type(expected_revision) is not int
            or current.draft_revision != expected_revision
        ):
            return None
        values = self.state.providers["audio_cpp"]
        if (
            values.get("mode") != "managed"
            or values.get("managed_setup_source") != "guided"
        ):
            return None
        existing = list(self._audio_cpp_guided_packages())
        if any(item.public_model_id == package.public_model_id for item in existing):
            return None

        focus_id = self._focused_id() if self.is_mounted else None
        values["guided_packages"] = [
            item.model_dump(mode="json") for item in (*existing, package)
        ]
        if not values.get("guided_default_model_id"):
            values["guided_default_model_id"] = package.public_model_id
        self._synchronize_draft_revision()
        self.result_text = (
            "Added one installed Model Library package to the unsaved draft. "
            "Review it and choose Save when ready."
        )
        self._audio_cpp_cleanup_action_mounted = False
        self.refresh(recompose=True)
        self.call_after_refresh(self._restore_focus, focus_id)
        return current

    def on_mount(self) -> None:
        """Apply responsive layout without performing provider work."""

        owner = self._managed_cleanup_owner()
        if owner is not None:
            owner.retry_cleanup()
        self._sync_responsive_layout()
        self._apply_audio_cpp_mode_visibility()

    def command_allowed(self, command: str) -> bool:
        """Allow printable letter commands only when text entry is not focused."""

        is_letter_shortcut = (
            len(command) == 1 and command.isprintable() and command.isalpha()
        )
        return not (
            is_letter_shortcut and isinstance(self.app.focused, (Input, TextArea))
        )

    def action_save(self) -> None:
        """Run the panel Save shortcut after text-entry ownership is checked."""

        if self.command_allowed("s"):
            self.request_save()

    async def action_revert(self) -> None:
        """Run the panel Revert shortcut after text-entry ownership is checked."""

        if self.command_allowed("r") and self.has_unsaved_changes():
            await self.revert_to_saved()

    def on_resize(self) -> None:
        """Keep labels and actions inside the available Settings detail width."""

        self._sync_responsive_layout()

    def on_unmount(self) -> None:
        """Fence and cancel any package scan owned by this Settings panel."""

        self._fence_audio_cpp_package_scan()
        self._transfer_managed_refs()

    def _fence_audio_cpp_package_scan(self) -> None:
        """Invalidate even an uncooperative late scan before changing its owner."""

        self._audio_cpp_scan_revision += 1
        self.workers.cancel_group(self, _AUDIO_CPP_PACKAGE_SCAN_GROUP)

    def _sync_responsive_layout(self) -> None:
        """Stack fields and actions when one row cannot fit four cells."""

        self.set_class(
            self.size.width < _GLOBAL_SPEECH_TTS_STACK_WIDTH,
            "settings-speech-stacked",
        )

    def _apply_audio_cpp_mode_visibility(self) -> None:
        """Show the selected primary fields without replacing focused widgets."""

        if self.configure_provider != "audio_cpp":
            return
        mode = self.state.providers["audio_cpp"].get("mode", "external")
        setup_source = self.state.providers["audio_cpp"].get(
            "managed_setup_source",
            "user_json",
        )
        try:
            mode_control = self.query_one("#settings-speech-audio_cpp-mode", Select)
            if isinstance(mode_control.value, str):
                mode = mode_control.value
            source_control = self.query_one(
                "#settings-speech-audio_cpp-managed-setup-source",
                Select,
            )
            if isinstance(source_control.value, str):
                setup_source = source_control.value
            self.query_one("#settings-speech-audio-cpp-external-fields").display = (
                mode == "external"
            )
            self.query_one("#settings-speech-audio-cpp-managed-fields").display = (
                mode == "managed" and _AUDIO_CPP_MANAGED_UI_SUPPORTED
            )
            self.query_one(
                "#settings-speech-audio-cpp-managed-lifecycle-fields"
            ).display = mode == "managed" and _AUDIO_CPP_MANAGED_UI_SUPPORTED
            self.query_one("#settings-speech-audio-cpp-manual-json-fields").display = (
                mode == "managed"
                and _AUDIO_CPP_MANAGED_UI_SUPPORTED
                and setup_source == "user_json"
            )
            self.query_one("#settings-speech-audio-cpp-guided-fields").display = (
                mode == "managed"
                and _AUDIO_CPP_MANAGED_UI_SUPPORTED
                and setup_source == "guided"
            )
            self.query_one(
                "#settings-speech-audio-cpp-guided-advanced-fields"
            ).display = (
                mode == "managed"
                and _AUDIO_CPP_MANAGED_UI_SUPPORTED
                and setup_source == "guided"
            )
            handoff = self.query_one(
                "#settings-speech-audio-cpp-open-lab",
                Button,
            )
            handoff.label = self._audio_cpp_handoff_label(
                self.state.providers["audio_cpp"]
            )
        except QueryError:
            return

    def _sync_audio_cpp_handoff(self, *, draft_modified: bool) -> None:
        """Expose a saved-state handoff only when Speech Lab can observe it."""

        if self.configure_provider != "audio_cpp":
            return
        try:
            handoff = self.query_one(
                "#settings-speech-audio-cpp-open-lab",
                Button,
            )
        except QueryError:
            return
        if draft_modified:
            handoff.label = "Save Settings before opening Speech Lab"
            handoff.disabled = True
            handoff.tooltip = (
                "Save this draft so Speech Lab can use the saved configuration"
            )
        else:
            handoff.label = self._audio_cpp_handoff_label(
                self.state.providers["audio_cpp"]
            )
            handoff.disabled = False
            handoff.tooltip = (
                "Open the TTS Playground; Settings itself never contacts the "
                "configured server."
            )
        handoff.refresh(layout=True)

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

    def _unresolved_profile_label(self, profile_id: str) -> str:
        """Label a Select fallback option for an id absent from known profiles.

        Distinguishes "still loading" (`self._profile_choices is None` and
        the screen hasn't reported failure) from a confirmed failure/
        dangling reference -- the loading copy never says "unavailable".
        """
        if self._profile_choices is None and not self._profile_choices_unavailable:
            return f"{profile_id} (loading…)"
        return f"{profile_id} (unavailable)"

    def _default_profile_options(
        self, *, protect_value: str | None = None
    ) -> list[tuple[str, str]]:
        """Build the Select's options from the injected static profile list.

        Always includes the blank "None" choice. When a value that must
        stay selectable (the saved id, and/or `protect_value` -- the
        Select's own current value during a live refresh) is not among the
        known profiles -- store still loading, confirmed unavailable, or a
        dangling reference (e.g. deleted, or a malformed value from Task 2's
        non-rejecting loader) -- an explicit extra option is appended so the
        Select's value stays legal without inventing a display name or
        dropping the value.
        """
        options: list[tuple[str, str]] = [
            ("None — use the fields below", _NO_DEFAULT_PROFILE_ID)
        ]
        known_ids: set[str] = set()
        if self._profile_choices is not None:
            for display_name, profile_id in self._profile_choices:
                options.append((display_name, profile_id))
                known_ids.add(profile_id)
        for candidate in (self.state.defaults.default_profile_id, protect_value):
            if candidate and candidate not in known_ids:
                options.append((self._unresolved_profile_label(candidate), candidate))
                known_ids.add(candidate)
        return options

    def _default_profile_select_value(self) -> str:
        return self.state.defaults.default_profile_id or _NO_DEFAULT_PROFILE_ID

    def _default_profile_note_copy(self) -> str:
        """Explain an unresolvable or not-yet-resolved saved default.

        Three distinct states, never conflated (task 3 review round 1
        finding): still loading (never says "unavailable"), confirmed
        unavailable (the store answered with failure), and a dangling
        reference (the store answered successfully but doesn't know this
        id). A saved default is never silently dropped in any of them.
        """
        saved_id = self.state.defaults.default_profile_id
        if self._profile_choices is None:
            if not self._profile_choices_unavailable:
                if saved_id:
                    return (
                        "Loading voice profiles… the saved default voice "
                        f"profile ({saved_id}) is kept while this resolves."
                    )
                return "Loading voice profiles…"
            if saved_id:
                return (
                    "The voice profile list is unavailable right now, so "
                    f"the saved default voice profile ({saved_id}) cannot be "
                    "shown by name. It is kept and will not change unless "
                    "you pick something else here."
                )
            return "The voice profile list is unavailable right now."
        known_ids = {profile_id for _label, profile_id in self._profile_choices}
        if saved_id and saved_id not in known_ids:
            return (
                f"Saved default voice profile {saved_id} was not found (it "
                "may have been deleted) and is unavailable. It is kept and "
                "will not change unless you pick something else here."
            )
        return ""

    def _default_profile_note(self) -> Static:
        message = self._default_profile_note_copy()
        return Static(
            message,
            id="settings-speech-default-profile-note",
            classes=(
                "settings-status-row settings-speech-field-error"
                + (" settings-speech-field-error-visible" if message else "")
            ),
            markup=False,
        )

    def apply_profile_choices(
        self,
        profiles: Sequence[tuple[str, str]] | None,
        *,
        unavailable: bool = False,
    ) -> None:
        """Live-update the picker once the impure screen's fetch resolves.

        Mirrors `personas_screen.py`'s `_publish_character_tts_presentation`
        -> `control.apply_state(state)` pattern: a narrow, targeted update to
        already-mounted controls (the Select's options/value, the note's
        text), never a recompose. A recompose here previously stole focus
        out from under a deep-link navigation (task 3 review round 1,
        confirmed live against `test_bounded_speech_settings_deep_link_
        restores_provider_without_action`) -- this path touches nothing
        else, so it is safe to call at any time, including mid-edit.

        The Select's *current* value (whatever the user has it on, saved or
        not) is preserved across the options refresh: `Select.set_options`
        otherwise resets the selection, which would have silently reverted
        an in-progress pick.
        """
        self._profile_choices = list(profiles) if profiles is not None else None
        self._profile_choices_unavailable = profiles is None and unavailable
        try:
            select = self.query_one("#settings-speech-default-profile", Select)
        except QueryError:
            return
        current_value = (
            select.value if isinstance(select.value, str) else _NO_DEFAULT_PROFILE_ID
        )
        select.set_options(self._default_profile_options(protect_value=current_value))
        select.value = current_value
        try:
            note = self.query_one("#settings-speech-default-profile-note", Static)
        except QueryError:
            return
        message = self._default_profile_note_copy()
        note.update(message)
        note.set_class(bool(message), "settings-speech-field-error-visible")

    def _default_profile_row(self) -> Horizontal:
        select = Select(
            self._default_profile_options(),
            value=self._default_profile_select_value(),
            id="settings-speech-default-profile",
            allow_blank=False,
            compact=True,
            tooltip="Default voice profile",
            classes="settings-compact-select settings-speech-draft-field",
        )
        return self._row(
            "Default voice profile",
            Vertical(
                select,
                self._default_profile_note(),
                self._default_error(
                    "default_profile_id", "settings-speech-default-profile"
                ),
                classes="settings-speech-control-stack",
            ),
            classes="settings-select-row",
        )

    def _input(
        self,
        provider_id: str,
        field_id: str,
        label: str,
        *,
        placeholder: str = "",
        disabled: bool = False,
        default: object = "",
    ) -> Horizontal:
        current = self.state.providers[provider_id].get(field_id, default)
        return self._row(
            label,
            Input(
                value="" if current is None else str(current),
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
                    "This value is managed outside Chatbook. The displayed value is "
                    "a local fallback and cannot override the active value.",
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
        source_copy = (
            "Credential is available outside Chatbook."
            if state.source.value == "Environment"
            else "A local credential is available."
            if state.local_saved
            else "No credential is saved in Chatbook."
        )
        controls: list[Button | Static] = [
            Static(source_copy, classes="settings-status-row", markup=False),
            Static(
                "Use an external credential when possible; locally saved credentials "
                "remain on this device.",
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

    def _audio_cpp_guided_packages(self) -> tuple[AudioCppAcceptedPackage, ...]:
        """Return only well-formed accepted package rows from the current draft."""

        raw = self.state.providers["audio_cpp"].get("guided_packages", ())
        if not isinstance(raw, (list, tuple)):
            return ()
        packages: list[AudioCppAcceptedPackage] = []
        for item in raw:
            try:
                packages.append(AudioCppAcceptedPackage.model_validate(item))
            except ValueError:
                continue
        return tuple(packages)

    @staticmethod
    def _managed_refs_from_values(values: Mapping[str, object]) -> set[ArtifactRef]:
        """Return exact managed refs retained in one provider value mapping."""

        raw = values.get("guided_packages", ())
        if not isinstance(raw, (list, tuple)):
            return set()
        references: set[ArtifactRef] = set()
        for item in raw:
            try:
                package = AudioCppAcceptedPackage.model_validate(item)
            except ValueError:
                continue
            identity = package.managed_artifact
            if identity is not None:
                references.add(
                    ArtifactRef(
                        identity.artifact_id,
                        identity.revision,
                        identity.variant,
                    )
                )
        return references

    @staticmethod
    def _sorted_managed_refs(references: set[ArtifactRef]) -> tuple[ArtifactRef, ...]:
        return tuple(
            sorted(
                references,
                key=lambda item: (item.artifact_id, item.revision, item.variant),
            )
        )

    async def _acquire_managed_refs(self, references: set[ArtifactRef]) -> bool:
        """Acquire inactive-root shared leases without activating any selector."""

        owner = self._managed_cleanup_owner()
        if owner is not None and owner.cleanup_pending:
            owner.retry_cleanup()
            return False
        if self._managed_lease_hold is not None:
            return False
        if not references:
            return True
        if owner is None:
            return False
        try:
            self._managed_lease_hold = await owner.acquire_lease_hold(
                self._sorted_managed_refs(references),
                managed_service,
            )
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            if not isinstance(error, Exception):
                raise
            return False
        return True

    def _transfer_managed_refs(self) -> bool:
        """Transfer exact handles to the app-owned off-loop cleanup lane."""

        hold = self._managed_lease_hold
        if hold is None:
            return True
        owner = self._managed_cleanup_owner()
        if owner is None:
            return False
        owner.request_lease_release(hold)
        self._managed_lease_hold = None
        return True

    def _transfer_managed_refs_to_publication(
        self,
    ) -> AudioCppManagedLeasePublication | None:
        """Move the current exact hold into the app/service publication lane."""

        hold = self._managed_lease_hold
        if hold is None:
            return None
        owner = self._managed_cleanup_owner()
        if owner is None:
            raise RuntimeError("audio.cpp model cleanup owner is unavailable")
        publication = owner.transfer_lease_hold_to_publication(hold)
        self._managed_lease_hold = None
        return publication

    @staticmethod
    def _audio_cpp_safe_package_name(package: AudioCppAcceptedPackage) -> str:
        """Reduce one private package root to a bounded printable basename."""

        name = Path(package.canonical_root).name
        safe = "".join(character for character in name if character.isprintable())
        safe = safe.strip()[:96]
        return safe or "Selected local package"

    def _audio_cpp_package_copy(self, package: AudioCppAcceptedPackage) -> str:
        """Project a path-safe, evidence-specific Guided package review row."""

        try:
            recipe = AUDIO_CPP_RECIPE_REGISTRY.validate_accepted(package)
        except ValueError:
            return (
                f"Model ID: {package.public_model_id}\n"
                "Compatibility: Review required. Remove this package and scan it "
                "again before saving."
            )
        capability_labels = {
            "tts": "Text-to-speech",
            "clone": "Voice cloning",
            "design": "Voice design",
        }
        capabilities = ", ".join(
            capability_labels[item] for item in recipe.capabilities
        )
        reference = recipe.reference_requirement.value.replace("_", " ").title()
        release = AUDIO_CPP_PINNED_RELEASE.removeprefix("release-")
        recovery = (
            "Recovery: voice setup is required before this package can generate. "
            "Keep it registered in the shared server and choose a text-ready "
            "default for the one-click sample, or use compatible voice setup when "
            "that flow is available."
            if recipe.reference_requirement is AudioCppReferenceRequirement.REQUIRED
            else "Recovery: if files change or this package no longer matches "
            "exactly, remove it and scan the package again."
        )
        return (
            f"{recipe.display_name}\n"
            f"Model ID: {package.public_model_id}\n"
            f"Family / variant: {recipe.family} / {recipe.package_variant}\n"
            f"Task capability: {capabilities} | Reference: {reference}\n"
            f"Compatibility: Exact reviewed recipe for audio.cpp {release}; "
            "backend support is Expected and is verified at runtime in Speech Lab.\n"
            f"Local package: {self._audio_cpp_safe_package_name(package)} (full path "
            "hidden)\n"
            "Loads lazily. A loaded model may remain in memory until Shutdown; "
            "switching models does not promise an unload.\n"
            f"{recovery}"
        )

    @staticmethod
    def _audio_cpp_guided_default_text_ready(values: Mapping[str, object]) -> bool:
        """Return whether the exact draft default needs no voice reference."""

        default_model_id = values.get("guided_default_model_id")
        raw_packages = values.get("guided_packages", ())
        if not isinstance(default_model_id, str) or not isinstance(
            raw_packages, (list, tuple)
        ):
            return False
        for raw in raw_packages:
            try:
                package = AudioCppAcceptedPackage.model_validate(raw)
                if package.public_model_id != default_model_id:
                    continue
                recipe = AUDIO_CPP_RECIPE_REGISTRY.validate_accepted(package)
            except ValueError:
                return False
            return bool(
                recipe.projection.task == "tts"
                and recipe.reference_requirement
                is not AudioCppReferenceRequirement.REQUIRED
            )
        return False

    @classmethod
    def _audio_cpp_handoff_label(cls, values: Mapping[str, object]) -> str:
        """Return a truthful no-work handoff for the current audio.cpp draft."""

        if (
            values.get("mode") != "managed"
            or values.get("managed_setup_source") != "guided"
        ):
            return "Open Speech Lab to test or refresh"
        if cls._audio_cpp_guided_default_text_ready(values):
            return "Open Speech Lab & Hear a Sample"
        return "Open Speech Lab to Test Connection"

    def _audio_cpp_guided_model_options(self) -> list[tuple[str, str]]:
        options: list[tuple[str, str]] = []
        for package in self._audio_cpp_guided_packages():
            try:
                recipe = AUDIO_CPP_RECIPE_REGISTRY.validate_accepted(package)
                label = f"{recipe.display_name} — {package.public_model_id}"
                if (
                    recipe.reference_requirement
                    is AudioCppReferenceRequirement.REQUIRED
                ):
                    label += " — voice setup required"
            except ValueError:
                label = f"Review required — {package.public_model_id}"
            options.append((label, package.public_model_id))
        return options

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
            "#settings-speech-default-status": (
                f"Default voice setup: {self._defaults_configuration_state().value}."
            ),
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
            "#settings-speech-provider-current-status": (
                f"Current status: {self._connection_readiness().connection.value}."
            ),
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

    def _connection_readiness(self):
        """Project saved-fingerprint evidence independently from local validity."""

        provider_id = self.configure_provider
        revision = self._provider_configuration_revisions.get(provider_id)
        catalog = SpeechTTSConnectionState.NOT_TESTED
        sample = SpeechTTSConnectionState.NOT_TESTED
        if revision is not None:
            try:
                fingerprint = build_provider_test_fingerprint(
                    self.original_state,
                    provider_id=provider_id,
                    saved_revision=revision,
                )
            except (TypeError, ValueError):
                pass
            else:
                catalog = self._provider_test_evidence.catalog_state(fingerprint)
                sample = self._provider_test_evidence.sample_state(fingerprint)
        return combine_tts_readiness(
            self._configuration_state(),
            catalog,
            sample,
        )

    def _connection_status_copy(self) -> str:
        readiness = self._connection_readiness()
        return (
            f"Provider connection: {readiness.connection.value} — "
            f"catalog {readiness.catalog.value}; sample {readiness.sample.value}"
        )

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
        try:
            self.query_one(
                "#settings-speech-status-provider-connection",
                Static,
            ).update(self._connection_status_copy())
        except QueryError:
            pass
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
            yield Static(
                "Voice profiles are managed in Lab > Speech > Voice Profiles "
                "— open Speech Lab, above, to get there. Per-character "
                "voices are assigned in the Roleplay character editor's "
                "Voice & Speech section, not here.",
                id="settings-speech-profile-surfaces-note",
                classes="settings-detail-row",
                markup=False,
            )

        yield _SpeechSettingsCard(
            self._compose_global_defaults_body,
            id=self._GLOBAL_DEFAULTS_CARD_ID,
            classes="settings-focus-card",
        )

        yield _SpeechSettingsCard(
            self._compose_provider_setup_body,
            id=self._PROVIDER_SETUP_CARD_ID,
            classes="settings-focus-card",
        )

        yield from self._compose_realtime_section()

        yield _SpeechSettingsCard(
            self._compose_inspector_body,
            id=self._INSPECTOR_CARD_ID,
        )

        cleanup_pending = self.audio_cpp_result_cleanup_pending()
        with Horizontal(id="settings-speech-actions", classes="settings-action-row"):
            yield _SpeechTTSCleanupActionButton(
                "Save",
                id="settings-speech-save",
                variant="primary",
                disabled=self._latest_request_id is not None or cleanup_pending,
            )
            yield _SpeechTTSCleanupActionButton(
                "Revert", id="settings-speech-revert", disabled=cleanup_pending
            )
            yield _SpeechTTSCleanupActionButton(
                "Restore Non-secret Defaults",
                id="settings-speech-restore-defaults",
                disabled=cleanup_pending,
            )
            yield Button("Open Speech Lab", id="settings-speech-open-lab-bottom")
        yield Static(
            self.result_text,
            id="settings-speech-save-result",
            classes="settings-status-row",
            markup=False,
        )

    #: task-15475: the three cards a dropdown change can invalidate, each
    #: rebuildable on its own (see `_replace_card_bodies`). Composing their
    #: contents from a method rather than inline in `compose()` is what makes
    #: a card-scoped rebuild possible at all -- the whole-panel
    #: `await self.recompose()` these handlers used to call destroyed ~200
    #: widgets (and every one of them, per the audit) to repaint one card.
    _GLOBAL_DEFAULTS_CARD_ID = "settings-speech-global-defaults"
    _PROVIDER_SETUP_CARD_ID = "settings-speech-provider-setup"
    _INSPECTOR_CARD_ID = "settings-speech-inspector"

    async def _replace_card_bodies(self, *card_ids: str) -> None:
        """Rebuild the named focus cards in place, keeping focus.

        The card-scoped alternative to ``await self.recompose()``, which
        destroyed and rebuilt every widget in the panel (~200 of them, per the
        2026-08-11 input-latency audit) to repaint one card -- and dropped
        focus entirely while doing it, so a keyboard user operating a Select
        was left with ``app.focused is None`` after every change.

        Delegates to each card's own ``recompose()`` rather than hand-rolling
        ``remove_children()``/``mount_all()``: a body generator that opens a
        container (``_compose_provider_setup_body`` does, via
        ``_compose_provider_form``) can only be consumed inside a real compose
        -- Textual's ``Widget.__enter__`` indexes ``app._compose_stacks[-1]``
        and raises ``IndexError`` off one. ``recompose()`` sets that stack up,
        awaits the removal before mounting (Textual's ``remove`` is deferred,
        so mounting early raises ``DuplicateIds`` on every re-used id -- the
        same trap ``speech_playground_pane._replace_provider_regions``
        documents), and being awaited here it is ordered against the caller,
        unlike a fire-and-forget ``refresh(recompose=True)``.

        Focus is restored only when it was inside this panel to begin with:
        restoring unconditionally would let a programmatic value change (a
        test, a catalog load) STEAL focus from wherever the user actually is.

        Args:
            card_ids: Ids of the focus cards to rebuild, in mount order.

        Returns:
            None.
        """
        focused = self.app.focused if self.is_mounted else None
        focus_token = (
            self._focus_token(focused)
            if focused is not None and self in focused.ancestors_with_self
            else None
        )
        for card_id in card_ids:
            try:
                card = self.query_one(f"#{card_id}", _SpeechSettingsCard)
            except QueryError:
                # The panel was torn down (category switch, screen change)
                # while this coroutine was suspended -- nothing to rebuild.
                continue
            await card.recompose()
        self._restore_focus(focus_token)

    def _compose_global_defaults_body(self) -> ComposeResult:
        """Yield the Global defaults card's children.

        Reads ``self.state.defaults`` only -- the Provider setup card is keyed
        off ``self.configure_provider`` and the Realtime card off
        ``self._realtime_draft``, so neither is affected by a change here.
        """
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
        yield Static("Global defaults", classes="destination-section")
        yield Static(
            f"Default voice setup: {self._defaults_configuration_state().value}.",
            id="settings-speech-default-status",
            classes="settings-status-row",
            markup=False,
        )
        yield self._default_profile_row()
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

    def _compose_provider_setup_body(self) -> ComposeResult:
        """Yield the Provider setup card's children.

        Keyed off ``self.configure_provider`` and
        ``self.state.providers`` -- independent of the Global defaults
        card above (Configure Provider does not change the default).
        """
        yield Static("Provider setup", classes="destination-section")
        yield Static(
            f"Task: set up {TTS_PROVIDER_LABELS[self.configure_provider]} for "
            "application-wide speech.",
            id="settings-speech-provider-task",
            classes="settings-status-row",
            markup=False,
        )
        yield Static(
            f"Current status: {self._connection_readiness().connection.value}.",
            id="settings-speech-provider-current-status",
            classes="settings-status-row",
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

    def _compose_inspector_body(self) -> ComposeResult:
        """Yield the Configuration inspector card's children.

        Reads BOTH the defaults and the configured provider, so it is
        rebuilt alongside whichever of the two cards above changed.
        """
        with Collapsible(
            title="Configuration details",
            collapsed=True,
            id="settings-speech-details",
        ):
            yield Static(
                "Global default selection: "
                f"{self._defaults_configuration_state().value} — effective source "
                f"{self.state.defaults_source.value}.",
                id="settings-speech-default-source",
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
            saved_revision = self._provider_configuration_revisions.get(
                self.configure_provider
            )
            applied_revision = self._provider_applied_configuration_revisions.get(
                self.configure_provider
            )
            runtime_revision = self._provider_runtime_revisions.get(
                self.configure_provider
            )
            yield Static(
                f"Owner ID: app_tts.{self.configure_provider}. Revisions: saved "
                f"{saved_revision if saved_revision is not None else 'none'}, applied "
                f"{applied_revision if applied_revision is not None else 'none'}, "
                f"runtime {runtime_revision if runtime_revision is not None else 'none'}.",
                id="settings-speech-owner-details",
                classes="settings-status-row",
                markup=False,
            )
            credential = self.state.credentials.get(self.configure_provider)
            if credential is not None:
                shadow_copy = (
                    " A saved local fallback is shadowed."
                    if credential.local_shadowed
                    else ""
                )
                yield Static(
                    "Credential provenance: "
                    f"{credential.source.value}; raw environment key "
                    f"{credential.environment_variable} (read-only when active)."
                    f"{shadow_copy}",
                    id="settings-speech-credential-provenance",
                    classes="settings-status-row",
                    markup=False,
                )
            environment_fields = GLOBAL_TTS_PROVIDER_ENVIRONMENT_FIELDS.get(
                self.configure_provider,
                {},
            )
            if environment_fields:
                raw_keys = ", ".join(sorted(environment_fields.values()))
                yield Static(
                    f"Raw environment keys (read-only when active): {raw_keys}.",
                    id="settings-speech-environment-keys",
                    classes="settings-status-row",
                    markup=False,
                )
            if self.configure_provider == "audio_cpp":
                configured_audio_choices = self._audio_cpp_choices()
                guided_binary_source = (
                    str(
                        self.state.providers["audio_cpp"].get(
                            "guided_binary_source", "manual"
                        )
                    )
                    .replace("_", " ")
                    .title()
                )
                yield Static(
                    f"Guided binary selection source: {guided_binary_source}.",
                    id=self._field_dom_id(
                        "audio_cpp",
                        "guided_binary_source",
                    ),
                    classes="settings-detail-row",
                    markup=False,
                )
                yield Static(
                    self._audio_cpp_observation_copy(configured_audio_choices),
                    id="settings-speech-audio-cpp-observation-provenance",
                    classes="settings-detail-row",
                    markup=False,
                )
                yield Static(
                    self._audio_cpp_draft_attribution_copy(),
                    id="settings-speech-audio-cpp-draft-attribution",
                    classes="settings-detail-row",
                    markup=False,
                )

        with Collapsible(
            title="Scope inspector",
            collapsed=True,
            id="settings-speech-scope-inspector",
        ):
            yield Static(
                f"Selected setup: {TTS_PROVIDER_LABELS[self.configure_provider]}",
                id="settings-speech-inspector-summary",
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
                self._connection_status_copy(),
                id="settings-speech-status-provider-connection",
                classes="settings-status-row",
                markup=False,
            )
            yield Static(
                "Ordinary Save validates and persists locally. Use Speech Lab for "
                "connection tests, discovery, generation, and playback.",
                classes="settings-detail-row",
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
        with Vertical(id="settings-speech-realtime", classes="settings-focus-card"):
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
                    _realtime_provider_options(draft.provider),
                    value=draft.provider or DEFAULT_REALTIME_PROVIDER,
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
                    classes=("settings-compact-input settings-speech-draft-field"),
                ),
                error=self._error("realtime", "model"),
            )
            yield self._row(
                "Voice (optional)",
                Input(
                    value=draft.voice,
                    id="settings-speech-realtime-voice",
                    placeholder="Provider default",
                    classes=("settings-compact-input settings-speech-draft-field"),
                ),
                error=self._error("realtime", "voice"),
            )
            yield self._row(
                "Idle timeout (minutes)",
                Input(
                    value=draft.idle_timeout_minutes,
                    id="settings-speech-realtime-idle-timeout-minutes",
                    placeholder=str(DEFAULT_REALTIME_IDLE_TIMEOUT_MINUTES),
                    classes=("settings-compact-input settings-speech-draft-field"),
                ),
                error=self._error("realtime", "idle_timeout_minutes"),
            )
            yield self._row(
                "Turn detection",
                Select(
                    _REALTIME_TURN_DETECTION_OPTIONS,
                    value=draft.turn_detection,
                    id="settings-speech-realtime-turn-detection",
                    allow_blank=False,
                    compact=True,
                    classes="settings-compact-select settings-speech-draft-field",
                ),
                classes="settings-select-row",
                error=self._error("realtime", "turn_detection"),
            )
            yield Static(
                "Semantic turn detection ends your turn from what you said, "
                "not from how long you paused. Server VAD is silence-gated: "
                "in a room with keyboard clatter or background noise it can "
                "end a turn mid-sentence, and the fragments transcribe as "
                "words you never said. Pick server VAD only if you want the "
                "two numbers below.",
                classes="settings-detail-row",
                markup=False,
            )
            yield self._row(
                "VAD threshold (0-1, server VAD only)",
                Input(
                    value=draft.vad_threshold,
                    id="settings-speech-realtime-vad-threshold",
                    placeholder="Provider default",
                    disabled=draft.turn_detection != "server_vad",
                    classes=("settings-compact-input settings-speech-draft-field"),
                ),
                error=self._error("realtime", "vad_threshold"),
            )
            yield self._row(
                "End-of-turn silence (ms, server VAD only)",
                Input(
                    value=draft.vad_silence_ms,
                    id="settings-speech-realtime-vad-silence-ms",
                    placeholder="Provider default",
                    disabled=draft.turn_detection != "server_vad",
                    classes=("settings-compact-input settings-speech-draft-field"),
                ),
                error=self._error("realtime", "vad_silence_ms"),
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
                mode_options = [("External server", "external")]
                configured_mode = self.state.providers[provider_id].get(
                    "mode", "external"
                )
                if _AUDIO_CPP_MANAGED_UI_SUPPORTED:
                    mode_options.append(("Managed local server", "managed"))
                elif configured_mode == "managed":
                    mode_options.append(
                        (
                            "Managed local server (unavailable on Windows)",
                            "managed",
                        )
                    )
                yield self._select(
                    provider_id,
                    "mode",
                    "Server mode",
                    mode_options,
                )
                if not _AUDIO_CPP_MANAGED_UI_SUPPORTED:
                    yield Static(
                        "Managed local server is unavailable on Windows until its "
                        "native process lifecycle is qualified. Use External server.",
                        id="settings-speech-audio-cpp-managed-platform-notice",
                        classes="settings-status-row",
                        markup=False,
                    )
                with Vertical(id="settings-speech-audio-cpp-external-fields"):
                    yield Static(
                        "External server: Chatbook connects to a server that you "
                        "start and own.",
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
                    yield Static(
                        "Generation sends submitted text to this configured server.",
                        id="settings-speech-audio_cpp-privacy-notice",
                        classes="settings-status-row",
                        markup=False,
                    )
                with Vertical(id="settings-speech-audio-cpp-managed-fields"):
                    yield Static(
                        "Managed local server: Chatbook will execute the selected "
                        "file only after a deliberate Speech Lab, Console, or "
                        "Roleplay speech action. Review and trust the binary first.",
                        id="settings-speech-audio-cpp-managed-trust",
                        classes="settings-status-row",
                        markup=False,
                    )
                    yield self._select(
                        provider_id,
                        "managed_setup_source",
                        "Managed setup source",
                        [
                            ("Guided setup — no JSON editing", "guided"),
                            ("Use an existing server.json", "user_json"),
                        ],
                    )
                    with Vertical(id="settings-speech-audio-cpp-manual-json-fields"):
                        yield Static(
                            "Manual server.json keeps the existing audio.cpp workflow. "
                            "Chatbook validates but never rewrites your file.",
                            classes="settings-detail-row",
                            markup=False,
                        )
                        yield self._path(
                            provider_id,
                            "managed_binary_path",
                            "audiocpp_server binary path",
                            placeholder=(
                                "Choose a prebuilt audiocpp_server executable"
                            ),
                        )
                        yield Button(
                            "Use detected audiocpp_server",
                            id="settings-speech-audio-cpp-manual-use-detected",
                            compact=True,
                            tooltip=(
                                "Look for audiocpp_server on PATH and update only "
                                "this unsaved manual draft."
                            ),
                        )
                        yield self._path(
                            provider_id,
                            "managed_server_json_path",
                            "server.json path",
                            placeholder="Choose an existing server.json",
                        )
                        yield Static(
                            "Managed mode requires server.json to bind exactly to "
                            "127.0.0.1 with an explicit port. The server.json folder "
                            "becomes the child working directory, so relative paths "
                            "in that file resolve from that folder. Chatbook never "
                            "edits it.",
                            id="settings-speech-audio-cpp-managed-json-help",
                            classes="settings-detail-row",
                            markup=False,
                        )

                    with Vertical(id="settings-speech-audio-cpp-guided-fields"):
                        yield Static(
                            "Guided setup creates a private runtime configuration "
                            "from reviewed model packages. Install audio.cpp "
                            "separately; Chatbook does not download or run it while "
                            "you edit or Save Settings.",
                            id="settings-speech-audio-cpp-guided-intro",
                            classes="settings-detail-row",
                            markup=False,
                        )
                        yield self._path(
                            provider_id,
                            "guided_binary_path",
                            "audiocpp_server binary path",
                            placeholder=(
                                "Detect or choose a separately installed executable"
                            ),
                        )
                        yield Button(
                            "Use detected audiocpp_server",
                            id="settings-speech-audio-cpp-use-detected",
                            compact=True,
                            tooltip=(
                                "Look for audiocpp_server on PATH and update only "
                                "this unsaved Guided draft."
                            ),
                        )
                        yield Button(
                            "Add local package…",
                            id="settings-speech-audio-cpp-guided-add-package",
                            compact=True,
                            disabled=self.audio_cpp_result_cleanup_pending(),
                            tooltip=(
                                "Choose one package directory to scan against the "
                                "reviewed audio.cpp recipes."
                            ),
                        )
                        yield Button(
                            "Open Model Library…",
                            id="settings-speech-audio-cpp-open-model-library",
                            compact=True,
                            disabled=self.audio_cpp_result_cleanup_pending(),
                            tooltip=(
                                "Browse reviewed audio.cpp model packages without "
                                "saving this Settings draft."
                            ),
                        )
                        with Vertical(
                            id=self._field_dom_id(provider_id, "guided_packages"),
                            classes="settings-speech-audio-cpp-packages",
                        ):
                            packages = self._audio_cpp_guided_packages()
                            if not packages:
                                yield Static(
                                    "No reviewed model packages yet. Add a local "
                                    "Supertonic or PocketTTS package to continue.",
                                    classes=(
                                        "settings-detail-row "
                                        "settings-speech-audio-cpp-package-copy"
                                    ),
                                    markup=False,
                                )
                            for package in packages:
                                with Vertical(
                                    classes="settings-speech-audio-cpp-package-row"
                                ):
                                    yield Static(
                                        self._audio_cpp_package_copy(package),
                                        classes=(
                                            "settings-detail-row "
                                            "settings-speech-audio-cpp-package-copy"
                                        ),
                                        markup=False,
                                    )
                                    yield Button(
                                        "Remove package",
                                        id=(
                                            "settings-speech-audio-cpp-guided-"
                                            f"package-remove-{package.package_uuid}"
                                        ),
                                        compact=True,
                                        disabled=(
                                            self.audio_cpp_result_cleanup_pending()
                                        ),
                                        variant="warning",
                                        classes=(
                                            "settings-speech-audio-cpp-package-remove"
                                        ),
                                        tooltip=(
                                            "Remove this package from the unsaved "
                                            "Guided draft. No files are deleted."
                                        ),
                                    )
                            yield self._error(provider_id, "guided_packages")
                        model_options = self._audio_cpp_guided_model_options()
                        selected_model = self.state.providers[provider_id].get(
                            "guided_default_model_id"
                        )
                        if not isinstance(selected_model, str):
                            selected_model = Select.BLANK
                        yield self._row(
                            "Default Guided model",
                            Select(
                                model_options
                                or [("Add a reviewed package first", Select.BLANK)],
                                value=selected_model,
                                id=self._field_dom_id(
                                    provider_id, "guided_default_model_id"
                                ),
                                allow_blank=False,
                                compact=True,
                                disabled=self.audio_cpp_result_cleanup_pending(),
                                classes=(
                                    "settings-compact-select settings-speech-field "
                                    "settings-speech-draft-field"
                                ),
                            ),
                            classes="settings-select-row",
                            error=self._error(provider_id, "guided_default_model_id"),
                        )
                        yield self._select(
                            provider_id,
                            "guided_backend_preference",
                            "Compute backend",
                            [
                                ("Auto — choose a reviewed compatible backend", "auto"),
                                ("CPU", "cpu"),
                                ("Metal", "metal"),
                                ("CUDA", "cuda"),
                                ("Vulkan", "vulkan"),
                                ("HIP", "hip"),
                            ],
                        )
                        yield Static(
                            "One lazy audio.cpp child serves every accepted model. "
                            "A model may stay resident in memory until Shutdown.",
                            id="settings-speech-audio-cpp-guided-memory-note",
                            classes="settings-status-row",
                            markup=False,
                        )

                yield Button(
                    self._audio_cpp_handoff_label(self.state.providers[provider_id]),
                    id="settings-speech-audio-cpp-open-lab",
                    compact=True,
                    tooltip=(
                        "Open the TTS Playground; Settings itself never contacts "
                        "the configured server."
                    ),
                )
                with Collapsible(
                    title="Advanced lifecycle and safety limits",
                    collapsed=True,
                    id="settings-speech-audio-cpp-advanced",
                ):
                    with Vertical(
                        id="settings-speech-audio-cpp-managed-lifecycle-fields"
                    ):
                        yield self._input(
                            provider_id,
                            "managed_startup_timeout_seconds",
                            "Managed startup timeout (seconds)",
                            default=_AUDIO_CPP_MANAGED_DEFAULTS[
                                "managed_startup_timeout_seconds"
                            ],
                        )
                        yield self._input(
                            provider_id,
                            "managed_health_check_interval_seconds",
                            "Managed health interval (seconds)",
                            default=_AUDIO_CPP_MANAGED_DEFAULTS[
                                "managed_health_check_interval_seconds"
                            ],
                        )
                        yield self._input(
                            provider_id,
                            "managed_termination_grace_seconds",
                            "Managed termination grace (seconds)",
                            default=_AUDIO_CPP_MANAGED_DEFAULTS[
                                "managed_termination_grace_seconds"
                            ],
                        )
                    with Vertical(
                        id="settings-speech-audio-cpp-guided-advanced-fields"
                    ):
                        yield self._input(
                            provider_id,
                            "guided_device",
                            "Guided device index (blank = backend default)",
                            default="",
                        )
                        yield self._input(
                            provider_id,
                            "guided_threads",
                            "Guided CPU threads (blank = server default)",
                            default="",
                        )
                        yield self._input(
                            provider_id,
                            "guided_max_request_body_bytes",
                            "Guided max request body bytes",
                            default=_AUDIO_CPP_MANAGED_DEFAULTS[
                                "guided_max_request_body_bytes"
                            ],
                        )
                        yield self._input(
                            provider_id,
                            "guided_busy_timeout_ms",
                            "Guided busy timeout (milliseconds)",
                            default=_AUDIO_CPP_MANAGED_DEFAULTS[
                                "guided_busy_timeout_ms"
                            ],
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
                yield self._select(
                    provider_id,
                    "authentication_mode",
                    "Authentication",
                    [("API key", "api_key"), ("None", "none")],
                )
                yield self._row(
                    "Endpoint preset",
                    Horizontal(
                        Button(
                            "Use Official OpenAI",
                            id="settings-speech-openai-official-preset",
                            compact=True,
                            tooltip=(
                                "Use the official OpenAI speech endpoint with "
                                "API key authentication."
                            ),
                        ),
                        classes="settings-action-row",
                    ),
                )
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
            default_profile = self.query_one(
                "#settings-speech-default-profile", Select
            ).value
            if isinstance(default_profile, str):
                self.state.defaults.default_profile_id = (
                    default_profile if default_profile else None
                )
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
        active_audio_cpp_mode = values.get("mode", "external")
        active_audio_cpp_setup_source = values.get(
            "managed_setup_source",
            "user_json",
        )
        if self.configure_provider == "audio_cpp":
            try:
                selected_mode = self.query_one(
                    "#settings-speech-audio_cpp-mode", Select
                ).value
            except QueryError:
                selected_mode = active_audio_cpp_mode
            if isinstance(selected_mode, str):
                active_audio_cpp_mode = selected_mode
                values["mode"] = selected_mode
            try:
                selected_source = self.query_one(
                    "#settings-speech-audio_cpp-managed-setup-source",
                    Select,
                ).value
            except QueryError:
                selected_source = active_audio_cpp_setup_source
            if isinstance(selected_source, str):
                active_audio_cpp_setup_source = selected_source
                values["managed_setup_source"] = selected_source
        for field_id in GLOBAL_TTS_PROVIDER_FIELD_IDS[self.configure_provider]:
            if field_id == "credential":
                continue
            if self.configure_provider == "audio_cpp":
                if field_id == "mode":
                    continue
                if field_id == "base_url" and active_audio_cpp_mode != "external":
                    continue
                if active_audio_cpp_mode != "managed" and (
                    field_id in _AUDIO_CPP_MANAGED_FIELD_IDS
                    or field_id in _AUDIO_CPP_MANUAL_FIELD_IDS
                    or field_id in _AUDIO_CPP_GUIDED_FIELD_IDS
                    or field_id == "managed_setup_source"
                ):
                    continue
                if (
                    field_id in _AUDIO_CPP_MANUAL_FIELD_IDS
                    and active_audio_cpp_setup_source != "user_json"
                ):
                    continue
                if (
                    field_id in _AUDIO_CPP_GUIDED_FIELD_IDS
                    and active_audio_cpp_setup_source != "guided"
                ):
                    continue
            selector = f"#{self._field_dom_id(self.configure_provider, field_id)}"
            try:
                widget = self.query_one(selector)
            except QueryError:
                continue
            if isinstance(widget, Input):
                if field_id in {
                    "guided_device",
                    "guided_threads",
                    "guided_max_request_body_bytes",
                    "guided_busy_timeout_ms",
                }:
                    if not widget.value.strip() and field_id in {
                        "guided_device",
                        "guided_threads",
                    }:
                        values[field_id] = None
                    else:
                        try:
                            values[field_id] = int(widget.value)
                        except ValueError:
                            values[field_id] = widget.value
                else:
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
            enabled_widget = self.query_one("#settings-speech-realtime-enabled", Switch)
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
        try:
            mode_widget = self.query_one(
                "#settings-speech-realtime-turn-detection", Select
            )
            threshold_widget = self.query_one(
                "#settings-speech-realtime-vad-threshold", Input
            )
            silence_widget = self.query_one(
                "#settings-speech-realtime-vad-silence-ms", Input
            )
        except QueryError:
            return
        if isinstance(mode_widget.value, str):
            self._realtime_draft.turn_detection = mode_widget.value
        self._realtime_draft.vad_threshold = threshold_widget.value
        self._realtime_draft.vad_silence_ms = silence_widget.value

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

        # `default_profile_id` is a distinct precedence rung, deliberately not
        # part of `TTSPreferencesSnapshot` (see build_global_speech_tts_save_
        # proposal's own comment) -- `.snapshot()` above cannot see it, so it
        # must be compared explicitly or picking a default profile alone would
        # look like "no changes" and never enable Save.
        if (
            self.state.defaults.default_profile_id
            != self.original_state.defaults.default_profile_id
        ):
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
        self._sync_audio_cpp_handoff(draft_modified=is_modified)
        self._refresh_status_rows()
        self.post_message(
            self.DraftModified(
                is_modified,
                self.state,
                self.original_state,
                self.configure_provider,
                self.draft_snapshot(),
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
        field_selector = (
            f"#{self._field_dom_id(error.provider_id, error.field_id)}"
            if error.provider_id != "defaults"
            else {
                "default_profile_id": "#settings-speech-default-profile",
                "provider_id": "#settings-speech-default-provider",
                "model_mode": "#settings-speech-model-policy",
                "default_model": "#settings-speech-model-value",
                "voice_mode": "#settings-speech-voice-policy",
                "default_voice": "#settings-speech-voice-value",
                "response_format": "#settings-speech-output-format",
                "default_speed": "#settings-speech-speed",
            }.get(error.field_id, "#settings-speech-default-provider")
        )
        error_selector = f"{field_selector}-error"
        selector = (
            "#settings-speech-audio-cpp-guided-add-package"
            if error.provider_id == "audio_cpp" and error.field_id == "guided_packages"
            else field_selector
        )
        try:
            error_widget = self.query_one(error_selector, Static)
            error_widget.update(str(error))
            error_widget.add_class("settings-speech-field-error-visible")
        except QueryError:
            pass
        try:
            field = self.query_one(selector)
            for ancestor in field.ancestors:
                if isinstance(ancestor, Collapsible):
                    ancestor.collapsed = False
            field.focus()
            field.scroll_visible(animate=False)
        except QueryError:
            pass
        self._set_result(str(error), severity="error")

    def request_save(self) -> int | None:
        """Validate locally and post one atomic ordinary-save proposal."""
        if self._fence_audio_cpp_result_cleanup():
            return None
        self._collect_visible_state()
        if self._latest_request_id is not None:
            self._set_result("A global Speech & TTS save is already in progress.")
            return None
        if self.configure_provider == "openai":
            required_confirmation = required_openai_plaintext_confirmation_fingerprint(
                self.state
            )
            current_confirmation = self.state.openai_plaintext_confirmation
            if required_confirmation is not None and (
                current_confirmation is None
                or current_confirmation.origin_fingerprint != required_confirmation
            ):
                self.app.push_screen(
                    _OpenAINoneHTTPConfirmationModal(),
                    lambda confirmed: self._openai_plaintext_confirmation_result(
                        required_confirmation,
                        confirmed,
                    ),
                )
                return None
        self._clear_validation_errors()
        try:
            proposal = build_global_speech_tts_save_proposal(
                self.original_state,
                self.state,
                configure_provider=self.configure_provider,
            )
            if "audio_cpp" in proposal.changed_provider_ids:
                validate_audio_cpp_managed_settings(self.state.providers["audio_cpp"])
            defaults_changed = (
                proposal.preferences != self.original_state.defaults.snapshot()
                # `default_profile_id` lives outside `TTSPreferencesSnapshot`
                # (a distinct precedence rung -- see has_unsaved_changes and
                # build_global_speech_tts_save_proposal), so the snapshot
                # comparison above is blind to it. Without this,
                # `_pending_saved_defaults` below would stay `None` on a
                # default-profile-only save, and `original_state.defaults`
                # would never learn the persisted value -- leaving the panel
                # stuck reporting unsaved changes after a successful save.
                or self.state.defaults.default_profile_id
                != self.original_state.defaults.default_profile_id
            )
            if (
                self.configure_provider == "openai"
                and "OPENAI_NONE_HTTP_CONFIRMATION" in proposal.delete_setting_keys
            ):
                self.state.openai_plaintext_confirmation = None
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
        guided_packages: tuple[AudioCppAcceptedPackage, ...] = ()
        guided_lease_refs: set[ArtifactRef] = set()
        if "audio_cpp" in proposal.changed_provider_ids:
            saved_audio_cpp = proposal.settings.get("audio_cpp")
            if not isinstance(saved_audio_cpp, Mapping):
                raise AssertionError("validated audio.cpp settings are unavailable")
            guided_lease_refs = self._managed_refs_from_values(
                self.original_state.providers["audio_cpp"]
            ) | self._managed_refs_from_values(saved_audio_cpp)
            if (
                saved_audio_cpp.get("mode") == "managed"
                and saved_audio_cpp.get("managed_setup_source") == "guided"
            ):
                audio_cpp = AudioCppSettingsConfig.from_mapping(saved_audio_cpp)
                guided_packages = audio_cpp.guided_packages

        if (
            not defaults_changed
            and not proposal.settings
            and not proposal.delete_setting_keys
            and not realtime_changed
        ):
            self._set_result("No global Speech & TTS changes to save.")
            return None

        provider_save_required = bool(
            defaults_changed or proposal.settings or proposal.delete_setting_keys
        )
        if realtime_payload is not None and not (guided_packages or guided_lease_refs):
            if not self._persist_realtime_draft(realtime_payload):
                self._set_result(
                    "Realtime engine settings were not saved.",
                    severity="error",
                )
                return None
            self._realtime_original = replace(realtime_payload.persisted_draft)

        if not provider_save_required:
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
        provider_configuration_changed = (
            self.configure_provider in proposal.changed_provider_ids
        )
        self._pending_saved_provider_id = (
            self.configure_provider if provider_configuration_changed else None
        )
        self._pending_saved_provider_values = (
            deepcopy(self.state.providers[self.configure_provider])
            if provider_configuration_changed
            else None
        )
        self._pending_commit_defaults_after_handoff = bool(
            defaults_changed
            and provider_configuration_changed
            and proposal.preferences.provider_id == self.configure_provider
        )
        self._pending_saved_openai_confirmation = (
            deepcopy(self.state.openai_plaintext_confirmation)
            if self.configure_provider == "openai"
            and (proposal.settings or proposal.delete_setting_keys)
            else None
        )
        self._pending_saved_openai_confirmation_cleanup_needed = (
            False
            if "OPENAI_NONE_HTTP_CONFIRMATION" in proposal.settings
            or "OPENAI_NONE_HTTP_CONFIRMATION" in proposal.delete_setting_keys
            else None
        )
        if guided_packages or guided_lease_refs:
            self._set_result("Rechecking reviewed model packages locally before Save…")
            self.run_worker(
                self._revalidate_guided_save(
                    request_id=request_id,
                    packages=guided_packages,
                    lease_refs=guided_lease_refs,
                    proposal=proposal,
                    realtime_payload=realtime_payload,
                ),
                group=_AUDIO_CPP_PACKAGE_SAVE_VALIDATION_GROUP,
                exclusive=True,
                exit_on_error=False,
            )
        else:
            self._post_settings_save(request_id, proposal)
        return request_id

    def _post_settings_save(
        self,
        request_id: int,
        proposal: GlobalSpeechTTSSaveProposal,
    ) -> None:
        """Post one already-validated immutable Settings proposal."""

        if request_id != self._latest_request_id or not self.is_mounted:
            return
        self._set_result(
            "Saving global Speech & TTS settings locally…",
            severity="information",
        )
        publication_lease = self._transfer_managed_refs_to_publication()
        try:
            event = STTSSettingsSaveEvent(
                proposal.settings,
                delete_setting_keys=proposal.delete_setting_keys,
                preferences=proposal.preferences,
                request_id=request_id,
                reply_to=self,
                commit_defaults_after_handoff=(
                    self._pending_commit_defaults_after_handoff
                ),
                publication_lease=publication_lease,
            )
            self.app.post_message(event)
        except BaseException:
            if publication_lease is not None:
                publication_lease.abandon()
            raise

    async def _revalidate_guided_save(
        self,
        *,
        request_id: int,
        packages: tuple[AudioCppAcceptedPackage, ...],
        lease_refs: set[ArtifactRef],
        proposal: GlobalSpeechTTSSaveProposal,
        realtime_payload: _RealtimeSavePayload | None,
    ) -> None:
        """Recheck exact local package identities off the Textual message loop."""

        if not await self._acquire_managed_refs(lease_refs):
            self._abort_pending_save(
                request_id,
                "Managed model packages are busy or cleanup is still pending.",
            )
            return
        try:
            await revalidate_audio_cpp_guided_packages(packages)
        except asyncio.CancelledError:
            self._transfer_managed_refs()
            raise
        except AudioCppGuidedLaunchError:
            self._transfer_managed_refs()
            self._abort_pending_save(
                request_id,
                GlobalSpeechTTSValidationError(
                    "audio_cpp",
                    "guided_packages",
                    "A reviewed model package changed. Remove it and scan the "
                    "package again before saving.",
                ),
            )
            return
        except BaseException:
            self._transfer_managed_refs()
            raise

        if request_id != self._latest_request_id or not self.is_mounted:
            self._transfer_managed_refs()
            return
        if realtime_payload is not None:
            if not self._persist_realtime_draft(realtime_payload):
                self._abort_pending_save(
                    request_id,
                    "Realtime engine settings were not saved.",
                )
                return
            self._realtime_original = replace(realtime_payload.persisted_draft)
        try:
            self._post_settings_save(request_id, proposal)
        except BaseException:
            self._transfer_managed_refs()
            raise

    def _abort_pending_save(
        self,
        request_id: int,
        failure: GlobalSpeechTTSValidationError | str,
    ) -> None:
        """Release one local pre-persistence failure without posting a save."""

        if request_id != self._latest_request_id:
            return
        cleanup_succeeded = self._transfer_managed_refs()
        focus_id = self._completion_focus_id(self._pending_focus_control_id)
        leave_waiter = self._leave_save_waiters.pop(request_id, None)
        self._latest_request_id = None
        self._pending_focus_control_id = None
        self._pending_displaced_focus_control_id = None
        self._pending_focus_moved_after_displacement = False
        self._pending_credential_mutation = None
        self._pending_saved_defaults = None
        self._pending_saved_provider_id = None
        self._pending_saved_provider_values = None
        self._pending_commit_defaults_after_handoff = False
        self._pending_saved_openai_confirmation = None
        self._pending_saved_openai_confirmation_cleanup_needed = None
        self._set_save_pending(False)
        if not cleanup_succeeded:
            self._set_result(
                "Managed model package cleanup is still pending.",
                severity="error",
            )
        elif isinstance(failure, GlobalSpeechTTSValidationError):
            self._show_validation_error(failure)
        else:
            self._set_result(failure, severity="error")
            self.call_later(self._restore_focus, focus_id)
        self._refresh_status_rows()
        if leave_waiter is not None and not leave_waiter.done():
            leave_waiter.set_result(False)

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
        if idle_minutes.is_integer():
            # A whole number of minutes belongs in the user's config file
            # as `5`, not `5.0` (final review M5). `float` stays for a
            # deliberate fractional value like 2.5.
            idle_minutes = int(idle_minutes)
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
            "turn_detection": self._realtime_draft.turn_detection,
        }
        removed: list[str] = []
        voice = self._realtime_draft.voice.strip()
        if voice:
            realtime_section["voice"] = voice
        else:
            removed.append("voice")

        # The provider REJECTS these under semantic turn detection
        # (`unknown_parameter`, live-confirmed), so semantic mode deletes
        # them outright rather than leaving stale numbers in config for a
        # later reader to hand over.
        if self._realtime_draft.turn_detection == "server_vad":
            threshold = self._validated_optional_number(
                self._realtime_draft.vad_threshold,
                field="vad_threshold",
                message="VAD threshold must be a number between 0 and 1.",
                cast=float,
                low=0.0,
                high=1.0,
            )
            silence = self._validated_optional_number(
                self._realtime_draft.vad_silence_ms,
                field="vad_silence_ms",
                message="End-of-turn silence must be a positive whole "
                "number of milliseconds.",
                cast=int,
                low=1,
                high=None,
            )
            if threshold is None:
                removed.append("vad_threshold")
            else:
                realtime_section["vad_threshold"] = threshold
            if silence is None:
                removed.append("vad_silence_ms")
            else:
                realtime_section["vad_silence_ms"] = silence
        else:
            removed.extend(("vad_threshold", "vad_silence_ms"))
        delete_keys: dict[str, tuple[str, ...]] = (
            {"realtime": tuple(removed)} if removed else {}
        )
        dictation_section = {"handsfree_engine": self._realtime_draft.handsfree_engine}
        return _RealtimeSavePayload(
            section_values={
                "realtime": realtime_section,
                "dictation": dictation_section,
            },
            delete_keys=delete_keys,
            persisted_draft=replace(self._realtime_draft),
        )

    @staticmethod
    def _validated_optional_number(
        raw: str,
        *,
        field: str,
        message: str,
        cast: Any,
        low: float,
        high: float | None,
    ) -> Any:
        """Validate one optional numeric knob; blank means "unset".

        Raises the same `GlobalSpeechTTSValidationError` the rest of this
        panel uses, so a bad value refuses the WHOLE Save with an inline
        error rather than being silently dropped -- a silently dropped
        threshold reads to the user as "the setting does nothing".
        """
        text = (raw or "").strip()
        if not text:
            return None
        try:
            value = cast(text)
        except (TypeError, ValueError):
            raise GlobalSpeechTTSValidationError("realtime", field, message)
        if value < low or (high is not None and value > high):
            raise GlobalSpeechTTSValidationError("realtime", field, message)
        return value

    @on(Select.Changed, "#settings-speech-realtime-turn-detection")
    def handle_realtime_turn_detection_changed(self, event: Select.Changed) -> None:
        """Enable the server_vad-only numbers only in server_vad mode.

        Disabled rather than hidden: the values stay visible, so a user
        who switches modes can see what they had configured instead of
        wondering where it went.
        """
        server_vad = event.value == "server_vad"
        for widget_id in (
            "#settings-speech-realtime-vad-threshold",
            "#settings-speech-realtime-vad-silence-ms",
        ):
            try:
                self.query_one(widget_id, Input).disabled = not server_vad
            except QueryError:
                continue

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
        self._pending_commit_defaults_after_handoff = False
        self._pending_saved_openai_confirmation = None
        self._pending_saved_openai_confirmation_cleanup_needed = None
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
            staged_pending = (
                publication_state == "pending"
                and provider_id in result.staged_provider_ids
            )
            if result.failure_phase == "cache_reload":
                runtime_state = SpeechTTSRuntimeState.UNAVAILABLE
                category = SpeechTTSDiagnosticCategory.CONFIGURATION
                recovery = SpeechTTSNavigationIntent.CONFIGURE
            elif staged_pending:
                runtime_state = SpeechTTSRuntimeState.NOT_CHECKED
                category = SpeechTTSDiagnosticCategory.CONFIGURATION
                recovery = SpeechTTSNavigationIntent.TEST
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
            if publication_state != "pending" or staged_pending:
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
        saved_openai_confirmation = self._pending_saved_openai_confirmation
        saved_openai_confirmation_cleanup_needed = (
            self._pending_saved_openai_confirmation_cleanup_needed
        )
        if (
            result.defaults_activation_status == "activation_not_ready"
            and "pending" in result.provider_statuses.values()
            and saved_defaults is not None
        ):
            self._deferred_default_activation_drafts[result.request_id] = (
                deepcopy(saved_defaults),
                saved_provider_id,
                deepcopy(saved_provider_values),
            )
        self._pending_credential_mutation = None
        self._pending_saved_defaults = None
        self._pending_saved_provider_id = None
        self._pending_saved_provider_values = None
        self._pending_commit_defaults_after_handoff = False
        self._pending_saved_openai_confirmation = None
        self._pending_saved_openai_confirmation_cleanup_needed = None
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
            self._transfer_managed_refs()
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
            if saved_defaults is not None and result.defaults_activated is not False:
                self.original_state.defaults = saved_defaults
                self.state.defaults_source = GlobalSpeechTTSEffectiveSource.SAVED_LOCAL
                self.original_state.defaults_source = (
                    GlobalSpeechTTSEffectiveSource.SAVED_LOCAL
                )
            if (
                saved_provider_id is not None
                and saved_provider_values is not None
                and result.defaults_activated is not False
            ):
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
            if saved_openai_confirmation_cleanup_needed is not None:
                self.original_state.openai_plaintext_confirmation = (
                    saved_openai_confirmation
                )
                self.state.openai_plaintext_confirmation_cleanup_needed = (
                    saved_openai_confirmation_cleanup_needed
                )
                self.original_state.openai_plaintext_confirmation_cleanup_needed = (
                    saved_openai_confirmation_cleanup_needed
                )

        self._record_save_runtime_statuses(
            result,
            saved_provider_id=saved_provider_id,
        )
        for provider_id in result.provider_statuses:
            if provider_id in result.staged_provider_ids:
                self._provider_runtime_request_ids.pop(provider_id, None)
            else:
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
        if result.defaults_activation_status == "rollback_failed":
            result_copy = (
                "Defaults were saved, but rollback failed. Runtime still uses the "
                "previous default; restart may use the new default. Retry to reconcile."
            )
        elif (
            result.defaults_activation_status == "activation_not_ready"
            and "pending" in result.provider_statuses.values()
        ):
            result_copy = (
                "Saved locally; default activation is waiting for the TTS handoff. "
                "Keep this draft open until activation completes."
            )
        elif result.defaults_activated is False:
            result_copy = (
                "Saved, activation failed. The previous default remains active; "
                "retry after the provider is available."
            )
        elif cache_reload_failed:
            result_copy = (
                "Saved locally, but the runtime configuration cache reload failed; "
                f"restart or retry. Provider handoff: {handoff}."
            )
        elif (
            saved_provider_id == "audio_cpp"
            and result.provider_statuses.get("audio_cpp") == "pending"
            and "audio_cpp" in result.staged_provider_ids
        ):
            desired_mode = (
                saved_provider_values.get("mode", "external")
                if saved_provider_values is not None
                else "external"
            )
            guided = bool(
                desired_mode == "managed"
                and saved_provider_values is not None
                and saved_provider_values.get("managed_setup_source") == "guided"
            )
            guided_text_ready = bool(
                guided
                and saved_provider_values is not None
                and self._audio_cpp_guided_default_text_ready(saved_provider_values)
            )
            handoff_copy = (
                "Open Speech Lab & Hear a Sample to apply the saved Guided settings."
                if guided_text_ready
                else "Open Speech Lab to test the saved Guided settings. Voice "
                "setup is required before the selected default can generate."
                if guided
                else "Open Speech Lab to apply the saved Managed settings."
                if desired_mode == "managed"
                else "Open Speech Lab to apply External mode and stop any active "
                "managed server."
            )
            result_copy = (
                "Configuration saved — ready to test. The active audio.cpp "
                f"configuration remains unchanged. {handoff_copy}"
                if guided
                else "Saved locally; the active audio.cpp configuration remains "
                f"unchanged. {handoff_copy}"
            )
        else:
            result_copy = f"Saved locally. Runtime reconfiguration: {handoff}."
        self._set_result(
            result_copy,
            severity=(
                "warning"
                if result.defaults_activated is False
                or cache_reload_failed
                or "unavailable" in result.provider_statuses.values()
                else "information"
            ),
        )
        self._announce_draft_state()
        self._transfer_managed_refs()
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
            defaults_activated=result.defaults_activated,
            defaults_activation_status=result.defaults_activation_status,
        )
        deferred = self._deferred_default_activation_drafts.pop(
            result.request_id,
            None,
        )
        if result.defaults_activation_status == "committed" and deferred is not None:
            saved_defaults, saved_provider_id, saved_provider_values = deferred
            self.original_state.defaults = deepcopy(saved_defaults)
            self.state.defaults_source = GlobalSpeechTTSEffectiveSource.SAVED_LOCAL
            self.original_state.defaults_source = (
                GlobalSpeechTTSEffectiveSource.SAVED_LOCAL
            )
            if saved_provider_id is not None and saved_provider_values is not None:
                self.original_state.providers[saved_provider_id] = deepcopy(
                    saved_provider_values
                )
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
            bounded_result,
            saved_provider_id=None,
        )
        handoff = ", ".join(
            f"{TTS_PROVIDER_LABELS[provider_id]}: {status}"
            for provider_id, status in matching_statuses.items()
        )
        result_copy = f"Saved locally. Runtime reconfiguration completed: {handoff}."
        if result.defaults_activation_status == "committed":
            result_copy = (
                f"Saved locally. Runtime reconfiguration completed: {handoff}. "
                "Default activation completed."
            )
        elif result.defaults_activation_status == "rollback_failed":
            result_copy = (
                "Defaults were saved, but rollback failed. Runtime still uses the "
                "previous default; restart may use the new default. Retry to reconcile."
            )
        self._set_result(
            result_copy,
            severity=(
                "error"
                if "unavailable" in matching_statuses.values()
                or result.defaults_activation_status == "rollback_failed"
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

    def _openai_plaintext_confirmation_result(
        self,
        origin_fingerprint: str,
        confirmed: bool,
    ) -> None:
        """Accept consent only while the same normalized origin still owns it."""

        if not confirmed:
            self._set_result("Unauthenticated HTTP settings were not saved.")
            return
        self._collect_visible_state()
        required = required_openai_plaintext_confirmation_fingerprint(self.state)
        if required != origin_fingerprint:
            self._set_result(
                "The OpenAI-compatible destination changed. Review and save again.",
                severity="warning",
            )
            return
        self.state.openai_plaintext_confirmation = OpenAIPlaintextConfirmation(
            origin_fingerprint
        )
        self.state.openai_plaintext_confirmation_cleanup_needed = False
        self.request_save()

    @on(Button.Pressed, "#settings-speech-openai-official-preset")
    def handle_openai_official_preset(self, event: Button.Pressed) -> None:
        """Apply the official endpoint only after restoring API-key auth."""

        event.stop()
        authentication = self.query_one(
            "#settings-speech-openai-authentication-mode",
            Select,
        )
        endpoint = self.query_one("#settings-speech-openai-base-url", Input)
        self.state.providers["openai"]["authentication_mode"] = "api_key"
        authentication.value = "api_key"
        self.state.openai_plaintext_confirmation = None
        self.state.providers["openai"]["base_url"] = (
            "https://api.openai.com/v1/audio/speech"
        )
        endpoint.value = "https://api.openai.com/v1/audio/speech"
        self._announce_draft_state()

    def _path_picker_result(self, target_selector: str, path: Path | None) -> None:
        if path is None:
            return
        try:
            self.query_one(target_selector, Input).value = str(path)
        except QueryError:
            return
        if target_selector == "#settings-speech-audio_cpp-guided-binary-path":
            self.state.providers["audio_cpp"]["guided_binary_source"] = "manual"
            try:
                self.query_one(
                    "#settings-speech-audio_cpp-guided-binary-source",
                    Static,
                ).update("Guided binary selection source: Manual.")
            except QueryError:
                pass

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

        if self._fence_audio_cpp_result_cleanup():
            return False

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
            references = self._managed_refs_from_values(
                self.state.providers["audio_cpp"]
            ) | self._managed_refs_from_values(
                self.original_state.providers["audio_cpp"]
            )
            if not await self._acquire_managed_refs(references):
                self._set_result(
                    "Managed model packages are busy or cleanup is still pending.",
                    severity="error",
                )
                return False
            try:
                self._reset_to_saved()
            finally:
                self._transfer_managed_refs()
            self.post_message(
                self.DraftModified(
                    False,
                    self.state,
                    self.original_state,
                    self.configure_provider,
                    self.draft_snapshot(),
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
        selector = self.query_one(
            "#settings-speech-configure-provider",
            Select,
        )
        with selector.prevent(Select.Changed):
            selector.value = self.configure_provider
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
        await self._replace_card_bodies(
            self._PROVIDER_SETUP_CARD_ID, self._INSPECTOR_CARD_ID
        )
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
        await self._replace_card_bodies(
            self._GLOBAL_DEFAULTS_CARD_ID, self._INSPECTOR_CARD_ID
        )
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
        await self._replace_card_bodies(
            self._GLOBAL_DEFAULTS_CARD_ID, self._INSPECTOR_CARD_ID
        )
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
        await self._replace_card_bodies(
            self._GLOBAL_DEFAULTS_CARD_ID, self._INSPECTOR_CARD_ID
        )
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
        await self._replace_card_bodies(
            self._GLOBAL_DEFAULTS_CARD_ID, self._INSPECTOR_CARD_ID
        )
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

    @on(Select.Changed, "#settings-speech-audio_cpp-mode")
    def handle_audio_cpp_mode_changed(self, event: Select.Changed) -> None:
        if self._syncing or not isinstance(event.value, str):
            return
        self._fence_audio_cpp_package_scan()
        if event.value == "managed" and not _AUDIO_CPP_MANAGED_UI_SUPPORTED:
            with event.select.prevent(Select.Changed):
                event.select.value = "external"
            self.state.providers["audio_cpp"]["mode"] = "external"
            self._apply_audio_cpp_mode_visibility()
            self._set_result(
                "Managed local server is unavailable on Windows. Use External "
                "server instead.",
                severity="warning",
            )
            return
        values = self.state.providers["audio_cpp"]
        values["mode"] = event.value
        if (
            event.value == "managed"
            and self.original_state.providers["audio_cpp"].get("mode", "external")
            != "managed"
            and not values.get("managed_binary_path")
            and not values.get("managed_server_json_path")
            and not values.get("guided_binary_path")
            and not values.get("guided_packages")
        ):
            values["managed_setup_source"] = "guided"
            try:
                source = self.query_one(
                    "#settings-speech-audio_cpp-managed-setup-source",
                    Select,
                )
                with source.prevent(Select.Changed):
                    source.value = "guided"
            except QueryError:
                pass
        self._apply_audio_cpp_mode_visibility()
        self._announce_draft_state()

    @on(Select.Changed, "#settings-speech-audio_cpp-managed-setup-source")
    def handle_audio_cpp_setup_source_changed(self, event: Select.Changed) -> None:
        if self._syncing or not isinstance(event.value, str):
            return
        self._fence_audio_cpp_package_scan()
        self.state.providers["audio_cpp"]["managed_setup_source"] = event.value
        self._apply_audio_cpp_mode_visibility()
        self._announce_draft_state()

    @on(Button.Pressed, "#settings-speech-audio-cpp-use-detected")
    @on(Button.Pressed, "#settings-speech-audio-cpp-manual-use-detected")
    def handle_audio_cpp_use_detected(self, event: Button.Pressed) -> None:
        event.stop()
        detected = detect_audio_cpp_server_binary()
        if detected is None:
            self._set_result(
                "audiocpp_server was not found on PATH. Keep the current draft "
                "or use Browse to choose a prebuilt binary.",
                severity="warning",
            )
            return
        guided = event.button.id == "settings-speech-audio-cpp-use-detected"
        field_id = "guided_binary_path" if guided else "managed_binary_path"
        try:
            self.query_one(
                f"#{self._field_dom_id('audio_cpp', field_id)}", Input
            ).value = detected
        except QueryError:
            return
        if guided:
            self.state.providers["audio_cpp"]["guided_binary_source"] = "path"
            try:
                self.query_one(
                    "#settings-speech-audio_cpp-guided-binary-source",
                    Static,
                ).update("Guided binary selection source: Path.")
            except QueryError:
                pass
        self._set_result(
            "Detected audiocpp_server in the unsaved draft. Review it and Save; "
            "the server was not started.",
            severity="information",
        )

    @on(Button.Pressed, "#settings-speech-audio-cpp-guided-add-package")
    def handle_audio_cpp_add_package(self, event: Button.Pressed) -> None:
        event.stop()
        if self._fence_audio_cpp_result_cleanup():
            return
        self.app.push_screen(
            SelectDirectory(title="Choose an audio.cpp model package directory"),
            self._audio_cpp_package_picker_result,
        )

    @on(Button.Pressed, "#settings-speech-audio-cpp-open-model-library")
    def handle_audio_cpp_open_model_library(self, event: Button.Pressed) -> None:
        """Delegate exact request staging to the Settings navigation owner."""

        event.stop()
        if self._fence_audio_cpp_result_cleanup():
            return
        snapshot = self.draft_snapshot()
        values = snapshot.state.providers["audio_cpp"]
        if (
            values.get("mode") != "managed"
            or values.get("managed_setup_source") != "guided"
        ):
            self._set_result(
                "Choose Managed and Guided setup before opening Model Library.",
                severity="warning",
            )
            return
        stage = getattr(self.screen, "stage_audio_cpp_model_library_request", None)
        if not callable(stage) or not stage(snapshot):
            self._set_result(
                "Model Library could not be opened. Your draft is unchanged.",
                severity="error",
            )

    def _audio_cpp_package_picker_result(self, path: Path | None) -> None:
        """Start one latest-wins bounded scan for an explicitly selected root."""

        if path is None:
            return
        if self._fence_audio_cpp_result_cleanup():
            return
        self._collect_visible_state()
        values = self.state.providers["audio_cpp"]
        if (
            values.get("mode") != "managed"
            or values.get("managed_setup_source") != "guided"
        ):
            return
        self._audio_cpp_scan_revision += 1
        revision = self._audio_cpp_scan_revision
        self._set_result(
            "Scanning the selected package against reviewed audio.cpp recipes…"
        )
        self.run_worker(
            self._scan_audio_cpp_package(path, revision),
            group=_AUDIO_CPP_PACKAGE_SCAN_GROUP,
            exclusive=True,
            exit_on_error=False,
        )

    async def _scan_audio_cpp_package(self, path: Path, revision: int) -> None:
        """Accept exact reviewed candidates only when this scan still owns the UI."""

        try:
            result = await scan_audio_cpp_package_root_async(
                path,
                request_revision=revision,
            )
        except asyncio.CancelledError:
            raise
        except (AudioCppPackageScanError, OSError, ValueError):
            if revision == self._audio_cpp_scan_revision and self.is_mounted:
                self._set_result(
                    "The selected package could not be reviewed. Choose a readable "
                    "package directory and try again.",
                    severity="error",
                )
            return
        if (
            revision != self._audio_cpp_scan_revision
            or not self.is_mounted
            or result.request_revision != revision
            or self.audio_cpp_result_cleanup_pending()
        ):
            return

        values = self.state.providers["audio_cpp"]
        if (
            values.get("mode") != "managed"
            or values.get("managed_setup_source") != "guided"
        ):
            return
        existing = list(self._audio_cpp_guided_packages())
        identities = {
            (
                item.canonical_root,
                item.package_variant,
                item.configuration_identity,
                item.weight_identity,
            )
            for item in existing
        }
        model_ids = {item.public_model_id for item in existing}
        added = 0
        conflicts = 0
        for discovery in result.discoveries:
            if discovery.match.state not in {
                AudioCppMatchState.EXACT,
                AudioCppMatchState.AMBIGUOUS,
            }:
                continue
            for candidate in discovery.match.candidates:
                identity = (
                    candidate.canonical_root,
                    candidate.recipe.package_variant,
                    candidate.configuration_identity,
                    candidate.weight_identity,
                )
                if identity in identities:
                    continue
                accepted = candidate.accept()
                if accepted.public_model_id in model_ids:
                    conflicts += 1
                    continue
                existing.append(accepted)
                identities.add(identity)
                model_ids.add(accepted.public_model_id)
                added += 1

        values["guided_packages"] = [
            package.model_dump(mode="json") for package in existing
        ]
        if not values.get("guided_default_model_id") and existing:
            values["guided_default_model_id"] = existing[0].public_model_id
        await self.recompose()
        self._apply_audio_cpp_mode_visibility()
        try:
            self.query_one(
                "#settings-speech-audio-cpp-guided-add-package",
                Button,
            ).focus()
        except QueryError:
            pass
        if added:
            suffix = " A conflicting public model ID was skipped." if conflicts else ""
            qualifier = (
                " Review every added variant and remove any you do not want."
                if added > 1
                else " Review it before Save."
            )
            self._set_result(
                f"Added {added} exact reviewed package candidate"
                f"{'s' if added != 1 else ''} to the unsaved draft."
                f"{qualifier}{suffix}"
            )
        elif result.outcome is AudioCppScanOutcome.CANCELLED:
            self._set_result("The package scan was cancelled.")
        else:
            self._set_result(
                "No new exact reviewed package was found. Choose a supported "
                "Supertonic or PocketTTS package, or review the scan evidence.",
                severity="warning",
            )
        self._announce_draft_state()

    @on(Button.Pressed, ".settings-speech-audio-cpp-package-remove")
    async def handle_audio_cpp_remove_package(self, event: Button.Pressed) -> None:
        event.stop()
        if self._fence_audio_cpp_result_cleanup():
            return
        button_id = event.button.id or ""
        prefix = "settings-speech-audio-cpp-guided-package-remove-"
        if not button_id.startswith(prefix):
            return
        package_uuid = button_id.removeprefix(prefix)
        packages = [
            package
            for package in self._audio_cpp_guided_packages()
            if package.package_uuid != package_uuid
        ]
        values = self.state.providers["audio_cpp"]
        proposed_values = dict(values)
        proposed_values["guided_packages"] = [
            package.model_dump(mode="json") for package in packages
        ]
        references = self._managed_refs_from_values(
            values
        ) | self._managed_refs_from_values(proposed_values)
        if not await self._acquire_managed_refs(references):
            self._set_result(
                "Managed model packages are busy or cleanup is still pending.",
                severity="error",
            )
            return
        values["guided_packages"] = [
            package.model_dump(mode="json") for package in packages
        ]
        remaining_ids = {package.public_model_id for package in packages}
        if values.get("guided_default_model_id") not in remaining_ids:
            values["guided_default_model_id"] = (
                packages[0].public_model_id if packages else None
            )
        try:
            await self.recompose()
            self._apply_audio_cpp_mode_visibility()
            try:
                self.query_one(
                    "#settings-speech-audio-cpp-guided-add-package",
                    Button,
                ).focus()
            except QueryError:
                pass
            self._set_result(
                "Removed the package from the unsaved draft. No local files were deleted."
            )
            self._announce_draft_state()
        finally:
            self._transfer_managed_refs()

    @on(Input.Changed, ".settings-speech-draft-field")
    @on(Select.Changed, ".settings-speech-draft-field")
    @on(Switch.Changed, ".settings-speech-draft-field")
    def handle_draft_field_changed(
        self,
        event: Input.Changed | Select.Changed | Switch.Changed,
    ) -> None:
        control = getattr(event, "control", None)
        if (
            not self._syncing
            and getattr(control, "id", None) not in _SEMANTIC_DRAFT_CONTROL_IDS
        ):
            self._announce_draft_state()

    def _reset_to_saved(self) -> None:
        """Reset the draft model without starting a competing recompose."""

        self._fence_audio_cpp_package_scan()
        self._clear_validation_errors()
        self.state = deepcopy(self.original_state)
        self._realtime_draft = replace(self._realtime_original)
        self.result_text = "Reverted to the last successfully loaded global values."

    async def revert_to_saved(self) -> None:
        """Restore the last successfully loaded or published snapshot."""

        if self._fence_audio_cpp_result_cleanup():
            return

        references = self._managed_refs_from_values(
            self.state.providers["audio_cpp"]
        ) | self._managed_refs_from_values(self.original_state.providers["audio_cpp"])
        if not await self._acquire_managed_refs(references):
            self._set_result(
                "Managed model packages are busy or cleanup is still pending.",
                severity="error",
            )
            return
        try:
            self._reset_to_saved()
            get_current_worker()
        except NoActiveWorker:
            try:
                await self.recompose()
                self._announce_draft_state()
            finally:
                self._transfer_managed_refs()
        else:
            # A worker awaiting recompose can deadlock against the same
            # message pump that owns guarded navigation. Schedule that path.
            self.refresh(recompose=True)
            self.call_after_refresh(self._announce_draft_state)
            self._transfer_managed_refs()

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
        if self._fence_audio_cpp_result_cleanup():
            return
        self._collect_visible_state()
        self._clear_validation_errors()
        restored = restore_non_secret_defaults(
            self.state,
            configure_provider=self.configure_provider,
        )
        references = self._managed_refs_from_values(
            self.state.providers["audio_cpp"]
        ) | self._managed_refs_from_values(restored.providers["audio_cpp"])
        if not await self._acquire_managed_refs(references):
            self._set_result(
                "Managed model packages are busy or cleanup is still pending.",
                severity="error",
            )
            return
        self.state = restored
        self.result_text = (
            "Non-secret defaults restored in the draft; choose Save to persist them."
        )
        try:
            await self.recompose()
            self._announce_draft_state()
        finally:
            self._transfer_managed_refs()

    @on(Button.Pressed, "#settings-speech-open-lab")
    @on(Button.Pressed, "#settings-speech-open-lab-bottom")
    @on(Button.Pressed, "#settings-speech-audio-cpp-open-lab")
    def handle_open_lab(self, event: Button.Pressed) -> None:
        event.stop()
        provider_id = (
            "audio_cpp"
            if event.button.id == "settings-speech-audio-cpp-open-lab"
            else self.configure_provider
        )
        self.run_worker(
            self._open_lab(
                SpeechTTSNavigationTarget(
                    provider_id,
                    SpeechTTSNavigationIntent.TEST,
                ),
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
        elif target_selector == ("#settings-speech-audio_cpp-managed-binary-path"):
            picker = FileOpen(
                title="Choose prebuilt audiocpp_server",
                filters=Filters(("Executable files", lambda path: path.is_file())),
            )
        elif target_selector == ("#settings-speech-audio_cpp-guided-binary-path"):
            picker = FileOpen(
                title="Choose separately installed audiocpp_server",
                filters=Filters(("Executable files", lambda path: path.is_file())),
            )
        elif target_selector == ("#settings-speech-audio_cpp-managed-server-json-path"):
            picker = FileOpen(
                title="Choose existing server.json",
                filters=Filters(("JSON files", lambda path: path.suffix == ".json")),
            )
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
