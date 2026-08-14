"""THESIS: A bounded voice control desk, not a card gallery.
OWN-WORLD: Inherit flat $panel/$surface layers, semantic
$primary/$accent/$warning/$error, round structural borders, and one-cell rhythm.
STORY: Repository rows appear first; authority enrichment follows while the
selected exact voice and every recovery state remain legible.
FIRST VIEWPORT: Concise title and purpose, one search-and-paging toolbar, a
central selectable table, one status/detail/recovery region, and an explicit
action rail.
FORM: Established-surface local extension; compact control desk with no concept
seed, decorative cards, gradients, side stripes, or new visual identity.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Literal, Protocol, cast
from uuid import UUID, uuid4

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Input,
    Label,
    Select,
    Static,
    TextArea,
)

from tldw_chatbook.TTS import (
    LoadedTTSProfile,
    ProfileRepositoryError,
    ProfileServiceError,
    ProfileValidationError,
    STTSGeneratedAudio,
    TTSPlaygroundSelectionPreset,
    TTSProfileAvailability,
    TTSProfileAvailabilitySnapshot,
    TTSProfileDraft,
    TTSProfilePageSnapshot,
    TTSVoiceBundleHandle,
    TTSVoiceBundleImportChoice,
    TTSVoiceBundleImportResult,
    TTSVoiceBundleReview,
)
from tldw_chatbook.TTS.profile_portability import (
    PortableTTSProfile,
    portable_profile_json,
)
from tldw_chatbook.TTS.profile_types import PROFILE_PROVIDER_REQUIRES_EXACT_VOICE
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Speech.speech_runtime_status import (
    speech_tts_navigation_context,
)
from tldw_chatbook.UI.Speech.speech_settings_contracts import (
    SpeechTTSNavigationIntent,
    SpeechTTSNavigationTarget,
)
from tldw_chatbook.UI.tts_profile_recovery import (
    TTSProfileDependencyActionProjection,
    dependency_recovery_actions,
)
from tldw_chatbook.Utils.input_validation import validate_text_input

PROFILE_PAGE_SIZE = 50
PROFILE_SEARCH_DEBOUNCE_SECONDS = 0.25
PROFILE_SEARCH_MAX_CHARACTERS = 128
PROFILE_STORE_UNAVAILABLE_COPY = (
    "Profile storage unavailable. Ordinary speech tools remain available. "
    "Choose Refresh to retry."
)
PROFILE_CONFLICT_COPY = (
    "Profile storage changed while this action was pending. Refresh, review "
    "the current profile, and retry."
)
PROFILE_ACTION_FAILED_COPY = (
    "The profile action could not be completed. No persisted values were "
    "changed. Refresh and retry."
)
PROFILE_DELETE_PROTECTED_COPY = (
    "This profile is assigned and cannot be deleted. Remove its assignments "
    "before retrying."
)
# Task 5 (slice 3): the app-wide default lives in config (`[app_tts]
# default_profile_id`), never in the profile store, so it cannot ride the
# `assignment_count` machinery above -- `STTSProfileLibrary` supplies it
# separately (see `_is_configured_default_profile`). Deletion itself stays
# unblocked here: Task 4 already made speech refuse honestly, with a
# one-tap "global voice" override, once the configured default no longer
# resolves -- this copy only WARNS before the destructive act, never
# fabricates an automatic silent fallback.
PROFILE_DELETE_APP_DEFAULT_COPY = (
    "This is the app-wide default voice. Deleting it leaves no default "
    "voice configured — future default-voice speech will ask to use the "
    "global voice instead."
)
_STORE_UNAVAILABLE_CODES = frozenset(
    {
        "closed",
        "corrupt_data",
        "invalid_state",
        "restoring",
        "schema_corrupt",
        "schema_partial",
        "schema_unsupported",
        "terminal",
        "unavailable",
    }
)
_PROFILE_LOADING_COPY = "Loading voice profiles…"
_PROFILE_EMPTY_COPY = (
    "No voice profiles match. Change the search or choose Refresh to check again."
)
_PROFILE_LOAD_FAILED_COPY = (
    "Voice profiles could not be loaded. Choose Refresh to retry; ordinary "
    "speech remains available."
)
_PROFILE_AVAILABILITY_FAILED_COPY = (
    "Profiles are loaded, but availability is unverified. Choose Refresh to "
    "retry; exact persisted values were not changed."
)
_PROFILE_STALE_COPY = (
    "Profile storage changed while this page loaded. Existing rows were kept; "
    "choose Refresh to load the current store."
)
_PROFILE_VALIDATION_COPY = (
    "Review the profile name, model, and voice. Exact values were not saved."
)
_PROFILE_SEARCH_VALIDATION_COPY = "Search must be 128 characters or fewer."
_PROFILE_NAME_REQUIRED_COPY = "Enter a profile name. The result was not saved."
_PROFILE_UNAVAILABLE_COPY = (
    "The exact profile selection is unavailable. Refresh current capabilities "
    "or edit the persisted model and voice."
)
# This copy routes from `profile_unverified` and `stale_configuration` (see
# `profile_action_error_copy` below). All FIVE raise sites of those two codes
# in `profile_service.py` were traced:
#   :1331 stale_configuration  -- `observe_availability`'s page-snapshot flow;
#       gated on `audio_cpp_supported`, the provider_id == _PROFILE_PROVIDER_ID
#       subset of the page built a few lines above. Legacy-unreachable.
#   :1402 stale_configuration  -- `observe_portable_profile`; opens with
#       `if draft.provider_id != _PROFILE_PROVIDER_ID: return` (an early
#       return with availability="unverified", before this raise).
#       Legacy-unreachable.
#   :2357/:2374 profile_unverified -- both inside
#       `_require_authoritative_capability`, which opens with
#       `if draft.provider_id != _PROFILE_PROVIDER_ID: return`.
#       Legacy-unreachable.
#   :1612 (leads to :2330) stale_configuration -- `create_from_artifact`
#       calls `_require_configuration_revision(selection.provider_id, ...)`
#       with NO provider gate, and `_selection_is_profile_safe` permits any
#       provider with a known response-format table. `configuration_revision`
#       is a genuine per-provider counter (`adapter_registry.py`,
#       `TTS_Generation.py`), so `stale_configuration` IS raisable here for a
#       legacy provider. LEGACY-REACHABLE -- the two live callers
#       (`STTS_Window.py`, `speech_profile_mixin.py`) already special-case
#       this code with `_PROFILE_RESULT_STALE_COPY` before falling through to
#       this toast; `STTSProfileLibrary.create_from_artifact` below mirrors
#       that (slice 2, task 1 fix round).
_PROFILE_UNVERIFIED_COPY = (
    "The exact profile selection could not be verified. Refresh and retry "
    "without changing persisted values."
)
_PROFILE_VERIFIED_COPY = "Verified"
_PROFILE_NEEDS_TEST_COPY = "Needs test"
_PROFILE_UNAVAILABLE_STATUS_COPY = "Unavailable"
_PROFILE_TEST_CONTEXT_LIMIT = 256
#: Copy shown when the generated audio predates the current settings, so
#: saving it as a profile would record something the user did not hear.
#: Kept verbatim from `STTS_Window` / `speech_profile_mixin`, which define
#: their own copies of this same string for the same `stale_configuration`
#: case reached through `create_from_artifact`.
_PROFILE_RESULT_STALE_COPY = (
    "TTS settings changed after this audio was generated. Generate a new "
    "result before saving it as a profile."
)
PROFILE_EXPORT_COMPLETE_COPY = "Voice profile exported."
PROFILE_BUNDLE_EXPORT_COMPLETE_COPY = "Portable voice bundle exported."
PROFILE_BUNDLE_IMPORT_COMPLETE_COPY = "Portable voice bundle imported."
PROFILE_BUNDLE_UNSUPPORTED_COPY = (
    "Portable voice bundles are unavailable on Windows because secure file "
    "permissions cannot yet be guaranteed. Sanitized JSON export remains available."
)
PROFILE_BUNDLE_WARNING_COPY = (
    "This bundle contains plaintext voice audio and transcript. Anyone with the "
    "file can access them. I confirm I have permission to export and share this "
    "material."
)
PROFILE_BUNDLE_IMPORT_WARNING_COPY = (
    "Importing reads plaintext voice audio and transcript. The bundle declaration "
    "is not proof of permission or identity."
)
PROFILE_BUNDLE_MIGRATED_COPY = (
    "Recipe provenance unavailable. Preview or generate the voice, save it as a "
    "new profile, then reassign or remove the old profile."
)


class _ProfileService(Protocol):
    async def list_profiles(
        self,
        *,
        search: str | None = None,
        offset: int = 0,
    ) -> TTSProfilePageSnapshot: ...

    async def observe_availability(
        self,
        page: TTSProfilePageSnapshot,
    ) -> TTSProfileAvailabilitySnapshot: ...

    async def create_from_artifact(
        self,
        display_name: str,
        artifact: STTSGeneratedAudio,
    ) -> LoadedTTSProfile: ...

    async def update_profile(
        self,
        loaded: LoadedTTSProfile,
        draft: TTSProfileDraft,
    ) -> LoadedTTSProfile: ...

    async def duplicate_profile(
        self,
        loaded: LoadedTTSProfile,
        display_name: str,
    ) -> LoadedTTSProfile: ...

    async def assignment_count(self, loaded: LoadedTTSProfile) -> int: ...

    async def delete_profile(self, loaded: LoadedTTSProfile) -> None: ...

    def preview_preset(
        self,
        loaded: LoadedTTSProfile,
        availability: TTSProfileAvailability,
    ) -> TTSPlaygroundSelectionPreset: ...

    def record_sample_evidence(
        self,
        loaded: LoadedTTSProfile,
        artifact: STTSGeneratedAudio,
    ) -> None: ...


ProfileServiceLoader = Callable[[], Awaitable[_ProfileService | None]]


class _VoiceBundleService(Protocol):
    async def inspect(self, source: Path) -> TTSVoiceBundleReview: ...

    async def commit(
        self,
        handle: TTSVoiceBundleHandle,
        choice: TTSVoiceBundleImportChoice,
    ) -> TTSVoiceBundleImportResult: ...

    async def export(
        self,
        profile_id: UUID,
        destination: Path,
        *,
        expected_generation: int,
        expected_revision: int,
        acknowledged: bool,
    ) -> None: ...

    async def invalidate(self, handle: TTSVoiceBundleHandle) -> None: ...


VoiceBundleServiceLoader = Callable[[], Awaitable[_VoiceBundleService | None]]


@dataclass(frozen=True, slots=True)
class VoiceBundleActionProjection:
    """Immutable visible and executable truth for one portability action."""

    operation: Literal[
        "sanitized_export",
        "bundle_export",
        "import_create",
        "import_reuse",
        "import_copy",
    ]
    label: str
    tooltip: str
    disabled: bool
    recovery: str | None = None


@dataclass(frozen=True, slots=True)
class VoiceBundleReviewDecision:
    choice: Literal["create", "reuse", "copy"]
    inactive_consent: bool


def voice_bundle_export_actions(
    *,
    bundle_disabled: bool,
    bundle_recovery: str | None,
) -> tuple[VoiceBundleActionProjection, VoiceBundleActionProjection]:
    """Project both export choices from one immutable source of truth."""

    sanitized = VoiceBundleActionProjection(
        operation="sanitized_export",
        label="Export sanitized profile",
        tooltip="Export profile settings without voice audio or transcript.",
        disabled=False,
    )
    bundle = VoiceBundleActionProjection(
        operation="bundle_export",
        label="Export portable voice bundle",
        tooltip=(bundle_recovery or "Export plaintext voice audio and transcript."),
        disabled=bundle_disabled,
        recovery=bundle_recovery,
    )
    return sanitized, bundle


def voice_bundle_import_choice(
    action: VoiceBundleActionProjection,
    *,
    inactive_consent: bool,
) -> TTSVoiceBundleImportChoice:
    """Translate only an enabled projected import operation for the service."""

    choices: dict[str, Literal["create", "reuse", "copy"]] = {
        "import_create": "create",
        "import_reuse": "reuse",
        "import_copy": "copy",
    }
    choice = choices.get(action.operation)
    if action.disabled or choice is None:
        raise ValueError("action is not an enabled import operation")
    return TTSVoiceBundleImportChoice(
        choice=choice,
        inactive_consent=inactive_consent,
    )


def voice_bundle_review_action(
    review: TTSVoiceBundleReview,
    choice: Literal["create", "reuse", "copy"],
    *,
    inactive_consent: bool,
) -> VoiceBundleActionProjection:
    """Project the exact operation the confirmation control will execute."""

    allowed = choice in review.allowed_choices
    if choice == "reuse":
        allowed = allowed and review.exact_private_duplicate
    needs_consent = choice in {"create", "copy"} and review.dependency_state != "exact"
    disabled = not allowed or (needs_consent and not inactive_consent)
    recovery = None
    if not allowed:
        recovery = "Choose an available destination."
    elif needs_consent and not inactive_consent:
        recovery = "Acknowledge that the imported profile will remain inactive."
    return VoiceBundleActionProjection(
        operation=cast(
            Literal["import_create", "import_reuse", "import_copy"],
            f"import_{choice}",
        ),
        label={"create": "Create", "reuse": "Reuse", "copy": "Create copy"}[choice],
        tooltip=recovery or "Confirm the reviewed import destination.",
        disabled=disabled,
        recovery=recovery,
    )


@dataclass(frozen=True, slots=True)
class _PageRequest:
    mount_token: int
    request_id: int
    search: str | None
    offset: int


_PROFILE_FOCUS_TARGETS = frozenset(
    {
        "stts-profile-table",
        "stts-profile-preview-btn",
        "stts-profile-edit-btn",
        "stts-profile-duplicate-btn",
        "stts-profile-export-btn",
        "stts-profile-refresh-btn",
        "stts-profile-delete-btn",
    }
)


@dataclass(frozen=True, slots=True)
class ProfileLibraryContinuity:
    """Bounded widget-free state needed to restore the profile library."""

    selected_profile_id: UUID | None
    cursor_row: int
    scroll_x: int
    scroll_y: int
    focus_target: str | None
    search: str | None
    offset: int


@dataclass(frozen=True, slots=True)
class ProfileVerificationResult:
    """One exact successful profile test awaiting library reconciliation."""

    profile_id: UUID
    repository_generation: int
    profile_revision: int
    availability: TTSProfileAvailability


@dataclass(frozen=True, slots=True)
class _ProfileTestContext:
    """Process-local authority for one exact, non-secret profile test."""

    service: _ProfileService
    loaded: LoadedTTSProfile


@dataclass(frozen=True, slots=True)
class _ProfileTestRegistration:
    """One immutable preset paired with opaque process-local authority."""

    preset: TTSPlaygroundSelectionPreset
    context_token: UUID


_PROFILE_TEST_CONTEXTS: dict[UUID, _ProfileTestContext] = {}
_PROFILE_TEST_CONTEXTS_LOCK = Lock()


def _profile_test_key(
    preset: TTSPlaygroundSelectionPreset,
) -> tuple[UUID, int, int] | None:
    identity = (
        preset.profile_id,
        preset.repository_generation,
        preset.profile_revision,
    )
    if not all(value is not None for value in identity):
        return None
    profile_id, repository_generation, profile_revision = identity
    assert isinstance(profile_id, UUID)
    assert isinstance(repository_generation, int)
    assert isinstance(profile_revision, int)
    return profile_id, repository_generation, profile_revision


def _preset_matches_loaded(
    preset: TTSPlaygroundSelectionPreset,
    loaded: LoadedTTSProfile,
) -> bool:
    profile = loaded.profile
    return all(
        source == expected
        for source, expected in (
            (preset.profile_id, profile.profile_id),
            (preset.repository_generation, loaded.repository_generation),
            (preset.profile_revision, profile.revision),
            (preset.provider_id, profile.provider_id),
            (preset.model_id, profile.model_id),
            (preset.voice_id, profile.voice_id),
            (preset.response_format, profile.response_format),
            (preset.speed, profile.speed),
            (dict(preset.options), dict(profile.options)),
        )
    )


def _remember_profile_test_context(
    service: _ProfileService,
    loaded: LoadedTTSProfile,
    preset: TTSPlaygroundSelectionPreset,
) -> _ProfileTestRegistration:
    """Attach exact repository identity and retain bounded process authority."""

    profile = loaded.profile
    exact = replace(
        preset,
        profile_id=profile.profile_id,
        repository_generation=loaded.repository_generation,
        profile_revision=profile.revision,
    )
    if not _preset_matches_loaded(exact, loaded):
        raise ValueError("Profile test preset does not match the selected profile")
    context_token = uuid4()
    with _PROFILE_TEST_CONTEXTS_LOCK:
        _PROFILE_TEST_CONTEXTS[context_token] = _ProfileTestContext(service, loaded)
        while len(_PROFILE_TEST_CONTEXTS) > _PROFILE_TEST_CONTEXT_LIMIT:
            _PROFILE_TEST_CONTEXTS.pop(next(iter(_PROFILE_TEST_CONTEXTS)))
    return _ProfileTestRegistration(exact, context_token)


def _resolve_profile_test_context(
    context_token: UUID | None,
    preset: TTSPlaygroundSelectionPreset,
) -> _ProfileTestContext | None:
    if type(context_token) is not UUID or _profile_test_key(preset) is None:
        return None
    with _PROFILE_TEST_CONTEXTS_LOCK:
        context = _PROFILE_TEST_CONTEXTS.get(context_token)
    if context is None or not _preset_matches_loaded(preset, context.loaded):
        return None
    return context


def _consume_profile_test_context(
    context_token: UUID | None,
    preset: TTSPlaygroundSelectionPreset,
) -> _ProfileTestContext | None:
    """Atomically consume one matching context exactly once."""

    if type(context_token) is not UUID or _profile_test_key(preset) is None:
        return None
    with _PROFILE_TEST_CONTEXTS_LOCK:
        context = _PROFILE_TEST_CONTEXTS.get(context_token)
        if context is None or not _preset_matches_loaded(preset, context.loaded):
            return None
        return _PROFILE_TEST_CONTEXTS.pop(context_token)


def _retire_profile_test_context(context_token: UUID | None) -> bool:
    """Release one owned context without affecting newer sessions."""

    if type(context_token) is not UUID:
        return False
    with _PROFILE_TEST_CONTEXTS_LOCK:
        return _PROFILE_TEST_CONTEXTS.pop(context_token, None) is not None


def _profile_test_context_count() -> int:
    """Return the bounded registry size for lifecycle verification."""

    with _PROFILE_TEST_CONTEXTS_LOCK:
        return len(_PROFILE_TEST_CONTEXTS)


class ProfilePreviewRequested(Message):
    """Request a one-shot exact Playground preview without synthesizing."""

    def __init__(
        self,
        preset: TTSPlaygroundSelectionPreset,
        continuity: ProfileLibraryContinuity,
        context_token: UUID,
    ) -> None:
        super().__init__()
        self.preset = preset
        self.continuity = continuity
        self.context_token = context_token


class ProfileTestVerified(Message):
    """Carry one verified result to the navigation owner without widget refs."""

    def __init__(self, result: ProfileVerificationResult) -> None:
        super().__init__()
        self.result = result


class ProfileVerificationReconciled(Message):
    """Retire one pending result after a fresh library mount checks it."""

    def __init__(self, result: ProfileVerificationResult) -> None:
        super().__init__()
        self.result = result


class ProfileLibraryRestoreReady(Message):
    """Signal that remounted rows have final geometry for focus restoration."""

    def __init__(self, ownership_token: int) -> None:
        super().__init__()
        self.ownership_token = ownership_token


def _assignment_copy(count: int) -> str:
    noun = "assignment" if count == 1 else "assignments"
    return f"{count} {noun}"


def _availability_cell_text(availability: TTSProfileAvailability | None) -> str:
    """Render one row's "Availability" cell -- the single source both
    `_publish_page` and `_publish_availability` call, so they cannot diverge.

    `recovery_action == "none"` on an `"unverified"` availability means the
    provider has no catalog to preflight, so the state is permanent, not a
    transient result waiting on Refresh (`_ALLOWED_RECOVERY_ACTIONS` in
    `profile_service.py` documents the contract). That case alone gets the
    honest "No catalog check" copy; every other state -- including
    audio_cpp's transient "unverified" -- keeps today's plain
    `.state.title()` rendering.
    """
    if availability is None:
        return "Checking"
    if availability.dependency.display:
        return availability.dependency.display
    status = {
        "available": _PROFILE_VERIFIED_COPY,
        "unverified": _PROFILE_NEEDS_TEST_COPY,
        "unavailable": _PROFILE_UNAVAILABLE_STATUS_COPY,
    }[availability.state]
    if availability.dependency.advisory_display:
        return f"{status} · {availability.dependency.advisory_display}"
    return status


def profile_action_error_copy(error: BaseException) -> str:
    """Map structured failures without rendering exception-owned values."""

    code = getattr(error, "code", None)
    if isinstance(error, ProfileRepositoryError):
        if code in {"conflict", "stale"}:
            return PROFILE_CONFLICT_COPY
        if code in _STORE_UNAVAILABLE_CODES:
            return PROFILE_STORE_UNAVAILABLE_COPY
    if isinstance(error, ProfileServiceError):
        if code in {"profile_unavailable", "unsupported_profile"}:
            return _PROFILE_UNAVAILABLE_COPY
        if code in {"profile_unverified", "stale_configuration"}:
            return _PROFILE_UNVERIFIED_COPY
    if isinstance(error, ProfileValidationError):
        return _PROFILE_VALIDATION_COPY
    return PROFILE_ACTION_FAILED_COPY


def _is_store_unavailable(error: BaseException) -> bool:
    return (
        isinstance(error, ProfileRepositoryError)
        and error.code in _STORE_UNAVAILABLE_CODES
    )


class TTSProfileNameModal(ModalScreen[str | None]):
    """Ask for one display name before saving an eligible Playground result."""

    BINDINGS = (("escape", "dismiss", "Cancel"),)

    DEFAULT_CSS = """
    TTSProfileNameModal {
        align: center middle;
        background: $background 75%;
    }

    #stts-profile-name-dialog {
        width: 100%;
        max-width: 64;
        height: auto;
        background: $panel;
        border: round $accent;
        padding: 1;
    }

    #stts-profile-name-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #stts-profile-name-error {
        height: auto;
        color: $warning;
    }

    #stts-profile-name-actions {
        height: 3;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="stts-profile-name-dialog"):
            yield Label("Save result as profile", id="stts-profile-name-title")
            yield Input(
                placeholder="Profile name",
                id="stts-profile-name-input",
            )
            yield Static("", id="stts-profile-name-error")
            with Horizontal(id="stts-profile-name-actions"):
                yield Button(
                    "Save",
                    id="stts-profile-name-save",
                    variant="primary",
                )
                yield Button("Cancel", id="stts-profile-name-cancel")

    def on_mount(self) -> None:
        self.query_one("#stts-profile-name-input", Input).focus()

    def _submit(self) -> None:
        name = self.query_one("#stts-profile-name-input", Input).value.strip()
        if not name:
            self.query_one("#stts-profile-name-error", Static).update(
                Text(_PROFILE_NAME_REQUIRED_COPY)
            )
            return
        self.dismiss(name)

    @on(Input.Submitted, "#stts-profile-name-input")
    def _handle_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._submit()

    @on(Button.Pressed, "#stts-profile-name-save")
    def _handle_save(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit()

    @on(Button.Pressed, "#stts-profile-name-cancel")
    def _handle_cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)


@dataclass(frozen=True, slots=True)
class TTSCloneProfileSaveReview:
    """Validated clone-profile name and explicit post-save destination."""

    display_name: str
    choose_character: bool

    def __post_init__(self) -> None:
        if (
            type(self.display_name) is not str
            or not self.display_name
            or self.display_name != self.display_name.strip()
        ):
            raise ValueError("clone profile name must be trimmed and non-empty")
        if type(self.choose_character) is not bool:
            raise TypeError("choose_character must be a boolean")


class TTSCloneProfileSaveReviewModal(ModalScreen[TTSCloneProfileSaveReview | None]):
    """Review a clone result and choose whether to continue to Roleplay."""

    BINDINGS = (("escape", "dismiss", "Cancel"),)

    DEFAULT_CSS = """
    TTSCloneProfileSaveReviewModal {
        align: center middle;
        background: $background 75%;
    }

    #stts-clone-profile-dialog {
        width: 100%;
        max-width: 72;
        height: auto;
        background: $panel;
        border: round $accent;
        padding: 1;
    }

    #stts-clone-profile-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #stts-clone-profile-copy, #stts-clone-profile-error {
        height: auto;
    }

    #stts-clone-profile-error {
        color: $warning;
    }

    #stts-clone-profile-actions {
        height: auto;
        margin-top: 1;
    }

    #stts-clone-profile-actions Button {
        width: 100%;
        min-width: 0;
        height: 3;
        border: none;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="stts-clone-profile-dialog"):
            yield Label("Save cloned voice", id="stts-clone-profile-title")
            yield Static(
                "Name this exact generated voice. Saving does not change the "
                "global default or assign it to a character.",
                id="stts-clone-profile-copy",
            )
            yield Input(
                placeholder="Voice profile name",
                id="stts-clone-profile-name",
            )
            yield Static("", id="stts-clone-profile-error")
            with Vertical(id="stts-clone-profile-actions"):
                yield Button(
                    "Save unassigned",
                    id="stts-clone-profile-save-unassigned",
                )
                yield Button(
                    "Save & choose character",
                    id="stts-clone-profile-save-choose-character",
                    variant="primary",
                )
                yield Button("Cancel", id="stts-clone-profile-cancel")

    def on_mount(self) -> None:
        self.query_one("#stts-clone-profile-name", Input).focus()

    def _submit(self, *, choose_character: bool) -> None:
        name = self.query_one("#stts-clone-profile-name", Input).value.strip()
        if not name:
            self.query_one("#stts-clone-profile-error", Static).update(
                Text(_PROFILE_NAME_REQUIRED_COPY)
            )
            return
        self.dismiss(TTSCloneProfileSaveReview(name, choose_character))

    @on(Button.Pressed, "#stts-clone-profile-save-unassigned")
    def _handle_save_unassigned(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit(choose_character=False)

    @on(Button.Pressed, "#stts-clone-profile-save-choose-character")
    def _handle_save_choose_character(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit(choose_character=True)

    @on(Input.Submitted, "#stts-clone-profile-name")
    def _handle_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._submit(choose_character=False)

    @on(Button.Pressed, "#stts-clone-profile-cancel")
    def _handle_cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)


class TTSProfileEditorModal(ModalScreen[TTSProfileDraft | None]):
    """Focused exact-value editor for one immutable loaded profile token."""

    BINDINGS = (("escape", "dismiss", "Cancel"),)

    DEFAULT_CSS = """
    TTSProfileEditorModal {
        align: center middle;
        background: $background 75%;
    }

    #stts-profile-editor-dialog {
        width: 76;
        height: auto;
        max-height: 34;
        background: $panel;
        border: round $accent;
        padding: 0 1;
    }

    #stts-profile-editor-title {
        text-style: bold;
    }

    #stts-profile-editor-scope,
    #stts-profile-editor-fixed,
    #stts-profile-editor-error {
        height: auto;
        color: $text-muted;
    }

    #stts-profile-editor-error {
        color: $warning;
    }

    .stts-profile-editor-field {
        height: 3;
    }

    .stts-profile-editor-field Label {
        width: 16;
        padding-top: 1;
    }

    .stts-profile-editor-field Input {
        width: 1fr;
    }

    #stts-profile-editor-actions {
        height: 3;
        align-horizontal: right;
    }

    #stts-profile-editor-actions Button {
        min-width: 12;
        height: 3;
        border: none;
        margin-left: 1;
    }

    #stts-profile-editor-actions Button:focus {
        outline: heavy $accent;
    }
    """

    def __init__(
        self,
        loaded: LoadedTTSProfile,
        *,
        assignment_count: int,
        mode: Literal["edit", "duplicate"],
        initial_draft: TTSProfileDraft | None = None,
    ) -> None:
        super().__init__()
        self.loaded = loaded
        self.assignment_count = assignment_count
        self.mode = mode
        self.initial_draft = initial_draft
        profile = loaded.profile
        self.initial_name = (
            ""
            if mode == "duplicate"
            else (
                profile.display_name
                if initial_draft is None
                else initial_draft.display_name
            )
        )
        self.initial_model_id = (
            profile.model_id if initial_draft is None else initial_draft.model_id
        )
        self.initial_voice_id = (
            profile.voice_id if initial_draft is None else initial_draft.voice_id
        )

    def compose(self) -> ComposeResult:
        profile = self.loaded.profile
        duplicate = self.mode == "duplicate"
        title = "Duplicate voice profile" if duplicate else "Edit voice profile"
        assignment_label = _assignment_copy(self.assignment_count)
        if duplicate:
            scope_copy = (
                f"The source has {assignment_label}. Duplicating copies its "
                "exact generation values without changing the source."
            )
        else:
            verb = "uses" if self.assignment_count == 1 else "use"
            scope_copy = (
                f"{assignment_label} {verb} this profile. Editing a shared "
                "profile changes future speech for every assignment."
            )
        with Vertical(id="stts-profile-editor-dialog"):
            yield Label(title, id="stts-profile-editor-title")
            yield Static(
                Text(scope_copy),
                id="stts-profile-editor-scope",
            )
            yield Static(
                Text(
                    f"Provider: {profile.provider_id}  "
                    f"Format: {profile.response_format}  "
                    f"Speed: {profile.speed:g}  Options: preserved exactly"
                ),
                id="stts-profile-editor-fixed",
            )
            with Horizontal(classes="stts-profile-editor-field"):
                yield Label("Name")
                yield Input(
                    value=self.initial_name,
                    placeholder=(
                        "Required new unique name" if duplicate else "Profile name"
                    ),
                    id="stts-profile-editor-name",
                )
            with Horizontal(classes="stts-profile-editor-field"):
                yield Label("Exact model")
                yield Input(
                    value=self.initial_model_id,
                    id="stts-profile-editor-model",
                    disabled=duplicate,
                )
            with Horizontal(classes="stts-profile-editor-field"):
                yield Label("Exact voice")
                yield Input(
                    value=self.initial_voice_id or "",
                    # Only audio.cpp genuinely supports a server-default voice
                    # (`PROFILE_PROVIDER_REQUIRES_EXACT_VOICE`); promising one
                    # for a legacy provider would invite a save the domain
                    # then refuses.
                    placeholder=(
                        "Required"
                        if PROFILE_PROVIDER_REQUIRES_EXACT_VOICE.get(
                            profile.provider_id,
                            True,
                        )
                        else "Server default"
                    ),
                    id="stts-profile-editor-voice",
                    disabled=duplicate,
                )
            yield Static("", id="stts-profile-editor-error")
            with Horizontal(id="stts-profile-editor-actions"):
                yield Button(
                    "Cancel",
                    id="stts-profile-editor-cancel",
                )
                yield Button(
                    "Save",
                    id="stts-profile-editor-save",
                    variant="primary",
                )

    def on_mount(self) -> None:
        self.query_one("#stts-profile-editor-name", Input).focus()

    @on(Button.Pressed, "#stts-profile-editor-cancel")
    def _handle_cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#stts-profile-editor-save")
    def _handle_save(self, event: Button.Pressed) -> None:
        event.stop()
        try:
            draft = self._draft_from_form()
        except ProfileValidationError:
            self.query_one("#stts-profile-editor-error", Static).update(
                Text(_PROFILE_VALIDATION_COPY)
            )
            return
        self.dismiss(draft)

    def _draft_from_form(self) -> TTSProfileDraft:
        profile = self.loaded.profile
        voice = self.query_one("#stts-profile-editor-voice", Input).value
        return TTSProfileDraft(
            display_name=self.query_one(
                "#stts-profile-editor-name",
                Input,
            ).value,
            provider_id=profile.provider_id,
            model_id=self.query_one(
                "#stts-profile-editor-model",
                Input,
            ).value,
            voice_id=voice if voice != "" else None,
            response_format=profile.response_format,
            speed=profile.speed,
            options=profile.options,
        )


class TTSProfileDeleteModal(ModalScreen[bool]):
    """Confirm deletion using an advisory assignment count."""

    BINDINGS = (("escape", "dismiss(False)", "Cancel"),)

    DEFAULT_CSS = """
    TTSProfileDeleteModal {
        align: center middle;
        background: $background 75%;
    }

    #stts-profile-delete-dialog {
        width: 68;
        height: auto;
        background: $panel;
        border: round $error;
        padding: 1 2;
    }

    #stts-profile-delete-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #stts-profile-delete-copy {
        height: auto;
        margin-bottom: 1;
    }

    #stts-profile-delete-target {
        height: auto;
        margin-bottom: 1;
    }

    #stts-profile-delete-actions {
        height: 3;
        align-horizontal: right;
    }

    #stts-profile-delete-actions Button {
        min-width: 12;
        height: 3;
        border: none;
        margin-left: 1;
    }

    #stts-profile-delete-actions Button:focus {
        outline: heavy $accent;
    }
    """

    def __init__(
        self,
        *,
        display_name: str,
        assignment_count: int,
        is_app_default: bool = False,
    ) -> None:
        super().__init__()
        self.display_name = display_name
        self.assignment_count = assignment_count
        self.is_app_default = is_app_default

    def compose(self) -> ComposeResult:
        protected = self.assignment_count > 0
        with Vertical(id="stts-profile-delete-dialog"):
            yield Label("Delete voice profile", id="stts-profile-delete-title")
            target = Text("Profile: ")
            target.append(Text(self.display_name))
            yield Static(target, id="stts-profile-delete-target")
            if protected:
                copy = (
                    f"This profile has {_assignment_copy(self.assignment_count)}. "
                    "Remove them before deletion. The count is advisory and "
                    "profile storage checks again at mutation time."
                )
            else:
                copy = (
                    "No assignments were observed. This advisory count may "
                    "change; profile storage remains the final authority."
                )
            if self.is_app_default:
                copy = f"{copy}\n\n{PROFILE_DELETE_APP_DEFAULT_COPY}"
            yield Static(Text(copy), id="stts-profile-delete-copy")
            with Horizontal(id="stts-profile-delete-actions"):
                yield Button(
                    "Cancel" if not protected else "Close",
                    id="stts-profile-delete-cancel",
                )
                yield Button(
                    "Delete",
                    id="stts-profile-delete-confirm",
                    variant="error",
                    disabled=protected,
                )

    @on(Button.Pressed, "#stts-profile-delete-cancel")
    def _handle_cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(False)

    @on(Button.Pressed, "#stts-profile-delete-confirm")
    def _handle_delete(self, event: Button.Pressed) -> None:
        event.stop()
        if self.assignment_count <= 0:
            self.dismiss(True)


class TTSProfileExportChoiceModal(ModalScreen[VoiceBundleActionProjection | None]):
    """Keep sanitized JSON the default while exposing explicit bundle export."""

    BINDINGS = (("escape", "dismiss(None)", "Cancel"),)

    def __init__(
        self,
        sanitized_action: VoiceBundleActionProjection,
        bundle_action: VoiceBundleActionProjection,
    ) -> None:
        super().__init__()
        if sanitized_action.operation != "sanitized_export":
            raise ValueError("sanitized action has the wrong operation")
        if bundle_action.operation != "bundle_export":
            raise ValueError("bundle action has the wrong operation")
        self.sanitized_action = sanitized_action
        self.bundle_action = bundle_action

    def compose(self) -> ComposeResult:
        with Vertical(classes="stts-portability-dialog"):
            yield Label("Export voice profile", classes="stts-portability-title")
            yield Static(
                "Sanitized JSON omits voice audio and transcript. A portable "
                "bundle includes both in plaintext.",
                classes="stts-portability-copy",
            )
            with Horizontal(classes="stts-portability-actions"):
                yield Button("Cancel", id="stts-export-choice-cancel")
                sanitized = Button(
                    self.sanitized_action.label,
                    id="stts-export-choice-sanitized",
                    variant="primary",
                    disabled=self.sanitized_action.disabled,
                )
                sanitized.tooltip = self.sanitized_action.tooltip
                yield sanitized
                bundle = Button(
                    self.bundle_action.label,
                    id="stts-export-choice-bundle",
                    disabled=self.bundle_action.disabled,
                )
                bundle.tooltip = self.bundle_action.tooltip
                yield bundle
            if self.bundle_action.recovery:
                yield Static(
                    self.bundle_action.recovery,
                    id="stts-export-choice-recovery",
                    classes="stts-portability-recovery",
                )

    def on_mount(self) -> None:
        self.query_one("#stts-export-choice-sanitized", Button).focus()

    @on(Button.Pressed, "#stts-export-choice-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#stts-export-choice-sanitized")
    def _sanitized(self, event: Button.Pressed) -> None:
        event.stop()
        if not self.sanitized_action.disabled:
            self.dismiss(self.sanitized_action)

    @on(Button.Pressed, "#stts-export-choice-bundle")
    def _bundle(self, event: Button.Pressed) -> None:
        event.stop()
        if not self.bundle_action.disabled:
            self.dismiss(self.bundle_action)


class TTSVoiceBundleConsentModal(ModalScreen[bool]):
    """Operation-local plaintext warning; acknowledgement is never persisted."""

    BINDINGS = (("escape", "dismiss(False)", "Cancel"),)

    def __init__(self, *, mode: Literal["export", "import"]) -> None:
        super().__init__()
        self.mode = mode

    def compose(self) -> ComposeResult:
        copy = (
            PROFILE_BUNDLE_WARNING_COPY
            if self.mode == "export"
            else PROFILE_BUNDLE_IMPORT_WARNING_COPY
        )
        with Vertical(classes="stts-portability-dialog"):
            yield Label(
                "Review portable voice bundle",
                classes="stts-portability-title",
            )
            yield Static(copy, classes="stts-portability-copy")
            yield Checkbox(
                (
                    "I confirm I have permission to export and share this material."
                    if self.mode == "export"
                    else "I understand this declaration is not proof of permission or identity."
                ),
                id="bundle-warning-ack",
            )
            with Horizontal(classes="stts-portability-actions"):
                yield Button("Cancel", id="bundle-warning-cancel")
                yield Button(
                    "Continue",
                    id="bundle-warning-continue",
                    variant="primary",
                    disabled=True,
                )

    def on_mount(self) -> None:
        self.query_one("#bundle-warning-ack", Checkbox).focus()

    @on(Checkbox.Changed, "#bundle-warning-ack")
    def _changed(self, event: Checkbox.Changed) -> None:
        self.query_one("#bundle-warning-continue", Button).disabled = not event.value

    @on(Button.Pressed, "#bundle-warning-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(False)

    @on(Button.Pressed, "#bundle-warning-continue")
    def _continue(self, event: Button.Pressed) -> None:
        event.stop()
        if self.query_one("#bundle-warning-ack", Checkbox).value:
            self.dismiss(True)


class TTSVoiceBundleReviewModal(ModalScreen[VoiceBundleReviewDecision | None]):
    """Review only service-owned safe facts and one authoritative action."""

    BINDINGS = (("escape", "dismiss(None)", "Cancel"),)

    def __init__(self, review: TTSVoiceBundleReview) -> None:
        super().__init__()
        self.review = review
        self._choice = review.allowed_choices[0]

    def compose(self) -> ComposeResult:
        review = self.review
        facts = (
            f"Profile: {review.profile_name}\nUUID: {review.profile_id}\n"
            f"Provider / model: {review.provider_id} / {review.model_id}\n"
            f"Recipe: {review.recipe_id} revision {review.recipe_revision}\n"
            f"Dependency: {review.dependency_state.replace('_', ' ')}\n"
            f"UUID conflict: {'yes' if review.uuid_conflict else 'no'}\n"
            f"Name conflict: {'yes' if review.name_conflict else 'no'}\n"
            f"Exact private duplicate: {'yes' if review.exact_private_duplicate else 'no'}"
        )
        if review.copy_profile_id is not None and review.copy_profile_name is not None:
            facts = (
                f"{facts}\nProposed copy: {review.copy_profile_name}\n"
                f"Proposed copy UUID: {review.copy_profile_id}"
            )
        options = tuple(
            (
                {
                    "create": "Create profile",
                    "reuse": "Reuse exact duplicate",
                    "copy": "Create copy",
                }[choice],
                choice,
            )
            for choice in review.allowed_choices
        )
        with ScrollableContainer(classes="stts-portability-dialog stts-review-dialog"):
            yield Label("Review voice bundle import", classes="stts-portability-title")
            yield TextArea(
                facts,
                id="stts-bundle-review-facts",
                read_only=True,
                soft_wrap=True,
                show_line_numbers=False,
                compact=True,
            )
            yield Label("Destination")
            yield Select(
                options,
                value=self._choice,
                allow_blank=False,
                id="stts-bundle-review-choice",
            )
            yield Checkbox(
                "Create this profile inactive until a compatible model is available.",
                id="stts-bundle-inactive-consent",
            )
            yield Static(
                "",
                id="stts-bundle-review-recovery",
                classes="stts-portability-recovery",
            )
            with Horizontal(classes="stts-portability-actions"):
                yield Button(
                    "Confirm", id="stts-bundle-review-confirm", variant="primary"
                )
                yield Button("Cancel", id="stts-bundle-review-cancel")

    def on_mount(self) -> None:
        self._sync_action()
        self.query_one("#stts-bundle-review-facts", TextArea).focus()

    def _sync_action(self) -> VoiceBundleActionProjection:
        consent = self.query_one("#stts-bundle-inactive-consent", Checkbox)
        needs_consent = (
            self._choice in {"create", "copy"}
            and self.review.dependency_state != "exact"
        )
        consent.display = needs_consent
        action = voice_bundle_review_action(
            self.review,
            self._choice,
            inactive_consent=consent.value,
        )
        confirm = self.query_one("#stts-bundle-review-confirm", Button)
        confirm.label = action.label
        confirm.disabled = action.disabled
        confirm.tooltip = action.tooltip
        self.query_one("#stts-bundle-review-recovery", Static).update(
            action.recovery or ""
        )
        return action

    @on(Select.Changed, "#stts-bundle-review-choice")
    def _choice_changed(self, event: Select.Changed) -> None:
        if event.value in {"create", "reuse", "copy"}:
            self._choice = cast(Literal["create", "reuse", "copy"], event.value)
            self._sync_action()

    @on(Checkbox.Changed, "#stts-bundle-inactive-consent")
    def _consent_changed(self, _event: Checkbox.Changed) -> None:
        self._sync_action()

    @on(Button.Pressed, "#stts-bundle-review-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#stts-bundle-review-confirm")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        action = self._sync_action()
        if action.disabled:
            return
        operation_choice = cast(
            Literal["create", "reuse", "copy"], action.operation.removeprefix("import_")
        )
        self.dismiss(
            VoiceBundleReviewDecision(
                choice=operation_choice,
                inactive_consent=self.query_one(
                    "#stts-bundle-inactive-consent", Checkbox
                ).value,
            )
        )


_OwnedProfileModal = (
    TTSProfileEditorModal
    | TTSProfileDeleteModal
    | TTSProfileExportChoiceModal
    | TTSVoiceBundleConsentModal
    | TTSVoiceBundleReviewModal
)
_RetainedEditorDraft = tuple[tuple[int, UUID], TTSProfileDraft]


class STTSProfileLibrary(Widget):
    """Bounded profile list whose repository rows precede capability status."""

    DEFAULT_CSS = """
    STTSProfileLibrary {
        height: 100%;
        width: 100%;
        background: $background;
        padding: 0 1;
    }

    #stts-profile-header {
        height: auto;
        background: $panel;
        border: tall $accent;
        padding: 0 1;
    }

    #stts-profile-title {
        text-style: bold;
    }

    #stts-profile-purpose {
        color: $text-muted;
    }

    #stts-profile-toolbar {
        height: 3;
    }

    #stts-profile-search {
        width: 1fr;
        height: 3;
        border: round $surface-lighten-1;
        background: $surface-darken-1;
    }

    #stts-profile-search:focus {
        border: round $accent;
    }

    #stts-profile-toolbar Button,
    #stts-profile-actions Button {
        width: auto;
        height: 3;
        min-width: 0;
        border: none;
    }

    #stts-profile-toolbar Button {
        padding: 0 1;
    }

    #stts-profile-actions Button {
        padding: 0;
    }

    #stts-profile-toolbar Button:focus,
    #stts-profile-actions Button:focus {
        outline: heavy $accent;
    }

    #stts-profile-table {
        height: 1fr;
        min-height: 3;
        background: $surface;
        border: round $surface-lighten-1;
    }

    #stts-profile-status {
        height: auto;
        max-height: 6;
        background: $panel;
        border: round $surface-lighten-1;
        color: $text;
        padding: 0 1;
    }

    #stts-profile-status-copy {
        width: 100%;
        height: auto;
    }

    #stts-profile-status-copy.selected-detail {
        height: 2;
        min-height: 2;
        max-height: 2;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    #stts-profile-identifiers {
        display: none;
        width: 100%;
        height: 1;
        min-height: 1;
        max-height: 1;
        border: none;
        padding: 0;
        background: $panel;
        color: $text;
        scrollbar-size-horizontal: 0;
    }

    #stts-profile-dependency-actions {
        display: none;
        width: 100%;
        height: 1;
        min-height: 1;
    }

    #stts-profile-dependency-actions Button {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        border: none;
        padding: 0 1;
        margin-right: 1;
    }

    #stts-profile-actions {
        height: 3;
        align-horizontal: right;
    }

    #stts-profile-delete-btn {
        background: $error;
    }

    #stts-profile-actions Button:disabled {
        color: $text-disabled;
        background: $surface-darken-1;
        opacity: 60%;
    }
    """

    def __init__(
        self,
        service_loader: ProfileServiceLoader,
        *,
        default_profile_id_reader: Callable[[], object | None] | None = None,
        voice_bundle_service_loader: VoiceBundleServiceLoader | None = None,
        bundle_platform_supported: bool | None = None,
        continuity: ProfileLibraryContinuity | None = None,
        pending_verification: ProfileVerificationResult | None = None,
        focus_restore_token: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._service_loader = service_loader
        # Task 5 (slice 3): reads the persisted `[app_tts] default_profile_id`
        # setting -- injected exactly like `TTSEventHandler`'s own reader
        # (`Event_Handlers/TTS_Events/tts_events.py`, wired the same way from
        # `app.py`) so tests never touch real config and `None` means "no
        # app-default voice is wired up here." The profile *store* has no way
        # to know this fact (it lives in config, not a repository row), so it
        # is supplied here, one layer up, rather than bent into
        # `assignment_count`.
        self._default_profile_id_reader = default_profile_id_reader
        self._voice_bundle_service_loader = voice_bundle_service_loader
        self._voice_bundle_service: _VoiceBundleService | None = None
        self._bundle_platform_supported = (
            os.name == "posix"
            if bundle_platform_supported is None
            else bundle_platform_supported
        )
        self._active_bundle_handle: TTSVoiceBundleHandle | None = None
        self._bundle_invalidation_tasks: dict[
            TTSVoiceBundleHandle, asyncio.Task[None]
        ] = {}
        self._portability_request_id = 0
        self._service: _ProfileService | None = None
        self._live = False
        self._mount_token = 0
        self._request_id = 0
        self._active_page_task: asyncio.Task[None] | None = None
        self._active_page_phase = "idle"
        # Cleanup-only work has no current request token or publication
        # authority. It is bounded separately from the one active pipeline.
        self._retained_cleanup_task: asyncio.Task[None] | None = None
        self._pending_page_request: _PageRequest | None = None
        self._search_timer: Timer | None = None
        self._rendered_request: _PageRequest | None = None
        self._rendered_repository_generation: int | None = None
        self._rendered_profile_ids: tuple[str, ...] = ()
        self._availability_configuration_revision: int | None = None
        self._availability_catalog_revision: int | None = None
        self._loaded_rows: dict[str, LoadedTTSProfile] = {}
        self._row_availability: dict[str, TTSProfileAvailability] = {}
        self._selected_profile: LoadedTTSProfile | None = None
        self._initial_continuity = continuity
        self._post_load_continuity: ProfileLibraryContinuity | None = None
        self._refresh_continuity: ProfileLibraryContinuity | None = None
        self._pending_verification = pending_verification
        self._focus_restore_token = focus_restore_token
        self._retained_editor_draft: _RetainedEditorDraft | None = None
        self._active_modal: _OwnedProfileModal | None = None
        self._search = None if continuity is None else continuity.search
        self._offset = 0 if continuity is None else continuity.offset
        self._total = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="stts-profile-header"):
            yield Label("Voice profiles", id="stts-profile-title")
            yield Static(
                "Manage exact model and voice selections for every speech provider.",
                id="stts-profile-purpose",
            )
        with Horizontal(id="stts-profile-toolbar"):
            yield Input(
                placeholder="Search voice profiles",
                id="stts-profile-search",
            )
            yield Button(
                "Previous",
                id="stts-profile-previous-btn",
                disabled=True,
            )
            yield Button("Next", id="stts-profile-next-btn", disabled=True)
        yield DataTable(
            id="stts-profile-table",
            cursor_type="row",
            zebra_stripes=True,
        )
        with Vertical(id="stts-profile-status"):
            yield Static(
                Text(_PROFILE_LOADING_COPY),
                id="stts-profile-status-copy",
            )
            yield TextArea(
                "",
                id="stts-profile-identifiers",
                read_only=True,
                soft_wrap=False,
                show_line_numbers=False,
                compact=True,
            )
            with Horizontal(id="stts-profile-dependency-actions", classes="hidden"):
                yield Button(
                    "Recovery",
                    id="stts-profile-dependency-primary-btn",
                    classes="hidden",
                    disabled=True,
                )
                yield Button(
                    "Recovery",
                    id="stts-profile-dependency-advisory-btn",
                    classes="hidden",
                    disabled=True,
                )
        with Horizontal(id="stts-profile-actions"):
            yield Button(
                "Preview",
                id="stts-profile-preview-btn",
                disabled=True,
                variant="primary",
            )
            yield Button("Edit", id="stts-profile-edit-btn", disabled=True)
            yield Button(
                "Duplicate",
                id="stts-profile-duplicate-btn",
                disabled=True,
            )
            yield Button(
                "Export",
                id="stts-profile-export-btn",
                disabled=True,
            )
            import_button = Button(
                "Import bundle",
                id="stts-profile-import-btn",
                disabled=(
                    self._voice_bundle_service_loader is None
                    or not self._bundle_platform_supported
                ),
            )
            import_button.tooltip = (
                "Review and import a portable voice bundle."
                if self._bundle_platform_supported
                else PROFILE_BUNDLE_UNSUPPORTED_COPY
            )
            yield import_button
            yield Button("Refresh", id="stts-profile-refresh-btn")
            yield Button(
                "Delete",
                id="stts-profile-delete-btn",
                disabled=True,
                variant="error",
            )

    def on_mount(self) -> None:
        self._live = True
        self._mount_token += 1
        table = self.query_one("#stts-profile-table", DataTable)
        columns = table.add_columns(
            "Name",
            "Model",
            "Voice",
            "Availability",
            "Revision",
        )
        self._availability_column = columns[3]
        # Dynamic Speech navigation can mount this parent before the trailing
        # action buttons have completed their own mount cycle.
        self.call_after_refresh(
            self._queue_page_request,
            self._search,
            self._offset,
        )

    async def on_unmount(self) -> None:
        self._live = False
        self._mount_token += 1
        self._pending_page_request = None
        self._retained_editor_draft = None
        modal = self._active_modal
        if modal is not None:
            self._dismiss_owned_modal(modal)
        self._active_modal = None
        self._portability_request_id += 1
        handle = self._active_bundle_handle
        self._active_bundle_handle = None
        if handle is not None:
            await self._invalidate_bundle_handle(handle)
        timer = self._search_timer
        self._search_timer = None
        if timer is not None:
            timer.stop()
        tasks = tuple(
            task
            for task in (
                self._active_page_task,
                self._retained_cleanup_task,
            )
            if task is not None
        )
        self._active_page_task = None
        self._active_page_phase = "idle"
        self._retained_cleanup_task = None
        for task in tasks:
            if not task.done():
                task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                continue
            except Exception:  # noqa: BLE001,S112 - settle cleanup safely
                continue

    def _queue_page_request(
        self,
        search: str | None,
        offset: int,
    ) -> None:
        if not self._live:
            return
        same_page = search == self._search and max(0, offset) == self._offset
        if same_page and self._selected_profile is not None:
            self._refresh_continuity = self.navigation_continuity()
        else:
            self._refresh_continuity = None
        self._request_id += 1
        self._search = search
        self._offset = max(0, offset)
        request = _PageRequest(
            mount_token=self._mount_token,
            request_id=self._request_id,
            search=self._search,
            offset=self._offset,
        )
        if not same_page:
            self._selected_profile = None
        self._sync_selected_actions()
        self._set_status(_PROFILE_LOADING_COPY)
        task = self._active_page_task
        if task is not None and not task.done():
            self._pending_page_request = request
            if self._active_page_phase == "availability":
                if task.cancelling() == 0:
                    task.cancel()
                if self._retained_cleanup_slot_available():
                    self._detach_active_cleanup(task)
            return
        self._start_page_pipeline(request)

    def navigation_continuity(self) -> ProfileLibraryContinuity:
        """Capture bounded process-local UI intent without retaining widgets."""

        table = self.query_one("#stts-profile-table", DataTable)
        selected_id = (
            None
            if self._selected_profile is None
            else self._selected_profile.profile.profile_id
        )
        focused = self.app.focused
        focused_id = None if focused is None else focused.id
        focus_target = (
            focused_id if focused_id in _PROFILE_FOCUS_TARGETS else None
        )
        return ProfileLibraryContinuity(
            selected_profile_id=selected_id,
            cursor_row=max(0, min(table.cursor_row, PROFILE_PAGE_SIZE - 1)),
            scroll_x=max(0, table.scroll_offset.x),
            scroll_y=max(0, table.scroll_offset.y),
            focus_target=focus_target,
            search=self._search,
            offset=max(0, self._offset),
        )

    def _start_page_pipeline(self, request: _PageRequest) -> None:
        self._pending_page_request = None
        task = asyncio.create_task(
            self._run_page_pipeline(request),
            name=f"stts_profile_page_{request.request_id}",
        )
        self._active_page_task = task
        self._active_page_phase = "service"
        task.add_done_callback(self._page_pipeline_done)

    @staticmethod
    def _consume_page_task(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()

    def _page_pipeline_done(self, task: asyncio.Task[None]) -> None:
        self._consume_page_task(task)
        if self._active_page_task is not task:
            return
        self._active_page_task = None
        self._active_page_phase = "idle"
        request = self._pending_page_request
        self._pending_page_request = None
        if self._live and request is not None:
            self._start_page_pipeline(request)

    def _retained_cleanup_slot_available(self) -> bool:
        retained = self._retained_cleanup_task
        if retained is None:
            return True
        if not retained.done():
            return False
        self._consume_page_task(retained)
        if self._retained_cleanup_task is retained:
            self._retained_cleanup_task = None
        return True

    def _detach_active_cleanup(self, task: asyncio.Task[None]) -> None:
        """Retain stale cleanup without granting page publication authority."""

        if (
            self._active_page_task is not task
            or self._retained_cleanup_task is not None
        ):
            return
        request = self._pending_page_request
        self._pending_page_request = None
        self._active_page_task = None
        self._active_page_phase = "idle"
        self._retained_cleanup_task = task
        task.add_done_callback(self._retained_cleanup_done)
        if self._live and request is not None:
            self._start_page_pipeline(request)

    def _retained_cleanup_done(self, task: asyncio.Task[None]) -> None:
        self._consume_page_task(task)
        if self._retained_cleanup_task is not task:
            return
        self._retained_cleanup_task = None
        active = self._active_page_task
        if (
            self._live
            and self._pending_page_request is not None
            and active is not None
            and not active.done()
            and self._active_page_phase == "availability"
        ):
            if active.cancelling() == 0:
                active.cancel()
            self._detach_active_cleanup(active)

    async def _run_page_pipeline(self, request: _PageRequest) -> None:
        try:
            service = self._service
            if service is None:
                service = await self._service_loader()
                if not self._request_is_current(request):
                    return
                self._service = service
            if service is None:
                if self._request_is_current(request):
                    self._publish_unavailable()
                return

            self._active_page_phase = "list"
            try:
                page = await service.list_profiles(
                    search=request.search,
                    offset=request.offset,
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - bounded failure only
                store_unavailable = _is_store_unavailable(error)
                if store_unavailable and self._service is service:
                    self._service = None
                if self._request_is_current(request):
                    if store_unavailable:
                        self._publish_unavailable()
                    else:
                        self._set_status(_PROFILE_LOAD_FAILED_COPY)
                return
            if not self._page_can_publish(request, page):
                return
            self._publish_page(request, page)
            await asyncio.sleep(0)
            if not self._rendered_page_is_current(request, page):
                return

            self._active_page_phase = "availability"
            try:
                availability = await service.observe_availability(page)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - publish bounded failure only
                if (
                    self._rendered_page_is_current(
                        request,
                        page,
                    )
                    and self._availability_status_can_publish()
                ):
                    self._set_status(_PROFILE_AVAILABILITY_FAILED_COPY)
                return
            if self._availability_can_publish(request, page, availability):
                pending_loaded = self._pending_verification_target(page)
                self._publish_availability(
                    page,
                    availability,
                    skip_profile_id=(
                        None
                        if pending_loaded is None
                        else pending_loaded.profile.profile_id
                    ),
                )
                await self._reconcile_pending_verification(
                    service,
                    request,
                    page,
                    pending_loaded,
                )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - publish bounded failure only
            if self._request_is_current(request):
                self._set_status(_PROFILE_LOAD_FAILED_COPY)

    def _request_is_current(self, request: _PageRequest) -> bool:
        return (
            self._live
            and request.mount_token == self._mount_token
            and request.request_id == self._request_id
            and request.search == self._search
            and request.offset == self._offset
        )

    def _page_can_publish(
        self,
        request: _PageRequest,
        page: object,
    ) -> bool:
        if not self._request_is_current(request):
            return False
        if type(page) is not TTSProfilePageSnapshot:
            self._set_status(_PROFILE_LOAD_FAILED_COPY)
            return False
        typed_page = cast(TTSProfilePageSnapshot, page)
        if len(typed_page.profiles) > PROFILE_PAGE_SIZE:
            self._set_status(_PROFILE_LOAD_FAILED_COPY)
            return False
        generation = self._rendered_repository_generation
        if generation is not None and typed_page.repository_generation < generation:
            self._set_status(_PROFILE_STALE_COPY)
            return False
        return True

    def _rendered_page_is_current(
        self,
        request: _PageRequest,
        page: TTSProfilePageSnapshot,
    ) -> bool:
        return (
            self._request_is_current(request)
            and self._rendered_request == request
            and self._rendered_repository_generation == page.repository_generation
            and self._rendered_profile_ids
            == tuple(str(profile.profile_id) for profile in page.profiles)
        )

    def _availability_can_publish(
        self,
        request: _PageRequest,
        page: TTSProfilePageSnapshot,
        snapshot: object,
    ) -> bool:
        if not self._rendered_page_is_current(request, page):
            return False
        if type(snapshot) is not TTSProfileAvailabilitySnapshot:
            if self._availability_status_can_publish():
                self._set_status(_PROFILE_AVAILABILITY_FAILED_COPY)
            return False
        typed_snapshot = cast(TTSProfileAvailabilitySnapshot, snapshot)
        if typed_snapshot.repository_generation != page.repository_generation:
            return False
        if tuple(str(item.profile_id) for item in typed_snapshot.profiles) != (
            self._rendered_profile_ids
        ):
            return False

        current_configuration = self._availability_configuration_revision
        if current_configuration is None:
            return True
        if typed_snapshot.configuration_revision < current_configuration:
            return False
        if typed_snapshot.configuration_revision > current_configuration:
            return True

        current_catalog = self._availability_catalog_revision
        incoming_catalog = typed_snapshot.catalog_revision
        if current_catalog is None:
            return True
        if incoming_catalog is None:
            return False
        return incoming_catalog >= current_catalog

    def _availability_status_can_publish(self) -> bool:
        return self._selected_profile is None or self.query_one(
            "#stts-profile-status-copy",
            Static,
        ).has_class("selected-detail")

    def _publish_unavailable(self) -> None:
        self._clear_rows()
        self._set_status(PROFILE_STORE_UNAVAILABLE_COPY)

    def _publish_page(
        self,
        request: _PageRequest,
        page: TTSProfilePageSnapshot,
    ) -> None:
        retained = self._retained_editor_draft
        if retained is not None and retained[0][0] != page.repository_generation:
            self._retained_editor_draft = None
        previous_rows = tuple(loaded.profile for loaded in self._loaded_rows.values())
        preserve_availability = (
            self._rendered_repository_generation == page.repository_generation
            and previous_rows == page.profiles
        )
        previous_availability = (
            dict(self._row_availability) if preserve_availability else {}
        )
        table = self.query_one("#stts-profile-table", DataTable)
        table.clear(columns=False)
        self._loaded_rows.clear()
        self._row_availability = previous_availability
        continuity = self._refresh_continuity or self._initial_continuity
        self._refresh_continuity = None
        self._initial_continuity = None
        self._selected_profile = None
        self._post_load_continuity = continuity
        self._rendered_request = request
        self._rendered_repository_generation = page.repository_generation
        self._rendered_profile_ids = tuple(
            str(profile.profile_id) for profile in page.profiles
        )
        if not preserve_availability:
            self._availability_configuration_revision = None
            self._availability_catalog_revision = None
        self._total = page.total
        self._sync_selected_actions()

        for profile in page.profiles:
            key = str(profile.profile_id)
            loaded = LoadedTTSProfile(
                repository_generation=page.repository_generation,
                profile=profile,
            )
            self._loaded_rows[key] = loaded
            availability = self._row_availability.get(key)
            table.add_row(
                Text(profile.display_name),
                Text(profile.model_id),
                Text(
                    profile.voice_id
                    if profile.voice_id is not None
                    else "Server default"
                ),
                Text(_availability_cell_text(availability)),
                Text(str(profile.revision)),
                key=key,
            )

        if continuity is not None:
            selected_key = (
                None
                if continuity.selected_profile_id is None
                else str(continuity.selected_profile_id)
            )
            selected = self._loaded_rows.get(selected_key)
            if selected is not None:
                self._selected_profile = selected
                selected_row = self._rendered_profile_ids.index(selected_key)
                table.move_cursor(row=selected_row, animate=False)
            elif page.profiles:
                table.move_cursor(
                    row=min(continuity.cursor_row, len(page.profiles) - 1),
                    animate=False,
                )
            self.call_after_refresh(self._restore_navigation_viewport, continuity)

        self._sync_paging()
        self._sync_selected_actions()
        if page.profiles:
            self._set_status(
                f"{len(page.profiles)} voice profiles loaded. "
                "Checking current availability…"
            )
        else:
            self._set_status(_PROFILE_EMPTY_COPY)

    def _restore_navigation_viewport(
        self,
        continuity: ProfileLibraryContinuity,
    ) -> None:
        """Restore focus and scroll after the remounted table has final geometry."""

        if not self._live:
            return
        table = self.query_one("#stts-profile-table", DataTable)
        table.scroll_to(
            x=continuity.scroll_x,
            y=continuity.scroll_y,
            animate=False,
        )
        if self._post_load_continuity is continuity:
            self._post_load_continuity = None
        token = self._focus_restore_token
        self._focus_restore_token = None
        if token is not None:
            self.post_message(ProfileLibraryRestoreReady(token))

    def _publish_availability(
        self,
        page: TTSProfilePageSnapshot,
        snapshot: TTSProfileAvailabilitySnapshot,
        *,
        skip_profile_id: UUID | None = None,
    ) -> None:
        table = self.query_one("#stts-profile-table", DataTable)
        self._row_availability.clear()
        for item in snapshot.profiles:
            if item.profile_id == skip_profile_id:
                continue
            key = str(item.profile_id)
            self._row_availability[key] = item
            table.update_cell(
                key,
                self._availability_column,
                Text(_availability_cell_text(item)),
            )
        self._availability_configuration_revision = snapshot.configuration_revision
        self._availability_catalog_revision = snapshot.catalog_revision
        self._sync_selected_actions()
        if self._selected_profile is not None:
            if self._availability_status_can_publish():
                self._show_selected_detail()
            return
        self._set_status(
            f"{len(page.profiles)} voice profiles loaded. "
            "Availability is current for this page."
        )

    def _pending_verification_target(
        self,
        page: TTSProfilePageSnapshot,
    ) -> LoadedTTSProfile | None:
        pending = self._pending_verification
        if pending is None:
            return None
        for profile in page.profiles:
            if (
                profile.profile_id == pending.profile_id
                and profile.revision == pending.profile_revision
                and page.repository_generation == pending.repository_generation
            ):
                return LoadedTTSProfile(
                    repository_generation=page.repository_generation,
                    profile=profile,
                )
        return None

    async def _reconcile_pending_verification(
        self,
        service: _ProfileService,
        request: _PageRequest,
        page: TTSProfilePageSnapshot,
        loaded: LoadedTTSProfile | None,
    ) -> None:
        """Re-query one exact row and retire stale/edit/delete handoffs."""

        pending = self._pending_verification
        if pending is None:
            return
        try:
            if loaded is not None:
                target_page = TTSProfilePageSnapshot(
                    repository_generation=loaded.repository_generation,
                    profiles=(loaded.profile,),
                    total=1,
                )
                snapshot = await service.observe_availability(target_page)
                availability = (
                    snapshot.profiles[0]
                    if type(snapshot) is TTSProfileAvailabilitySnapshot
                    and snapshot.repository_generation == loaded.repository_generation
                    and len(snapshot.profiles) == 1
                    else None
                )
                if (
                    self._rendered_page_is_current(request, page)
                    and availability is not None
                    and availability.profile_id == loaded.profile.profile_id
                    and availability.state == "available"
                    and pending.availability.state == "available"
                ):
                    self.publish_profile_test_availability(loaded, availability)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - retain bounded library copy
            if self._rendered_page_is_current(request, page):
                self._set_status(_PROFILE_AVAILABILITY_FAILED_COPY)
        finally:
            if self._pending_verification is pending:
                self._pending_verification = None
                if self._live:
                    self.post_message(ProfileVerificationReconciled(pending))

    def publish_profile_test_availability(
        self,
        loaded: LoadedTTSProfile,
        availability: TTSProfileAvailability,
    ) -> None:
        """Refresh one still-current row after matching sample evidence."""

        if not self._live or type(availability) is not TTSProfileAvailability:
            return
        key = str(loaded.profile.profile_id)
        current = self._loaded_rows.get(key)
        if current != loaded or availability.profile_id != loaded.profile.profile_id:
            return
        self._row_availability[key] = availability
        try:
            self.query_one("#stts-profile-table", DataTable).update_cell(
                key,
                self._availability_column,
                Text(_availability_cell_text(availability)),
            )
        except Exception:  # noqa: BLE001 - a stale row cannot publish
            return
        self._sync_selected_actions()
        if self._selected_profile == loaded:
            self._show_selected_detail()

    def _clear_rows(self) -> None:
        self.query_one("#stts-profile-table", DataTable).clear(columns=False)
        self._loaded_rows.clear()
        self._row_availability.clear()
        self._selected_profile = None
        self._rendered_request = None
        self._rendered_repository_generation = None
        self._rendered_profile_ids = ()
        self._availability_configuration_revision = None
        self._availability_catalog_revision = None
        self._total = 0
        self._sync_selected_actions()
        self._sync_paging()

    def _set_status(self, copy: str) -> None:
        if not self.is_mounted:
            return
        status = self.query_one("#stts-profile-status-copy", Static)
        status.remove_class("selected-detail")
        status.update(Text(copy))
        identifiers = self.query_one("#stts-profile-identifiers", TextArea)
        if identifiers.has_focus:
            self.query_one("#stts-profile-table", DataTable).focus()
        identifiers.display = False
        identifiers.load_text("")
        self._render_dependency_actions(())

    def _render_dependency_actions(
        self,
        actions: tuple[TTSProfileDependencyActionProjection, ...],
    ) -> None:
        """Render blocker then advisory from their immutable projections."""

        actions_by_role = {action.role: action for action in actions}
        container = self.query_one("#stts-profile-dependency-actions", Horizontal)
        container.display = bool(actions)
        container.set_class(not actions, "hidden")
        role_selectors: tuple[tuple[Literal["blocker", "advisory"], str], ...] = (
            ("blocker", "#stts-profile-dependency-primary-btn"),
            ("advisory", "#stts-profile-dependency-advisory-btn"),
        )
        for role, selector in role_selectors:
            button = self.query_one(selector, Button)
            action = actions_by_role.get(role)
            button.display = action is not None
            button.set_class(action is None, "hidden")
            button.disabled = action is None
            button.label = "Recovery" if action is None else action.label
            button.tooltip = None if action is None else action.tooltip

    def _sync_selected_actions(self) -> None:
        base_disabled = (
            self._selected_profile is None or not self._rendered_request_is_current()
        )
        for selector in (
            "#stts-profile-edit-btn",
            "#stts-profile-duplicate-btn",
            "#stts-profile-export-btn",
            "#stts-profile-delete-btn",
        ):
            self.query_one(selector, Button).disabled = base_disabled
        preview = self.query_one("#stts-profile-preview-btn", Button)
        availability = None
        if self._selected_profile is not None:
            availability = self._row_availability.get(
                str(self._selected_profile.profile.profile_id)
            )
        if availability is None:
            preview.label = "Checking"
            preview.disabled = True
        elif availability.state == "unavailable":
            preview.label = "Unavailable"
            preview.disabled = True
        elif availability.state == "unverified":
            preview.label = "Test in Playground"
            preview.disabled = base_disabled
        else:
            preview.label = "Preview"
            preview.disabled = base_disabled

    def _sync_paging(self) -> None:
        rendered_offset = (
            0 if self._rendered_request is None else self._rendered_request.offset
        )
        self.query_one("#stts-profile-previous-btn", Button).disabled = (
            rendered_offset <= 0
        )
        self.query_one("#stts-profile-next-btn", Button).disabled = (
            rendered_offset + len(self._loaded_rows) >= self._total
        )

    @on(DataTable.RowHighlighted, "#stts-profile-table")
    def _handle_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """A single click, or an arrow key, landing on a profile row.

        Textual fires `RowSelected` on *activation* -- Enter, or a second click
        -- so handling only that left a clicked profile highlighted but not
        selected, with the row actions still disabled (TASK-1180).

        This pane cannot use `DataTableClickSelectMixin`: the mixin forwards to
        a conventionally-named `on_data_table_row_selected`, and this handler is
        bound by an `@on` selector so it only applies to this one table. The
        pairing is explicit here instead.
        """
        if event.row_key is None or event.row_key.value is None:
            return
        # Only a focused table's cursor is being moved by a person; an
        # unfocused one is rebuilding its own rows, and forwarding that would
        # let a refresh select on the user's behalf. Same gate the mixin
        # applies, spelled out here because this pane cannot use it.
        table = getattr(event, "data_table", None)
        if table is None or not table.has_focus:
            return
        event.stop()
        self._handle_row_selected(
            DataTable.RowSelected(table, event.cursor_row, event.row_key)
        )

    @on(DataTable.RowSelected, "#stts-profile-table")
    def _handle_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        if not self._rendered_request_is_current():
            self._selected_profile = None
            self._sync_selected_actions()
            return
        loaded = self._loaded_rows.get(str(event.row_key.value))
        if loaded is None:
            return
        self._selected_profile = loaded
        self._sync_selected_actions()
        self._show_selected_detail()

    def _show_selected_detail(self) -> None:
        loaded = self._selected_profile
        if loaded is None:
            return
        profile = loaded.profile
        availability = self._row_availability.get(str(profile.profile_id))
        # Routed through the same helper as the DataTable cell: `state` is
        # only used below when neither of the `elif` branches applies (i.e.
        # never for "unverified" today), but computing it from the raw
        # `.state.title()` left a trap for a future branch reorder to leak
        # the bare "Unverified" word for a legacy no-catalog-check profile.
        state = _availability_cell_text(availability)
        voice = profile.voice_id if profile.voice_id is not None else "Server default"
        if availability is not None and availability.dependency.display:
            status_line = availability.dependency.display
        elif availability is not None and availability.state == "unavailable":
            status_line = "Unavailable — Refresh, then Edit."
        elif availability is not None and availability.state == "unverified":
            status_line = "Needs test — Open in Playground."
        else:
            status_line = f"{state}."
        if availability is not None and availability.dependency.advisory_display:
            status_line = (
                f"{status_line} {availability.dependency.advisory_display}. "
                "Preview or generate the voice, save it as a new profile, then "
                "reassign or remove the old profile."
            )
        self._render_dependency_actions(
            ()
            if availability is None
            else dependency_recovery_actions(availability.dependency)
        )
        status = self.query_one("#stts-profile-status-copy", Static)
        status.add_class("selected-detail")
        status.update(Text(f"{status_line}\nSelected: {profile.display_name}"))
        identifiers = self.query_one("#stts-profile-identifiers", TextArea)
        identifiers.load_text(f"{profile.provider_id} / {profile.model_id} / {voice}")
        identifiers.move_cursor((0, 0))
        identifiers.scroll_home(animate=False)
        identifiers.display = True

    def _action_target_is_current(self, loaded: LoadedTTSProfile) -> bool:
        return self._rendered_request_is_current() and self._selected_profile is loaded

    def _rendered_request_is_current(self) -> bool:
        request = self._rendered_request
        return request is not None and self._request_is_current(request)

    @staticmethod
    def _dismiss_owned_modal(modal: _OwnedProfileModal) -> None:
        if modal.is_mounted and modal.is_current:
            modal.dismiss(False if isinstance(modal, TTSProfileDeleteModal) else None)

    async def _push_owned_modal(self, modal: _OwnedProfileModal) -> object:
        active = self._active_modal
        if active is not None:
            self._dismiss_owned_modal(active)
        self._active_modal = modal
        try:
            return await self.app.push_screen_wait(modal)
        except asyncio.CancelledError:
            self._dismiss_owned_modal(modal)
            raise
        finally:
            if self._active_modal is modal:
                self._active_modal = None

    async def _service_for_action(self) -> _ProfileService | None:
        service = self._service
        if service is not None:
            return service
        try:
            service = await self._service_loader()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - isolate optional profile storage
            self._set_status(PROFILE_STORE_UNAVAILABLE_COPY)
            return None
        if service is None:
            self._set_status(PROFILE_STORE_UNAVAILABLE_COPY)
            return None
        if self._live:
            self._service = service
            return service
        return None

    async def _assignment_count(
        self,
        service: _ProfileService,
        loaded: LoadedTTSProfile,
    ) -> int | None:
        try:
            count = await service.assignment_count(loaded)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - map to bounded UI copy
            self._set_status(profile_action_error_copy(error))
            return None
        if type(count) is not int or count < 0:
            self._set_status(PROFILE_ACTION_FAILED_COPY)
            return None
        return count

    def _is_configured_default_profile(self, loaded: LoadedTTSProfile) -> bool:
        """Return whether `loaded` is the current `[app_tts]
        default_profile_id` -- read fresh at call time, never cached, since
        the setting can change between this widget mounting and a delete
        attempt.

        Normalizes exactly like `tts_events.py::_read_default_profile_id`:
        an absent reader, a non-string value, or a blank one all mean "no
        app-default configured." A malformed non-UUID string (Task 2's
        defined dangling state) can never equal a real profile's UUID, so
        it naturally resolves to `False` here too, with no separate case
        needed -- it can never be describing `loaded`.
        """

        reader = self._default_profile_id_reader
        if reader is None:
            return False
        raw_value = reader()
        if not isinstance(raw_value, str):
            return False
        stripped = raw_value.strip()
        if not stripped:
            return False
        try:
            configured_id = UUID(stripped)
        except (ValueError, AttributeError, TypeError):
            return False
        return configured_id == loaded.profile.profile_id

    async def create_from_artifact(
        self,
        display_name: str,
        artifact: STTSGeneratedAudio,
    ) -> LoadedTTSProfile | None:
        """Create from exact immutable provenance without rereading controls."""

        service = await self._service_for_action()
        if service is None:
            return None
        try:
            loaded = await service.create_from_artifact(display_name, artifact)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - map to bounded UI copy
            # `create_from_artifact` checks the exact provider's
            # configuration revision with no provider gate (see the trace
            # above `_PROFILE_UNVERIFIED_COPY`), so `stale_configuration` is
            # reachable for a legacy provider here -- unlike the other
            # actions on this widget. Mirror the two live callers
            # (`STTS_Window.py`, `speech_profile_mixin.py`), which already
            # special-case it with provider-agnostic copy instead of the
            # "Refresh and retry" toast that only makes sense for audio_cpp.
            copy = (
                _PROFILE_RESULT_STALE_COPY
                if getattr(error, "code", None) == "stale_configuration"
                else profile_action_error_copy(error)
            )
            self._set_status(copy)
            return None
        if self._live:
            self._queue_page_request(self._search, self._offset)
        return loaded

    async def edit_selected_profile(self) -> LoadedTTSProfile | None:
        """Edit the exact selected loaded version through the focused modal."""

        loaded = self._selected_profile
        if loaded is None:
            return None
        service = await self._service_for_action()
        if service is None:
            return None
        count = await self._assignment_count(service, loaded)
        if count is None or not self._action_target_is_current(loaded):
            return None

        loaded_key = (loaded.repository_generation, loaded.profile.profile_id)
        retained = self._retained_editor_draft
        initial_draft = (
            retained[1] if retained is not None and retained[0] == loaded_key else None
        )
        try:
            draft = await self._push_owned_modal(
                TTSProfileEditorModal(
                    loaded,
                    assignment_count=count,
                    mode="edit",
                    initial_draft=initial_draft,
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - isolate modal lifecycle failure
            self._set_status(PROFILE_ACTION_FAILED_COPY)
            return None
        if draft is None or not self._action_target_is_current(loaded):
            return None
        if type(draft) is not TTSProfileDraft:
            self._set_status(_PROFILE_VALIDATION_COPY)
            return None
        try:
            updated = await service.update_profile(loaded, draft)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - map to bounded UI copy
            if isinstance(error, ProfileRepositoryError) and error.code in {
                "conflict",
                "stale",
            }:
                self._retained_editor_draft = (loaded_key, draft)
            self._set_status(profile_action_error_copy(error))
            return None

        self._retained_editor_draft = None
        if self._live:
            self._queue_page_request(self._search, self._offset)
        return updated

    async def duplicate_selected_profile(self) -> LoadedTTSProfile | None:
        """Duplicate the exact selected version under an explicit new name."""

        loaded = self._selected_profile
        if loaded is None:
            return None
        service = await self._service_for_action()
        if service is None:
            return None
        count = await self._assignment_count(service, loaded)
        if count is None or not self._action_target_is_current(loaded):
            return None

        try:
            draft = await self._push_owned_modal(
                TTSProfileEditorModal(
                    loaded,
                    assignment_count=count,
                    mode="duplicate",
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - isolate modal lifecycle failure
            self._set_status(PROFILE_ACTION_FAILED_COPY)
            return None
        if draft is None or not self._action_target_is_current(loaded):
            return None
        if type(draft) is not TTSProfileDraft:
            self._set_status(_PROFILE_VALIDATION_COPY)
            return None
        try:
            duplicated = await service.duplicate_profile(
                loaded,
                cast(TTSProfileDraft, draft).display_name,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - map to bounded UI copy
            self._set_status(profile_action_error_copy(error))
            return None
        if self._live:
            self._queue_page_request(self._search, self._offset)
        return duplicated

    async def delete_selected_profile(self) -> bool:
        """Delete only after an advisory count and final repository check."""

        loaded = self._selected_profile
        if loaded is None:
            return False
        service = await self._service_for_action()
        if service is None:
            return False
        count = await self._assignment_count(service, loaded)
        if count is None or not self._action_target_is_current(loaded):
            return False

        try:
            confirmed = await self._push_owned_modal(
                TTSProfileDeleteModal(
                    display_name=loaded.profile.display_name,
                    assignment_count=count,
                    is_app_default=self._is_configured_default_profile(loaded),
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - isolate modal lifecycle failure
            self._set_status(PROFILE_ACTION_FAILED_COPY)
            return False
        if not self._action_target_is_current(loaded):
            return False
        if count > 0:
            self._set_status(PROFILE_DELETE_PROTECTED_COPY)
            return False
        if confirmed is not True:
            return False

        try:
            await service.delete_profile(loaded)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - map to bounded UI copy
            self._set_status(profile_action_error_copy(error))
            return False
        if self._live:
            self._queue_page_request(self._search, self._offset)
        return True

    async def _choose_profile_export_path(self) -> Path | None:
        """Choose a destination for one standalone sanitized profile payload."""

        from tldw_chatbook.Third_Party.textual_fspicker import Filters
        from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileSave

        picker = EnhancedFileSave(
            title="Export voice profile",
            default_filename="voice-profile.json",
            filters=Filters(
                ("JSON Files", lambda path: path.suffix.lower() == ".json"),
                ("All Files", lambda _path: True),
            ),
            context="tts_profile_export",
        )
        selected = await self.app.push_screen_wait(picker)
        if selected is None:
            return None
        return Path(str(selected))

    async def _voice_bundle_service_for_action(self) -> _VoiceBundleService | None:
        service = self._voice_bundle_service
        if service is not None:
            return service
        loader = self._voice_bundle_service_loader
        if loader is None or not self._bundle_platform_supported:
            self._set_status(PROFILE_BUNDLE_UNSUPPORTED_COPY)
            return None
        try:
            service = await loader()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - bounded copy only
            service = None
        if service is None:
            self._set_status(PROFILE_ACTION_FAILED_COPY)
            return None
        self._voice_bundle_service = service
        return service

    async def _invalidate_bundle_handle(self, handle: TTSVoiceBundleHandle) -> None:
        service = self._voice_bundle_service
        if service is None:
            return
        task = self._bundle_invalidation_tasks.get(handle)
        if task is None:
            task = asyncio.create_task(
                service.invalidate(handle),
                name="invalidate_tts_voice_bundle_review",
            )
            self._bundle_invalidation_tasks[handle] = task

            def _release_completed(candidate: asyncio.Task[None]) -> None:
                if self._bundle_invalidation_tasks.get(handle) is candidate:
                    self._bundle_invalidation_tasks.pop(handle, None)

            task.add_done_callback(_release_completed)
        cancellation: asyncio.CancelledError | None = None
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            cancellation = error
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
        except Exception:  # noqa: BLE001 - invalidation is best-effort cleanup
            return
        if cancellation is not None:
            raise cancellation

    async def _choose_voice_bundle_export_path(self) -> Path | None:
        from tldw_chatbook.Third_Party.textual_fspicker import Filters
        from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileSave

        picker = EnhancedFileSave(
            title="Export portable voice bundle",
            default_filename="voice-profile.tldw-voice.zip",
            filters=Filters(
                (
                    "Voice Bundles",
                    lambda path: path.name.lower().endswith(".tldw-voice.zip"),
                ),
                ("All Files", lambda _path: True),
            ),
            context="tts_voice_bundle_export",
        )
        selected = await self.app.push_screen_wait(picker)
        return None if selected is None else Path(str(selected))

    async def _choose_voice_bundle_import_path(self) -> Path | None:
        from tldw_chatbook.Third_Party.textual_fspicker import Filters
        from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen

        picker = EnhancedFileOpen(
            title="Import portable voice bundle",
            filters=Filters(
                (
                    "Voice Bundles",
                    lambda path: path.name.lower().endswith(".tldw-voice.zip"),
                ),
                ("All Files", lambda _path: True),
            ),
            context="tts_voice_bundle_import",
        )
        selected = await self.app.push_screen_wait(picker)
        return None if selected is None else Path(str(selected))

    @staticmethod
    def _write_profile_export(target: Path, content: str) -> None:
        from tldw_chatbook.Utils.path_validation import validate_path

        expanded = target.expanduser()
        if not expanded.parent.exists():
            raise ValueError("missing_destination")
        validated = validate_path(
            expanded,
            base_directory=expanded.parent,
            redact_paths=True,
        )
        validated.write_text(content, encoding="utf-8")

    async def export_selected_profile(self) -> bool:
        """Export sanitized JSON by default; bundle export is explicit and gated."""

        loaded = self._selected_profile
        if loaded is None or not self._action_target_is_current(loaded):
            return False
        profile = loaded.profile
        operation: Literal["sanitized_export", "bundle_export"] = "sanitized_export"
        reference = profile.reference
        if reference is not None:
            bundle_disabled = not self._bundle_platform_supported
            recovery = PROFILE_BUNDLE_UNSUPPORTED_COPY if bundle_disabled else None
            if reference.recipe_requirement is None:
                bundle_disabled = True
                recovery = PROFILE_BUNDLE_MIGRATED_COPY
            sanitized_action, bundle_action = voice_bundle_export_actions(
                bundle_disabled=bundle_disabled,
                bundle_recovery=recovery,
            )
            choice = await self._push_owned_modal(
                TTSProfileExportChoiceModal(sanitized_action, bundle_action)
            )
            if (
                type(choice) is not VoiceBundleActionProjection
                or choice.disabled
                or choice.operation not in {"sanitized_export", "bundle_export"}
            ):
                return False
            operation = choice.operation
        if operation == "bundle_export":
            return await self._export_selected_voice_bundle(loaded)
        try:
            content = portable_profile_json(
                PortableTTSProfile(
                    profile_id=profile.profile_id,
                    draft=TTSProfileDraft(
                        display_name=profile.display_name,
                        provider_id=profile.provider_id,
                        model_id=profile.model_id,
                        voice_id=profile.voice_id,
                        response_format=profile.response_format,
                        speed=profile.speed,
                        options=profile.options,
                    ),
                ),
                reference_present=reference is not None,
            )
            target = await self._choose_profile_export_path()
            if target is None or not self._action_target_is_current(loaded):
                return False
            await asyncio.to_thread(self._write_profile_export, target, content)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - never render path or profile values
            self._set_status(PROFILE_ACTION_FAILED_COPY)
            return False
        self._set_status(PROFILE_EXPORT_COMPLETE_COPY)
        return True

    async def _export_selected_voice_bundle(self, loaded: LoadedTTSProfile) -> bool:
        if not self._action_target_is_current(loaded):
            return False
        service = await self._voice_bundle_service_for_action()
        if service is None or not self._action_target_is_current(loaded):
            return False
        acknowledged = await self._push_owned_modal(
            TTSVoiceBundleConsentModal(mode="export")
        )
        if acknowledged is not True or not self._action_target_is_current(loaded):
            return False
        target = await self._choose_voice_bundle_export_path()
        if target is None or not self._action_target_is_current(loaded):
            return False
        profile = loaded.profile
        try:
            await service.export(
                profile.profile_id,
                target,
                expected_generation=loaded.repository_generation,
                expected_revision=profile.revision,
                acknowledged=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - service errors stay redacted
            self._set_status(PROFILE_ACTION_FAILED_COPY)
            return False
        if not self._action_target_is_current(loaded):
            return False
        self._set_status(PROFILE_BUNDLE_EXPORT_COMPLETE_COPY)
        return True

    async def import_voice_bundle(self) -> bool:
        """Warn before path authority, then review and commit safe facts only."""

        self._portability_request_id += 1
        request_id = self._portability_request_id
        acknowledged = await self._push_owned_modal(
            TTSVoiceBundleConsentModal(mode="import")
        )
        if acknowledged is not True or not self._portability_request_is_current(
            request_id
        ):
            return False
        source = await self._choose_voice_bundle_import_path()
        if source is None or not self._portability_request_is_current(request_id):
            return False
        service = await self._voice_bundle_service_for_action()
        if service is None or not self._portability_request_is_current(request_id):
            return False
        try:
            review = await service.inspect(source)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - source/private values never rendered
            self._set_status(PROFILE_ACTION_FAILED_COPY)
            return False
        if not self._portability_request_is_current(request_id):
            await self._invalidate_bundle_handle(review.handle)
            return False
        owned_handle: TTSVoiceBundleHandle | None = review.handle
        self._active_bundle_handle = owned_handle
        try:
            while self._portability_request_is_current(request_id):
                decision = await self._push_owned_modal(
                    TTSVoiceBundleReviewModal(review)
                )
                if type(decision) is not VoiceBundleReviewDecision:
                    return False
                action = voice_bundle_review_action(
                    review,
                    decision.choice,
                    inactive_consent=decision.inactive_consent,
                )
                if action.disabled:
                    continue
                try:
                    result = await service.commit(
                        review.handle,
                        voice_bundle_import_choice(
                            action,
                            inactive_consent=decision.inactive_consent,
                        ),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - no exception-owned values in UI
                    self._set_status(PROFILE_ACTION_FAILED_COPY)
                    return False
                finally:
                    if self._active_bundle_handle is owned_handle:
                        self._active_bundle_handle = None
                    # commit() owns and consumes its input handle on every
                    # terminal result, including a replacement review.
                    owned_handle = None
                successor = result.review
                if successor is not None:
                    review = successor
                    owned_handle = successor.handle
                    self._active_bundle_handle = owned_handle
                if not self._portability_request_is_current(request_id):
                    return False
                if result.status == "stale_inspection" and successor is not None:
                    self._set_status(
                        "The bundle or profile store changed. Review the updated facts and confirm again."
                    )
                    continue
                self._set_status(PROFILE_BUNDLE_IMPORT_COMPLETE_COPY)
                self._queue_page_request(self._search, self._offset)
                return True
            return False
        finally:
            handle = owned_handle
            if handle is not None:
                if self._active_bundle_handle is handle:
                    self._active_bundle_handle = None
                await self._invalidate_bundle_handle(handle)

    def _portability_request_is_current(self, request_id: int) -> bool:
        return self._live and request_id == self._portability_request_id

    @on(Input.Changed, "#stts-profile-search")
    def _handle_search_changed(self, event: Input.Changed) -> None:
        event.stop()
        timer = self._search_timer
        if timer is not None:
            timer.stop()
        self._search_timer = self.set_timer(
            PROFILE_SEARCH_DEBOUNCE_SECONDS,
            self._submit_search,
        )

    def _submit_search(self) -> None:
        self._search_timer = None
        value = self.query_one("#stts-profile-search", Input).value
        normalized = value.strip()
        if normalized and not validate_text_input(
            normalized,
            max_length=PROFILE_SEARCH_MAX_CHARACTERS,
            allow_html=True,
        ):
            self._set_status(_PROFILE_SEARCH_VALIDATION_COPY)
            return
        self._queue_page_request(normalized or None, 0)

    @on(Button.Pressed, "#stts-profile-previous-btn")
    def _handle_previous(self, event: Button.Pressed) -> None:
        event.stop()
        self._queue_page_request(
            self._search,
            max(0, self._offset - PROFILE_PAGE_SIZE),
        )

    @on(Button.Pressed, "#stts-profile-next-btn")
    def _handle_next(self, event: Button.Pressed) -> None:
        event.stop()
        self._queue_page_request(
            self._search,
            self._offset + PROFILE_PAGE_SIZE,
        )

    @on(Button.Pressed, "#stts-profile-refresh-btn")
    def _handle_refresh(self, event: Button.Pressed) -> None:
        event.stop()
        self._queue_page_request(self._search, self._offset)

    @on(Button.Pressed, "#stts-profile-preview-btn")
    def _handle_preview(self, event: Button.Pressed) -> None:
        event.stop()
        loaded = self._selected_profile
        if loaded is None or not self._action_target_is_current(loaded):
            return
        availability = self._row_availability.get(str(loaded.profile.profile_id))
        if availability is None:
            availability = TTSProfileAvailability(
                profile_id=loaded.profile.profile_id,
                state="unverified",
                recovery_action="refresh",
            )
        if availability.state == "unavailable":
            self._show_selected_detail()
            return
        service = self._service
        if service is None:
            self._set_status(PROFILE_STORE_UNAVAILABLE_COPY)
            return
        try:
            preset = service.preview_preset(loaded, availability)
        except Exception as error:  # noqa: BLE001 - map to bounded UI copy
            self._set_status(profile_action_error_copy(error))
            return
        if type(preset) is not TTSPlaygroundSelectionPreset:
            self._set_status(PROFILE_ACTION_FAILED_COPY)
            return
        try:
            registration = _remember_profile_test_context(
                self._service,
                loaded,
                preset,
            )
        except Exception:  # noqa: BLE001 - never expose profile-owned values
            self._set_status(PROFILE_ACTION_FAILED_COPY)
            return
        self.post_message(
            ProfilePreviewRequested(
                registration.preset,
                self.navigation_continuity(),
                registration.context_token,
            )
        )

    @on(
        Button.Pressed,
        "#stts-profile-dependency-primary-btn, #stts-profile-dependency-advisory-btn",
    )
    def _handle_dependency_recovery(self, event: Button.Pressed) -> None:
        """Execute only the operation projected for the selected fresh row."""

        event.stop()
        loaded = self._selected_profile
        if loaded is None or not self._action_target_is_current(loaded):
            return
        availability = self._row_availability.get(str(loaded.profile.profile_id))
        if availability is None:
            return
        actions = dependency_recovery_actions(availability.dependency)
        role = (
            "advisory"
            if event.button.id == "stts-profile-dependency-advisory-btn"
            else "blocker"
        )
        action = next((item for item in actions if item.role == role), None)
        if action is None:
            return
        operation = action.operation
        if operation == "open_audio_cpp_settings":
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
            return
        # Applying saved settings and migrating a null-provenance reference
        # both require the exact selected voice in Speech Lab. Reuse the
        # existing preview seam; neither action changes assignments/defaults.
        self._handle_preview(event)

    @on(Button.Pressed, "#stts-profile-edit-btn")
    def _handle_edit(self, event: Button.Pressed) -> None:
        event.stop()
        self.run_worker(
            self.edit_selected_profile(),
            name="edit_voice_profile",
            group="voice_profile_action",
            exclusive=True,
            exit_on_error=False,
        )

    @on(Button.Pressed, "#stts-profile-duplicate-btn")
    def _handle_duplicate(self, event: Button.Pressed) -> None:
        event.stop()
        self.run_worker(
            self.duplicate_selected_profile(),
            name="duplicate_voice_profile",
            group="voice_profile_action",
            exclusive=True,
            exit_on_error=False,
        )

    @on(Button.Pressed, "#stts-profile-export-btn")
    def _handle_export(self, event: Button.Pressed) -> None:
        event.stop()
        self.run_worker(
            self.export_selected_profile(),
            name="export_voice_profile",
            group="voice_profile_action",
            exclusive=True,
            exit_on_error=False,
        )

    @on(Button.Pressed, "#stts-profile-import-btn")
    def _handle_import(self, event: Button.Pressed) -> None:
        event.stop()
        self.run_worker(
            self.import_voice_bundle(),
            name="import_voice_bundle",
            group="voice_profile_action",
            exclusive=True,
            exit_on_error=False,
        )

    @on(Button.Pressed, "#stts-profile-delete-btn")
    def _handle_delete(self, event: Button.Pressed) -> None:
        event.stop()
        self.run_worker(
            self.delete_selected_profile(),
            name="delete_voice_profile",
            group="voice_profile_action",
            exclusive=True,
            exit_on_error=False,
        )
