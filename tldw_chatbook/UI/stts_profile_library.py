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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol, cast
from uuid import UUID

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Button, DataTable, Input, Label, Static

from tldw_chatbook.TTS import (
    LoadedTTSProfile,
    ProfileRepositoryError,
    ProfileServiceError,
    ProfileValidationError,
    STTSGeneratedAudio,
    TTSProfileAvailability,
    TTSProfileAvailabilitySnapshot,
    TTSProfileDraft,
    TTSProfilePageSnapshot,
)

PROFILE_PAGE_SIZE = 50
PROFILE_SEARCH_DEBOUNCE_SECONDS = 0.25
PROFILE_STORE_UNAVAILABLE_COPY = (
    "Profile storage unavailable. Ordinary Playground, settings, audiobook, "
    "dictation, and legacy speech remain available. Choose Refresh to retry "
    "profile storage."
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
    "No voice profiles match. Change the search or save a successful native "
    "audio.cpp result."
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
_PROFILE_UNAVAILABLE_COPY = (
    "The exact profile selection is unavailable. Refresh current capabilities "
    "or edit the persisted model and voice."
)
_PROFILE_UNVERIFIED_COPY = (
    "The exact profile selection could not be verified. Refresh and retry "
    "without changing persisted values."
)
_PROFILE_ACTION_WORKING_COPY = "Checking the exact profile version…"


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


ProfileServiceLoader = Callable[[], Awaitable[_ProfileService | None]]


@dataclass(frozen=True, slots=True)
class _PageRequest:
    mount_token: int
    request_id: int
    search: str | None
    offset: int


class ProfilePreviewRequested(Message):
    """Request a one-shot exact Playground preview without synthesizing."""

    def __init__(
        self,
        loaded: LoadedTTSProfile,
        availability: TTSProfileAvailability,
    ) -> None:
        super().__init__()
        self.loaded = loaded
        self.availability = availability


def _assignment_copy(count: int) -> str:
    noun = "assignment" if count == 1 else "assignments"
    return f"{count} {noun}"


def _error_copy(error: BaseException) -> str:
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
        padding: 1 2;
    }

    #stts-profile-editor-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #stts-profile-editor-scope,
    #stts-profile-editor-fixed,
    #stts-profile-editor-error {
        height: auto;
        color: $text-muted;
        margin-bottom: 1;
    }

    #stts-profile-editor-error {
        color: $warning;
    }

    .stts-profile-editor-field {
        height: 3;
        margin-bottom: 1;
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
                    placeholder="Server default",
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
        assignment_count: int,
    ) -> None:
        super().__init__()
        self.assignment_count = assignment_count

    def compose(self) -> ComposeResult:
        protected = self.assignment_count > 0
        with Vertical(id="stts-profile-delete-dialog"):
            yield Label("Delete voice profile", id="stts-profile-delete-title")
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


_OwnedProfileModal = TTSProfileEditorModal | TTSProfileDeleteModal
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
        max-height: 5;
        background: $panel;
        border: round $surface-lighten-1;
        color: $text;
        padding: 0 1;
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
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._service_loader = service_loader
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
        self._retained_editor_draft: _RetainedEditorDraft | None = None
        self._active_modal: _OwnedProfileModal | None = None
        self._search: str | None = None
        self._offset = 0
        self._total = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="stts-profile-header"):
            yield Label("Voice profiles", id="stts-profile-title")
            yield Static(
                "Manage exact native audio.cpp model and voice selections.",
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
        yield Static(_PROFILE_LOADING_COPY, id="stts-profile-status")
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
        self._queue_page_request(None, 0)

    async def on_unmount(self) -> None:
        self._live = False
        self._mount_token += 1
        self._pending_page_request = None
        self._retained_editor_draft = None
        modal = self._active_modal
        if modal is not None:
            self._dismiss_owned_modal(modal)
        self._active_modal = None
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
        self._request_id += 1
        self._search = search
        self._offset = max(0, offset)
        request = _PageRequest(
            mount_token=self._mount_token,
            request_id=self._request_id,
            search=self._search,
            offset=self._offset,
        )
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
                if self._rendered_page_is_current(request, page):
                    self._set_status(_PROFILE_AVAILABILITY_FAILED_COPY)
                return
            if self._availability_can_publish(request, page, availability):
                self._publish_availability(page, availability)
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
        self._selected_profile = None
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
                Text(
                    "Checking" if availability is None else availability.state.title()
                ),
                Text(str(profile.revision)),
                key=key,
            )

        self._sync_paging()
        if page.profiles:
            self._set_status(
                f"{len(page.profiles)} voice profiles loaded. "
                "Checking current availability…"
            )
        else:
            self._set_status(_PROFILE_EMPTY_COPY)

    def _publish_availability(
        self,
        page: TTSProfilePageSnapshot,
        snapshot: TTSProfileAvailabilitySnapshot,
    ) -> None:
        table = self.query_one("#stts-profile-table", DataTable)
        self._row_availability.clear()
        for item in snapshot.profiles:
            key = str(item.profile_id)
            self._row_availability[key] = item
            table.update_cell(
                key,
                self._availability_column,
                Text(item.state.title()),
            )
        self._availability_configuration_revision = snapshot.configuration_revision
        self._availability_catalog_revision = snapshot.catalog_revision
        if self._selected_profile is not None:
            self._show_selected_detail()
            return
        self._set_status(
            f"{len(page.profiles)} voice profiles loaded. "
            "Availability is current for this page."
        )

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
        self.query_one("#stts-profile-status", Static).update(Text(copy))

    def _sync_selected_actions(self) -> None:
        disabled = (
            self._selected_profile is None or not self._rendered_request_is_current()
        )
        for selector in (
            "#stts-profile-preview-btn",
            "#stts-profile-edit-btn",
            "#stts-profile-duplicate-btn",
            "#stts-profile-delete-btn",
        ):
            self.query_one(selector, Button).disabled = disabled

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
        state = "Checking" if availability is None else availability.state.title()
        voice = profile.voice_id if profile.voice_id is not None else "Server default"
        if availability is not None and availability.state == "unavailable":
            status_line = "Unavailable — Refresh, then Edit."
        elif availability is not None and availability.state == "unverified":
            status_line = "Unverified — Refresh and retry."
        else:
            status_line = f"{state}."
        self._set_status(
            f"{status_line}\n"
            f"Selected: {profile.display_name}\n"
            f"{profile.provider_id} / {profile.model_id} / {voice}"
        )

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
        self._set_status(_PROFILE_ACTION_WORKING_COPY)
        try:
            count = await service.assignment_count(loaded)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - map to bounded UI copy
            self._set_status(_error_copy(error))
            return None
        if type(count) is not int or count < 0:
            self._set_status(PROFILE_ACTION_FAILED_COPY)
            return None
        return count

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
            self._set_status(_error_copy(error))
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
            self._set_status(_error_copy(error))
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
            self._set_status(_error_copy(error))
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
                TTSProfileDeleteModal(assignment_count=count)
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
            self._set_status(_error_copy(error))
            return False
        if self._live:
            self._queue_page_request(self._search, self._offset)
        return True

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
        self._queue_page_request(value.strip() or None, 0)

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
        self.post_message(ProfilePreviewRequested(loaded, availability))

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
