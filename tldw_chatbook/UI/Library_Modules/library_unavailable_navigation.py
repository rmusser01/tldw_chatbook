"""Library-owned unavailable inspection and complete archive projection."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from loguru import logger

from ...Constants import (
    LIBRARY_NAV_CONTEXT_CHARACTER_BROWSE,
    LIBRARY_NAV_CONTEXT_CHARACTER_INSPECTION,
)
from ...Library.library_conversation_reader_state import LIBRARY_CONVERSATION_PAGE_SIZE

if TYPE_CHECKING:
    from ...Character_Chat.character_conversation_navigation import (
        UnresolvedConversationKey,
    )
    from ..Navigation.character_conversation_navigation import (
        LibraryUnavailableConversationInspection,
        LibraryUnavailableConversationsBrowse,
    )


@dataclasses.dataclass(frozen=True)
class _LibraryCharacterNavigationAdmission:
    """Typed Character route plus the exact Data Profile that admitted it."""

    route: (
        LibraryUnavailableConversationInspection | LibraryUnavailableConversationsBrowse
    )
    database: Any
    generation: int


@dataclasses.dataclass(frozen=True)
class _LibraryUnavailableBrowseScope:
    """Authority retained for the complete unavailable-only browse session."""

    database: Any
    authority: str
    selected: UnresolvedConversationKey
    navigation_generation: int


def build_canvas_state(self):
    """Build the conversations canvas display state from local records."""
    from ...Library.library_conversations_state import build_library_conversations_state

    state = build_library_conversations_state(
        self._conversation_records(),
        query=self._library_conversation_query,
        selected_id=self._selected_conversation_id,
        select_mode=self._library_conversations_select_mode,
        selected_ids=self._library_conversations_row_selection.ids,
        page=self._library_conversation_page,
        requested_page=self._library_conversation_requested_page,
        page_size=self._library_conversation_page_size,
        total_count=self._library_conversation_total,
        total_known=self._library_conversation_total_known,
        has_more=self._library_conversation_has_more,
        freshness=self._library_conversation_freshness,
        stale_copy=self._library_conversation_stale_copy,
        loading=self._library_conversation_loading,
        error_copy=self._library_conversation_error,
        selection_notice=self._library_conversation_selection_notice,
    )
    state = dataclasses.replace(
        state,
        query=self._library_conversation_requested_query,
        title=(
            f"Unavailable character conversations ({self._library_conversation_total})"
            if self._library_conversation_projection == "unavailable_character"
            else "Conversations"
        ),
        status_copy=(state.status_copy or self._library_conversation_selection_notice),
    )
    retained_reader_id = self._library_conversation_reader_state.selected_id
    if (
        retained_reader_id
        and not self._library_conversation_reader_state.unavailable
        and all(row.conversation_id != retained_reader_id for row in state.rows)
    ):
        state = dataclasses.replace(
            state,
            selected_id="",
            preview_lines=(),
            rows=tuple(dataclasses.replace(row, selected=False) for row in state.rows),
        )
    if self._library_conversation_deleted_selection_id:
        state = dataclasses.replace(
            state,
            selected_id="",
            preview_lines=(),
            rows=tuple(dataclasses.replace(row, selected=False) for row in state.rows),
        )
    if self._library_conversations_select_mode:
        self._library_conversations_row_selection.reconcile(
            r.conversation_id for r in state.rows
        )
    return state


def _library_character_navigation_admission(
    self,
    context: Mapping[str, Any],
    *,
    generation: int,
) -> _LibraryCharacterNavigationAdmission | None:
    """Parse one strict typed route and bind it to the active DB handle."""
    from ..Navigation.character_conversation_navigation import (
        deserialize_library_unavailable_browse,
        deserialize_library_unavailable_inspection,
    )

    route: (
        LibraryUnavailableConversationInspection
        | LibraryUnavailableConversationsBrowse
        | None
    ) = None
    try:
        if set(context) == {LIBRARY_NAV_CONTEXT_CHARACTER_INSPECTION}:
            payload = context.get(LIBRARY_NAV_CONTEXT_CHARACTER_INSPECTION)
            if isinstance(payload, Mapping):
                route = deserialize_library_unavailable_inspection(payload)
        elif set(context) == {LIBRARY_NAV_CONTEXT_CHARACTER_BROWSE}:
            payload = context.get(LIBRARY_NAV_CONTEXT_CHARACTER_BROWSE)
            if isinstance(payload, Mapping):
                route = deserialize_library_unavailable_browse(payload)
    except (TypeError, ValueError):
        logger.warning("Rejected invalid Library unavailable navigation")
        return None
    if route is None:
        return None
    database = getattr(self.app_instance, "chachanotes_db", None)
    if database is None:
        return None
    return _LibraryCharacterNavigationAdmission(route, database, generation)


def compose_character_return(self):
    """Expose the return only while the validated Character route owns this view."""
    from ...Library.library_shell_state import LIBRARY_ROW_BROWSE_CONVERSATIONS

    admission = self._navigation_controller.character_route
    if (
        admission is not None
        and _library_character_admission_is_current(self, admission)
        and self._library_selected_row_id == LIBRARY_ROW_BROWSE_CONVERSATIONS
    ):
        from ...Widgets.Library.library_character_return import LibraryCharacterReturn

        yield LibraryCharacterReturn(lambda: return_from_character(self, admission))


def return_from_character(self, admission) -> None:
    from ...Constants import CHARACTER_NAV_CONTEXT_RETURN_FOCUS
    from ..Navigation.main_navigation import NavigateToScreen

    if not _library_character_admission_is_current(self, admission):
        return
    target = admission.route.return_target
    self.post_message(
        NavigateToScreen(
            target.screen_id, {CHARACTER_NAV_CONTEXT_RETURN_FOCUS: target.focus_id}
        )
    )


def clear_character_return(self) -> None:
    from textual.screen import ModalScreen

    if isinstance(self.app.screen, ModalScreen):
        return
    self._navigation_controller.character_route = None
    self._navigation_controller.character_candidate = None
    self._library_navigation_context_generation += 1
    for control in self.query("#library-character-return"):
        control.display = False


def _library_character_admission_is_current(
    self,
    admission: _LibraryCharacterNavigationAdmission | None,
) -> bool:
    if admission is None:
        return True
    database = getattr(self.app_instance, "chachanotes_db", None)
    if database is not admission.database:
        return False
    navigation = self._navigation_controller
    return navigation.character_route is admission or (
        navigation.character_candidate is admission
        and admission.generation == self._library_navigation_context_generation
    )


def _discard_library_character_admission(
    self,
    admission: _LibraryCharacterNavigationAdmission | None,
) -> None:
    if (
        admission is not None
        and self._pending_library_character_navigation is admission
    ):
        self._pending_library_character_navigation = None


def _library_unavailable_browse_scope_is_current(
    self,
    scope: _LibraryUnavailableBrowseScope | None,
) -> bool:
    """Return whether an unavailable browse still owns its exact profile."""

    return bool(
        scope is not None
        and self._library_unavailable_browse_scope is scope
        and getattr(self.app_instance, "chachanotes_db", None) is scope.database
        and scope.navigation_generation
        == getattr(self._navigation_controller.character_route, "generation", None)
        and _library_character_authority_is_current(self, scope.authority)
    )


def _discard_library_unavailable_browse_scope(
    self,
    scope: _LibraryUnavailableBrowseScope | None,
    *,
    profile_changed: bool = False,
) -> None:
    """Discard one owned browse and fail closed on Data Profile churn."""

    if scope is None or self._library_unavailable_browse_scope is not scope:
        return
    self._library_unavailable_browse_scope = None
    if not profile_changed:
        return
    self._conversations_state.page_records = ()
    self._conversations_state.total = 0
    self._conversations_state.total_known = False
    self._conversations_state.has_more = False
    self._conversations_state.page_loaded = False
    self._conversations_state.loading = False
    self._conversations_state.error = "Data Profile changed · Retry"
    self._conversations_state.projection = ""
    self._selected_conversation_id = ""
    if self.is_mounted:
        self._sync_library_conversation_canvas()


def _library_character_authority_is_current(self, authority: str) -> bool:
    """UI-side fence over an off-loop validated snapshot; never performs SQL."""
    from ..Navigation.character_conversation_navigation import (
        LibraryUnavailableConversationInspection,
    )

    admission = self._navigation_controller.character_route
    if admission is None or not _library_character_admission_is_current(
        self, admission
    ):
        return False
    route = admission.route
    key = (
        route.unresolved
        if isinstance(route, LibraryUnavailableConversationInspection)
        else route.selected
    )
    return key.data_authority_id == authority


async def _validate_library_character_admission(self, admission) -> bool:
    """Read transactional authority off-loop, then retain only a current snapshot."""
    import asyncio

    from ..Navigation.character_conversation_navigation import (
        LibraryUnavailableConversationInspection,
    )

    if admission is None:
        return True
    if _library_character_admission_is_current(self, admission):
        return True

    def scope_is_current():
        return (
            getattr(self.app_instance, "chachanotes_db", None) is admission.database
            and admission.generation == self._library_navigation_context_generation
        )

    if not scope_is_current():
        return False
    initial_loading = self._navigation_controller.character_route is None
    if initial_loading:
        self._conversations_state.loading = True
    if self.is_mounted:
        self.notify("Loading character conversations…", severity="information")
    route = admission.route
    key = (
        route.unresolved
        if isinstance(route, LibraryUnavailableConversationInspection)
        else route.selected
    )

    try:
        authority = await asyncio.to_thread(admission.database.get_local_authority_id)
    except Exception as error:  # noqa: BLE001 - navigation rejects unavailable storage
        logger.warning(
            "Could not validate Library character navigation authority "
            "exception_type={}",
            type(error).__name__,
        )
        authority = None
    if not scope_is_current():
        return False
    if initial_loading:
        self._conversations_state.loading = False
    if authority != key.data_authority_id:
        _discard_library_character_admission(self, admission)
        return False
    self._navigation_controller.character_candidate = admission
    return True


async def _open_pending_library_character_navigation(self) -> None:
    """Consume one typed Character route without dropping intent or authority."""
    from ..Navigation.character_conversation_navigation import (
        LibraryUnavailableConversationInspection,
    )

    admission = self._pending_library_character_navigation
    if admission is None:
        return
    if not await _validate_library_character_admission(self, admission):
        _discard_library_character_admission(self, admission)
        return
    if not _library_character_admission_is_current(self, admission):
        _discard_library_character_admission(self, admission)
        return
    route = admission.route
    self._apply_navigation_context_state(
        {}, recompose=False, character_admission=admission
    )
    if isinstance(route, LibraryUnavailableConversationInspection):
        if self.is_mounted:
            await self.recompose()
        if not _library_character_admission_is_current(self, admission):
            _discard_library_character_admission(self, admission)
            return
        await self._open_library_item_by_id(
            "conversations",
            route.unresolved.conversation_id,
            entry_origin=True,
            required_database=admission.database,
            required_authority=route.unresolved.data_authority_id,
        )
        if not _library_character_admission_is_current(self, admission):
            _discard_library_character_admission(self, admission)
            self._pending_library_source_open = None
            return
        self._pending_library_source_open = None
        _discard_library_character_admission(self, admission)
        return

    scope = self._library_unavailable_browse_scope
    try:
        if not _library_unavailable_browse_scope_is_current(self, scope):
            _discard_library_unavailable_browse_scope(
                self,
                scope,
                profile_changed=True,
            )
            return
        normalized_query, generation = self._prepare_library_conversation_page_request(
            "", page=1
        )
        await _load_library_unavailable_conversation_page(
            self,
            scope,
            1,
            normalized_query,
            generation,
        )
    finally:
        _discard_library_character_admission(self, admission)


def _start_library_unavailable_conversation_page_request(
    self,
    page: int,
    query: str,
    *,
    refocus_filter: bool = False,
    focus_after_apply: str = "",
) -> None:
    """Start a page request owned by the retained unavailable projection."""

    scope = self._library_unavailable_browse_scope
    if not _library_unavailable_browse_scope_is_current(self, scope):
        _discard_library_unavailable_browse_scope(
            self,
            scope,
            profile_changed=True,
        )
        return
    requested_page = self._normalize_library_conversation_page(page)
    normalized_query, generation = self._prepare_library_conversation_page_request(
        query,
        page=requested_page,
        refocus_filter=refocus_filter,
        focus_after_apply=focus_after_apply,
    )
    self.run_worker(
        _load_library_unavailable_conversation_page(
            self,
            scope,
            requested_page,
            normalized_query,
            generation,
        ),
        exclusive=True,
        group="library_conversation_page",
    )


async def _load_library_unavailable_conversation_page(
    self,
    scope: _LibraryUnavailableBrowseScope,
    page: int,
    query: str,
    generation: int,
) -> None:
    """Stage and fence an unavailable-only page through final composition."""

    from ...Character_Chat.character_conversation_navigation import (
        CharacterConversationNavigationService,
    )

    requested_page = self._normalize_library_conversation_page(page)
    normalized_query = self._safe_text(query, max_length=200)
    requested_offset = (requested_page - 1) * LIBRARY_CONVERSATION_PAGE_SIZE

    def request_is_current() -> bool:
        return bool(
            generation == self._conversations_state.request_generation
            and _library_unavailable_browse_scope_is_current(self, scope)
        )

    if not request_is_current():
        _discard_library_unavailable_browse_scope(
            self,
            scope,
            profile_changed=True,
        )
        return
    unavailable_page = CharacterConversationNavigationService(
        scope.database
    ).unavailable_page
    call_kwargs: dict[str, Any] = {
        "offset": requested_offset,
        "limit": LIBRARY_CONVERSATION_PAGE_SIZE,
    }
    if normalized_query:
        call_kwargs["query"] = normalized_query
    try:
        result = await self._run_library_service_call(
            unavailable_page,
            **call_kwargs,
        )
    except Exception:  # noqa: BLE001 - this projection remains retryable
        if request_is_current():
            self._fail_library_conversation_request(
                requested_page,
                normalized_query,
                generation,
                copy="Could not load unavailable character conversations.",
            )
        else:
            _discard_library_unavailable_browse_scope(
                self,
                scope,
                profile_changed=True,
            )
        return
    if not request_is_current():
        _discard_library_unavailable_browse_scope(
            self,
            scope,
            profile_changed=True,
        )
        return

    try:
        total = result.total
        rows = tuple(result.rows)
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise ValueError("invalid unavailable total")
        if len(rows) > LIBRARY_CONVERSATION_PAGE_SIZE:
            raise ValueError("unavailable page exceeded its bound")
        records = tuple(
            {
                "conversation_id": row.unresolved.conversation_id,
                "title": row.title,
                "last_modified": row.last_modified,
                "unavailable_reason": row.unavailable_reason.value,
            }
            for row in rows
            if row.unresolved is not None
            and row.target is None
            and row.unresolved.data_authority_id == scope.authority
        )
        if len(records) != len(rows):
            raise ValueError("unavailable page mixed authorities or resolved rows")
    except (AttributeError, TypeError, ValueError):
        if request_is_current():
            self._fail_library_conversation_request(
                requested_page,
                normalized_query,
                generation,
                copy="Could not load unavailable character conversations.",
            )
        return

    # Keep staged old-profile data out of retained state and the DOM.  The
    # awaited composition is the last UI scheduling boundary; only a scope
    # still exact on both sides may publish the page synchronously.
    if self.is_mounted:
        await self.recompose()
    if not request_is_current():
        _discard_library_unavailable_browse_scope(
            self,
            scope,
            profile_changed=True,
        )
        return

    self._conversations_state.page_records = records
    self._conversations_state.page = requested_page
    self._conversations_state.total = total
    self._conversations_state.total_known = True
    self._conversations_state.has_more = requested_offset + len(records) < total
    self._conversations_state.page_loaded = True
    self._conversations_state.query = normalized_query
    self._conversations_state.requested_page = requested_page
    self._conversations_state.requested_query = normalized_query
    self._conversations_state.freshness = "fresh"
    self._conversations_state.loading = False
    self._conversations_state.error = ""
    self._conversations_state.projection = "unavailable_character"
    self._selected_conversation_id = scope.selected.conversation_id
    self._sync_library_conversation_canvas(
        then=self._conversations_controller._finish_library_conversation_page_apply
    )


def _apply_navigation_context_state(
    self,
    context: Mapping[str, Any],
    *,
    recompose: bool = True,
    character_admission: _LibraryCharacterNavigationAdmission | None = None,
) -> None:
    """Apply validated navigation context to canvas state and recompose.

    Split from ``apply_navigation_context`` so its mounted path can admit
    every pending save first (see
    ``_apply_navigation_context_after_flush``) while the pre-mount and
    clean-editor paths apply directly.
    """
    from ...Constants import (
        LIBRARY_MODE_CONVERSATIONS,
        LIBRARY_NAV_CONTEXT_CONVERSATION_ID,
        LIBRARY_NAV_CONTEXT_INGEST,
        LIBRARY_NAV_CONTEXT_MODE,
        LIBRARY_NAV_CONTEXT_NOTE_ID,
        LIBRARY_NAV_CONTEXT_NOTES_CREATE,
        LIBRARY_NAV_CONTEXT_OPEN_SOURCE_ID,
        LIBRARY_NAV_CONTEXT_OPEN_SOURCE_TYPE,
    )
    from ...Library.library_shell_state import (
        LIBRARY_ROW_BROWSE_COLLECTIONS,
        LIBRARY_ROW_BROWSE_CONVERSATIONS,
        LIBRARY_ROW_BROWSE_MEDIA,
        LIBRARY_ROW_BROWSE_NOTES,
        LIBRARY_ROW_BROWSE_PROMPTS,
        LIBRARY_ROW_CREATE_NOTE,
        LIBRARY_ROW_INGEST_MEDIA,
    )
    from .screen_constants import (
        LIBRARY_NAV_MODE_TO_ROW_ID,
        LIBRARY_NOTES_SOURCE_DATABASE,
    )

    if self._prompts_state.mutation_in_flight:
        return
    if character_admission is not None:
        if not _library_character_admission_is_current(self, character_admission):
            if not self.is_mounted:
                self._pending_library_character_navigation = character_admission
            return
        from ..Navigation.character_conversation_navigation import (
            LibraryUnavailableConversationsBrowse,
        )

        self._navigation_controller.character_route = character_admission
        self._navigation_controller.character_candidate = None
        self._pending_library_character_navigation = character_admission
        route = character_admission.route
        if isinstance(route, LibraryUnavailableConversationsBrowse):
            self._library_unavailable_browse_scope = _LibraryUnavailableBrowseScope(
                database=character_admission.database,
                authority=route.selected.data_authority_id,
                selected=route.selected,
                navigation_generation=character_admission.generation,
            )
        else:
            self._library_unavailable_browse_scope = None
        self._pending_library_source_open = None
        self._conversations_state.projection = (
            "unavailable_character"
            if isinstance(
                route,
                LibraryUnavailableConversationsBrowse,
            )
            else ""
        )
        self._set_library_destination_with_conversation_fence(
            LIBRARY_ROW_BROWSE_CONVERSATIONS
        )
        self._invalidate_library_workspace_depth_state()
        if self.is_mounted and recompose:
            self.refresh(recompose=True)
        return
    self._navigation_controller.character_route = None
    self._navigation_controller.character_candidate = None
    self._library_unavailable_browse_scope = None
    self._supersede_library_notes_navigation()
    raw_open_source_type = context.get(LIBRARY_NAV_CONTEXT_OPEN_SOURCE_TYPE)
    raw_open_source_id = context.get(LIBRARY_NAV_CONTEXT_OPEN_SOURCE_ID)
    open_source_type = ""
    open_source_id = ""
    should_open_pending_source = False
    if type(raw_open_source_type) is str and type(raw_open_source_id) is str:
        validated_source_type = self._safe_text(
            raw_open_source_type,
            max_length=64,
        )
        validated_source_id = self._safe_text(
            raw_open_source_id,
            max_length=500,
        )
        if (
            validated_source_type == raw_open_source_type
            and validated_source_id == raw_open_source_id
            and validated_source_type in ("media", "notes", "conversations", "prompt")
            and validated_source_id
        ):
            open_source_type = validated_source_type
            open_source_id = validated_source_id
            self._pending_library_source_open = (
                open_source_type,
                open_source_id,
            )
            should_open_pending_source = True
    requested_mode = self._safe_text(
        context.get(LIBRARY_NAV_CONTEXT_MODE),
        max_length=64,
    )
    conversation_id = self._safe_text(
        context.get(LIBRARY_NAV_CONTEXT_CONVERSATION_ID),
        max_length=200,
    )
    note_id = self._safe_text(
        context.get(LIBRARY_NAV_CONTEXT_NOTE_ID),
        max_length=200,
    )
    notes_create = bool(context.get(LIBRARY_NAV_CONTEXT_NOTES_CREATE))
    ingest_media = bool(context.get(LIBRARY_NAV_CONTEXT_INGEST))
    target_mode = requested_mode if requested_mode in LIBRARY_NAV_MODE_TO_ROW_ID else ""
    if conversation_id and not target_mode:
        target_mode = LIBRARY_MODE_CONVERSATIONS
    if target_mode:
        selected_row_id = LIBRARY_NAV_MODE_TO_ROW_ID.get(target_mode)
        if selected_row_id:
            self._set_library_destination_with_conversation_fence(selected_row_id)
        self._invalidate_library_workspace_depth_state()
    if conversation_id:
        self._selected_conversation_id = conversation_id
        self._set_library_destination_with_conversation_fence(
            LIBRARY_ROW_BROWSE_CONVERSATIONS
        )
        if not should_open_pending_source:
            # Persona still emits the legacy conversation_id context.
            # A paged snapshot may not contain that id, so resolve it
            # through the same point-lookup opener used by Search/RAG.
            self._pending_library_source_open = (
                "conversations",
                conversation_id,
            )
            should_open_pending_source = True
    if requested_mode == "notes" and not note_id:
        # "notes" is a canvas row, not a nav-context table entry (see
        # target_mode above), so it needs its own selection here --
        # mirrors handle_library_notes_row's list-view entry state.
        self._set_library_destination_with_conversation_fence(LIBRARY_ROW_BROWSE_NOTES)
    if notes_create:
        # Mirrors _select_library_rail_row(LIBRARY_ROW_CREATE_NOTE) --
        # the create-note rail row's own target_id. The rail row's
        # flush of a dirty editor is handled upstream by
        # apply_navigation_context's mounted dirty-editor branch; here we
        # only apply the selection the recompose reads. Reset the note
        # editor state FIRST (a mounted screen re-entered via this
        # deep link can still hold a previously opened note's
        # id/detail/version) then re-assert the create-note target state
        # AFTER, since the reset flips _library_notes_view back to
        # "list" -- same reset-then-set ordering as
        # _open_library_item_by_id's notes branch.
        self._set_library_notes_source(LIBRARY_NOTES_SOURCE_DATABASE)
        self._dispatch_database_note_identity_cleared()
        self._reset_library_note_editor_state()
        self._set_library_destination_with_conversation_fence(LIBRARY_ROW_CREATE_NOTE)
    if ingest_media:
        # Home's ingest-jobs "Open details" control re-points here
        # (L3b Task 6): running/queued/failed Library ingest jobs
        # mirror into Home's Running and Needs Attention sections, and
        # this deep link is their one-hop route back to the in-canvas
        # ingest queue. Mirrors
        # _select_library_rail_row(LIBRARY_ROW_INGEST_MEDIA) -- unlike
        # collections/note_id above, the ingest canvas reads the job
        # registry directly on recompose, so no async data fetch (and
        # therefore no on_mount deferral) is needed even pre-mount.
        self._set_library_destination_with_conversation_fence(LIBRARY_ROW_INGEST_MEDIA)
        # Mirrors _select_library_rail_row's reset: a cached LibraryScreen
        # re-entered via this deep link (e.g. from Home's ingest-jobs
        # "Open details" control) must never show a stale half-filled
        # form left over from a previous Ingest visit.
        self._reset_library_ingest_transient_state()
    if note_id:
        # Forward-compat entry point: the retired Notes tab's chat-sidebar
        # deep link carried a note id, and this rebuilds the editor for it.
        # No caller in the tree emits a note_id context today (the surviving
        # open_notes_workspace route carries none, landing on the list), so
        # this is exercised only by tests until such a producer is wired --
        # not orphaned wiring.
        self._set_library_notes_source(LIBRARY_NOTES_SOURCE_DATABASE)
        self._set_library_destination_with_conversation_fence(LIBRARY_ROW_BROWSE_NOTES)
        if self.is_mounted:
            self._begin_library_note_load(note_id)
        else:
            self._library_note_session.close_session()
            self._selected_note_id = note_id
            self._library_notes_view = "editor"
            self._library_note_load_state = "loading"
            self._library_note_load_message = ""
            self._library_note_autosave_state = "idle"
            self._library_note_confirming_delete = False
            self._library_note_preview = False
            self._library_note_editor_armed = False
        # A deep link never owns the current blank-note GC identity.
        self._library_note_pending_blank_gc_id = None
        self._library_note_session_blank_id = None
        self._library_note_title_user_edited = False
    if open_source_type:
        self._set_library_destination_with_conversation_fence(
            {
                "media": LIBRARY_ROW_BROWSE_MEDIA,
                "notes": LIBRARY_ROW_BROWSE_NOTES,
                "conversations": LIBRARY_ROW_BROWSE_CONVERSATIONS,
                "prompt": LIBRARY_ROW_BROWSE_PROMPTS,
            }[open_source_type]
        )
        if open_source_type == "media":
            self._selected_media_id = open_source_id
            self._library_media_view = "list"
        elif open_source_type == "notes":
            self._selected_note_id = open_source_id
            self._set_library_notes_source(LIBRARY_NOTES_SOURCE_DATABASE)
            self._library_notes_view = "list"
        elif open_source_type == "conversations":
            self._selected_conversation_id = open_source_id
        elif open_source_type == "prompt":
            try:
                self._prompts_state.selected_prompt_id = int(open_source_id)
            except ValueError:
                self._prompts_state.selected_prompt_id = None
            self._prompts_state.view = "list"
    if should_open_pending_source and self.is_mounted:
        self.run_worker(
            self._open_pending_library_source(),
            exclusive=True,
            group="library_nav_open_source",
        )
    # F-012: a deep link can change the active canvas (e.g. mode="search")
    # without a rail-row press -- the footer's `u` hint must follow the
    # canvas, not just the rail switch, or the key works unadvertised.
    self._register_footer_shortcuts()
    if self._library_notes_workflow_active():
        self._library_notes_stage = "notes"
        self._library_notes_explicit_stage_intent = not self.is_mounted
    else:
        self._library_notes_explicit_stage_intent = False
    if self.is_mounted:
        if self._library_selected_row_id == LIBRARY_ROW_BROWSE_COLLECTIONS:
            self.run_worker(
                self._load_library_collections_capture_entry(),
                exclusive=True,
                group="library_collections_capture_entry",
            )
        elif recompose and not open_source_type:
            self.refresh(recompose=True)
