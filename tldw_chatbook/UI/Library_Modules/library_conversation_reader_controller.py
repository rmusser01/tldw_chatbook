"""Library Conversations reader controller.

Controller PR of the Conversations extraction series (the decomposition
exemplar; ``backlog/docs/library-decomposition-recipe.md``,
``.superpowers/sdd/2026-09-01-library-decomposition-foundation`` task 7).
Owns the reader cluster moved verbatim out of ``LibraryScreen`` in
``tldw_chatbook/UI/Screens/library_screen.py``: the fenced progressive
load/bootstrap/retry pipeline for one selected conversation's saved
transcript, its mounted-pane sync, and the five ``@on`` handlers the
Conversations reader pane's buttons/input/message post
(``show_library_conversation_reader_read``/``_info``,
``find_in_library_conversation``,
``library_conversation_reader_messages_synced``,
``retry_library_conversation_reader``). ``LibraryScreen`` keeps one-line
delegators under every one of those original names, since each is reached
from outside this cluster (``@on`` dispatch, or another conversations
method not part of this move).

Note: the ``@on(...)`` decorators kept on this controller's own five copies
are inert here -- Textual dispatches ``@on`` handlers against the mounted
widget/screen that receives the message, and this controller is neither.
Real dispatch happens on ``LibraryScreen``'s own ``@on``-decorated
delegator (the copy that actually receives the message), which then calls
this controller's method directly by name. The decorator survives the move
only because the byte-for-byte canon (recipe §1) forbids editing a moved
body, decorator line included.

Three names from the 2026-09-01 brief snapshot are deliberately EXCLUDED
from this cluster: ``_conversation_message_count_label``,
``_conversation_workspace_label``, ``_conversation_updated_label``. The
brief described them as "pure helpers only [the reader cluster] calls" --
true when the brief was written, no longer true by the time this task
executed (the file churns ~14 commits/day; see the recipe's §6). Their
only caller at execution time is
``LibraryScreen._selected_conversation_handoff_payload`` (the "Use in
Console" handoff-payload builder, ~line 13298) -- a browse/handoff concern,
not reader, and itself not part of this cluster or this move. Moving them
would have required either dragging that unrelated method along (out of
scope) or adding a delegator hop the recipe does not call for when the only
caller is a non-cluster, non-reader method with no other use. Left in
place on the screen.

Two binding kinds only (moved method bodies are never edited -- every name
they reference that is not this controller's own state is rebound under
the SAME name the body already used; see
``ConsoleDictationController.__init__``,
``tldw_chatbook/UI/Console_Modules/dictation.py``, the canonical worked
example this constructor's shape mirrors):

1. **Framework services** (``run_worker``, ``query_one``, and this
   project's screen-level analogue of Textual's ``self.app`` --
   ``app_instance``) are live-read from the screen via ``@property`` on
   every access -- never snapshotted, so a test that monkeypatches one on
   the screen instance after construction still reaches the patch.
2. **Everything else** the cluster depends on that is not its own state is
   a NAMED constructor dependency -- a callable the screen builds to close
   over its own attribute lookup at CALL time, not at construction time.
   Two flavors appear here: general screen/browse-cluster helpers the
   reader bodies call with ``()`` (``_build_library_conversations_state``,
   ``_library_adaptive_reader_allocation_is_current``,
   ``_run_library_service_call``, ``_conversation_records``,
   ``_conversation_record_id``), and shared shell/other-subsystem STATE the
   reader bodies read as a bare attribute (``_library_loaded``,
   ``_library_lookup_error``, ``_library_notes_focus_intent_generation`` --
   Notes-subsystem, not Conversations -- ``_library_selected_row_id`` --
   the recipe's own canonical ≥2-subsystems example -- and
   ``_selected_conversation_id``, a per-source "currently selected"
   field shared with ``_selected_media_id``/``_selected_note_id`` in the
   screen's save/restore and cross-source-navigation plumbing, never
   exclusively Conversations' despite its name). Every one of the latter
   group is read-only inside this cluster -- confirmed by checking each
   assignment site in ``library_screen.py`` falls outside the 21 moved
   method bodies -- so each surfaces here as a read-only property that
   calls its accessor and returns the value, matching
   ``ConsoleDictationController``'s ``_console_undo_histories`` pattern for
   a bare-attribute dependency.

This subsystem's OWN state (every ``_library_conversation*`` /
``_library_conversations_*`` name the moved bodies reference) is exposed
through generated properties reading
``self._conversations_state_accessor().<field>`` -- the same generator
shape Task 6 installed on ``LibraryScreen`` itself, applied here to the
controller instead.
"""
from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal, TYPE_CHECKING

from textual import on
from textual.css.query import NoMatches, QueryError
from textual.widget import Widget
from textual.widgets import Button, Input

from ...Library.library_conversation_reader_state import (
    LIBRARY_CONVERSATION_PAGE_SIZE,
    ConversationReaderRequest,
    retry_conversation,
    select_conversation,
    set_conversation_find_query,
    set_conversation_reader_mode,
    settle_conversation_continuation,
    settle_conversation_error,
    settle_conversation_page,
    settle_conversation_unavailable,
)
from ...Library.library_shell_state import LIBRARY_ROW_BROWSE_CONVERSATIONS
from ...Utils.adaptive_reader_state import resolve_adaptive_reader_layout
from ...Widgets.Library import LibraryAdaptiveReaderShell, LibraryConversationReader
from .library_conversations_state import (
    CONVERSATIONS_PLURAL_STATE_FIELDS,
    LibraryConversationsState,
)
from .screen_constants import (
    LIBRARY_CONVERSATION_READER_MAX_CHARS,
    LIBRARY_CONVERSATION_READER_PROFILE,
)

if TYPE_CHECKING:
    from ..Screens.library_screen import LibraryScreen


class LibraryConversationReaderController:
    """Owns the Conversations reader's fenced load/bootstrap/retry pipeline.

    Holds no state of its own beyond what it reads and writes through
    ``LibraryConversationsState`` (via the injected accessor). ``LibraryScreen``
    constructs exactly one of these, in ``__init__`` right after
    ``self._conversations_state``, and keeps one-line delegators for the 21
    original names this cluster moved.
    """

    def __init__(
        self,
        screen: "LibraryScreen",
        *,
        conversations_state_accessor: Callable[[], LibraryConversationsState],
        build_conversations_state: Callable[[], Any],
        adaptive_reader_allocation_is_current: Callable[[Widget], bool],
        run_library_service_call: Callable[..., Awaitable[Any]],
        conversation_records: Callable[[], tuple[Mapping[str, Any], ...]],
        conversation_record_id: Callable[[Mapping[str, Any], int], str],
        library_loaded_accessor: Callable[[], bool],
        library_lookup_error_accessor: Callable[[], str],
        notes_focus_intent_generation_accessor: Callable[[], int],
        selected_row_id_accessor: Callable[[], str],
        selected_conversation_id_accessor: Callable[[], str],
    ) -> None:
        """Build the controller and bind everything its moved bodies need.

        Every one of the 21 method bodies below is a byte-for-byte copy of
        the pre-extraction ``LibraryScreen`` method: no internal line was
        edited to retarget a call or an attribute. That is possible because
        this constructor binds every name those bodies reference that is
        not this controller's own state, under the SAME name the original
        method used. See the module docstring for the two binding kinds
        this follows (mirroring ``ConsoleDictationController.__init__``,
        the canonical worked example).

        Args:
            screen: The Library screen. Used ONLY for the three framework
                services below (``run_worker``, ``query_one``,
                ``app_instance``) -- this cluster owns no DOM of its own,
                so there is no region boundary for it to cross.
            conversations_state_accessor: Returns the live
                ``LibraryConversationsState`` (``LibraryScreen._conversations_state``).
                Backs every generated ``_library_conversation*`` /
                ``_library_conversations_*`` property below, mirroring the
                shim generator Task 6 installed on the screen.
            build_conversations_state: ``LibraryScreen._build_library_conversations_state``
                -- builds the Conversations canvas display state from local
                records. A browse-cluster concern (not reader-owned, not
                part of this move); only ``_conversation_reader_list_summary``
                calls it, for the collapsed Items-pane pager count.
            adaptive_reader_allocation_is_current: ``LibraryScreen.
                _library_adaptive_reader_allocation_is_current`` -- the
                shared shell fence every subsystem's own
                ``_sync_library_<x>_reader_layout_from_shell`` checks before
                patching a resized shell in place.
            run_library_service_call: ``LibraryScreen._run_library_service_call``
                -- the shared off-thread service-call wrapper every Library
                subsystem's detail/list fetch goes through.
            conversation_records: ``LibraryScreen._conversation_records`` --
                the retained Conversations list-page records. Shared with
                the browse cluster (not reader-exclusive); this cluster
                only reads it to resolve one record by id.
            conversation_record_id: ``LibraryScreen._conversation_record_id``
                -- the general record-id resolver shared across the whole
                Conversations subsystem, not reader-specific.
            library_loaded_accessor: Reads ``LibraryScreen._library_loaded``
                -- whether the shared local-source lookup has completed at
                least once. Shell-wide, not Conversations-owned; read-only
                here (``_sync_library_conversation_reader``'s loading-status
                copy).
            library_lookup_error_accessor: Reads ``LibraryScreen.
                _library_lookup_error`` -- the shared local-source lookup's
                last error copy, same shell-wide scope and read-only use as
                ``library_loaded_accessor`` immediately above.
            notes_focus_intent_generation_accessor: Reads ``LibraryScreen.
                _library_notes_focus_intent_generation`` -- the Notes
                subsystem's find/focus-intent generation counter, reused by
                Conversations' own deferred-Find-focus fence
                (``_finish_library_conversation_find_focus``,
                ``find_in_library_conversation``). Read-only here.
            selected_row_id_accessor: Reads ``LibraryScreen.
                _library_selected_row_id`` -- the recipe's own canonical
                ≥2-subsystems shared field (226 refs at the time of
                writing). Read-only here: every write site in
                ``library_screen.py`` falls outside this cluster's 21
                methods.
            selected_conversation_id_accessor: Reads ``LibraryScreen.
                _selected_conversation_id`` -- a per-source "currently
                selected" field parallel to ``_selected_media_id``/
                ``_selected_note_id`` in the screen's save/restore and
                cross-source-navigation plumbing; despite its name, never
                exclusively Conversations-reader-owned. Read-only here.
        """
        self._screen = screen
        self._conversations_state_accessor = conversations_state_accessor
        self._build_conversations_state_fn = build_conversations_state
        self._adaptive_reader_allocation_is_current_fn = (
            adaptive_reader_allocation_is_current
        )
        self._run_library_service_call_fn = run_library_service_call
        self._conversation_records_fn = conversation_records
        self._conversation_record_id_fn = conversation_record_id
        self._library_loaded_accessor = library_loaded_accessor
        self._library_lookup_error_accessor = library_lookup_error_accessor
        self._notes_focus_intent_generation_accessor = (
            notes_focus_intent_generation_accessor
        )
        self._selected_row_id_accessor = selected_row_id_accessor
        self._selected_conversation_id_accessor = selected_conversation_id_accessor

    # -- framework services: live-read properties, never snapshotted -----

    @property
    def run_worker(self) -> Any:
        """``Screen.run_worker``, bound. See ``__init__``'s docstring."""
        return self._screen.run_worker

    @property
    def query_one(self) -> Any:
        """``Screen.query_one``, bound. See ``__init__``'s docstring."""
        return self._screen.query_one

    @property
    def app_instance(self) -> Any:
        """The running app instance, live-read from the screen.

        This project's screen-level analogue of Textual's own ``self.app``
        -- see ``__init__``'s docstring.
        """
        return self._screen.app_instance

    # -- named constructor dependencies -----------------------------------

    @property
    def _build_library_conversations_state(self) -> Any:
        """The injected ``build_conversations_state``. Kept under this name
        so the moved bodies below still call
        ``self._build_library_conversations_state()`` unchanged. See
        ``__init__``'s docstring."""
        return self._build_conversations_state_fn

    @property
    def _library_adaptive_reader_allocation_is_current(self) -> Any:
        """The injected ``adaptive_reader_allocation_is_current``. See
        ``__init__``'s docstring."""
        return self._adaptive_reader_allocation_is_current_fn

    @property
    def _run_library_service_call(self) -> Any:
        """The injected ``run_library_service_call``. See ``__init__``'s
        docstring."""
        return self._run_library_service_call_fn

    @property
    def _conversation_records(self) -> Any:
        """The injected ``conversation_records``. See ``__init__``'s
        docstring."""
        return self._conversation_records_fn

    @property
    def _conversation_record_id(self) -> Any:
        """The injected ``conversation_record_id``. See ``__init__``'s
        docstring."""
        return self._conversation_record_id_fn

    @property
    def _library_loaded(self) -> bool:
        """Calls the injected ``library_loaded_accessor``. Read-only here;
        see ``__init__``'s docstring."""
        return self._library_loaded_accessor()

    @property
    def _library_lookup_error(self) -> str:
        """Calls the injected ``library_lookup_error_accessor``. Read-only
        here; see ``__init__``'s docstring."""
        return self._library_lookup_error_accessor()

    @property
    def _library_notes_focus_intent_generation(self) -> int:
        """Calls the injected ``notes_focus_intent_generation_accessor``.
        Read-only here; see ``__init__``'s docstring."""
        return self._notes_focus_intent_generation_accessor()

    @property
    def _library_selected_row_id(self) -> str:
        """Calls the injected ``selected_row_id_accessor``. Read-only here;
        see ``__init__``'s docstring."""
        return self._selected_row_id_accessor()

    @property
    def _selected_conversation_id(self) -> str:
        """Calls the injected ``selected_conversation_id_accessor``.
        Read-only here; see ``__init__``'s docstring."""
        return self._selected_conversation_id_accessor()

    # -- moved bodies (byte-for-byte; see module docstring) ---------------

    def _conversation_reader_record(
        self, conversation_id: str
    ) -> Mapping[str, Any] | None:
        """Return the exact retained list record for one conversation id."""
        for index, record in enumerate(self._conversation_records()):
            if self._conversation_record_id(record, index) == conversation_id:
                return record
        return None

    def _conversation_reader_list_summary(self) -> str:
        """Summarize a collapsed Items pane inside protected work status."""
        state = self._build_library_conversations_state()
        pager = state.pager
        count = pager.title_count if pager is not None else None
        heading = "Conversations" if count is None else f"Conversations ({count})"
        titles = [
            str(record.get("title") or "").strip()
            for record in self._conversation_records()[:2]
        ]
        visible_titles = [title for title in titles if title]
        return " · ".join((heading, *visible_titles))

    @staticmethod
    def _conversation_reader_record_version(
        record: Mapping[str, Any] | None,
    ) -> int | None:
        """Read an authoritative optimistic-lock version without inventing one."""
        value = record.get("version") if record is not None else None
        return value if type(value) is int and value >= 0 else None

    def _sync_library_conversation_reader(self) -> bool:
        """Patch the mounted work pane from controller-owned pure state."""
        try:
            reader = self.query_one(
                "#library-conversation-reader", LibraryConversationReader
            )
        except (NoMatches, QueryError):
            return False
        metadata = dict(self._library_conversation_reader_loaded_metadata)
        if not self._library_loaded and not self._library_lookup_error:
            metadata["_list_status"] = "Loading local Library sources…"
        elif self._library_lookup_error:
            metadata["_list_status"] = self._library_lookup_error
        if not self._library_conversation_reader_layout.items_open:
            metadata["_list_summary"] = self._conversation_reader_list_summary()
        reader.sync_state(
            self._library_conversation_reader_state,
            loaded_metadata=metadata,
            selected_metadata=self._library_conversation_reader_selected_metadata,
        )
        self._finish_library_conversation_find_focus()
        return True

    def _finish_library_conversation_find_focus(self) -> None:
        """Reveal a deferred Find match only while its user intent is current."""
        intent = self._library_conversation_find_focus_intent
        state = self._library_conversation_reader_state
        if intent is None or not state.find_complete or not state.find_matches:
            return
        generation, focus_generation, query = intent
        if (
            self._library_selected_row_id != LIBRARY_ROW_BROWSE_CONVERSATIONS
            or generation != state.generation
            or query != state.find_query
            or focus_generation != self._library_notes_focus_intent_generation
        ):
            self._library_conversation_find_focus_intent = None
            return
        try:
            reader = self.query_one(
                "#library-conversation-reader", LibraryConversationReader
            )
        except (NoMatches, QueryError):
            return
        if reader.state.generation != generation or reader.state.find_query != query:
            self._library_conversation_find_focus_intent = None
            return
        if reader.focus_find_match(state.find_matches[0].message_id):
            self._library_conversation_find_focus_intent = None

    def _conversation_reader_request_is_current(
        self, request: ConversationReaderRequest
    ) -> bool:
        """Check the complete destination/id/version/generation request fence."""
        state = self._library_conversation_reader_state
        return (
            self._library_conversation_reader_mounted_authority
            and self._library_selected_row_id == LIBRARY_ROW_BROWSE_CONVERSATIONS
            and request.destination == "conversations"
            and request.conversation_id == state.selected_id
            and request.version == state.selected_version
            and request.generation == state.generation
        )

    def _invalidate_library_conversation_reader_authority(self) -> None:
        """Revoke every page/continuation and loaded-action generation."""
        state = self._library_conversation_reader_state
        self._library_conversation_reader_state = dataclasses.replace(
            state,
            generation=state.generation + 1,
            loading=False,
        )
        self._library_conversation_find_focus_intent = None

    def _conversation_reader_service(self) -> Any | None:
        """Return the existing local conversation detail authority."""
        direct = getattr(self.app_instance, "local_chat_conversation_service", None)
        if callable(getattr(direct, "get_library_conversation_messages", None)):
            return direct
        scope = getattr(self.app_instance, "chat_conversation_scope_service", None)
        local = getattr(scope, "local_service", None)
        if callable(getattr(local, "get_library_conversation_messages", None)):
            return local
        return None

    def _ensure_library_conversation_reader_selection(self) -> None:
        """Start the initial selected-row read once the permanent pane mounts."""
        conversation_id = self._selected_conversation_id
        if not conversation_id or self._library_conversations_select_mode:
            return
        state = self._library_conversation_reader_state
        record_version = self._conversation_reader_record_version(
            self._conversation_reader_record(conversation_id)
        )
        if state.selected_id == conversation_id and (
            (
                state.loading
                and (record_version is None or state.selected_version == record_version)
            )
            or (
                state.loaded_actions_eligible
                and (record_version is None or state.loaded_version == record_version)
            )
        ):
            return
        self._start_library_conversation_reader_selection(conversation_id)

    def _start_library_conversation_reader_selection(
        self, conversation_id: str
    ) -> None:
        """Select one list record and launch its fenced progressive reader."""
        self._library_conversation_deleted_selection_id = ""
        record = self._conversation_reader_record(conversation_id)
        if record is not None:
            self._library_conversation_reader_selected_metadata = dict(record)
        version = self._conversation_reader_record_version(record)
        if version is None:
            state = self._library_conversation_reader_state
            generation = state.generation + 1
            self._library_conversation_reader_state = dataclasses.replace(
                state,
                selected_id=conversation_id,
                selected_version=None,
                generation=generation,
                mode=state.mode if conversation_id == state.loaded_id else "read",
                error=None,
                loading=True,
                unavailable=False,
                bulk_active=False,
                bulk_selected_count=0,
                bulk_loaded_preview_selected=None,
            )
            self._sync_library_conversation_reader()
            self.run_worker(
                self._bootstrap_library_conversation_reader(
                    conversation_id, generation
                ),
                exclusive=True,
                group="library_conversation_reader",
            )
            return
        state, request = select_conversation(
            self._library_conversation_reader_state,
            conversation_id,
            version=version,
        )
        self._library_conversation_reader_state = state
        self._sync_library_conversation_reader()
        self.run_worker(
            self._load_library_conversation_reader(request),
            exclusive=True,
            group="library_conversation_reader",
        )

    async def _bootstrap_library_conversation_reader(
        self, conversation_id: str, bootstrap_generation: int
    ) -> None:
        """Obtain a missing real version from the authoritative detail envelope."""
        service = self._conversation_reader_service()
        read = getattr(service, "get_library_conversation_messages", None)
        if not callable(read):
            state = self._library_conversation_reader_state
            if self._conversation_reader_bootstrap_is_current(
                conversation_id, bootstrap_generation
            ):
                self._library_conversation_reader_state = dataclasses.replace(
                    state,
                    error="Conversation detail is unavailable.",
                    loading=False,
                    unavailable=False,
                )
                self._sync_library_conversation_reader()
            return
        try:
            response = await self._run_library_service_call(
                read,
                conversation_id,
                message_offset=0,
                message_limit=LIBRARY_CONVERSATION_PAGE_SIZE,
                max_chars=LIBRARY_CONVERSATION_READER_MAX_CHARS,
            )
        except Exception:
            if self._conversation_reader_bootstrap_is_current(
                conversation_id, bootstrap_generation
            ):
                state = self._library_conversation_reader_state
                self._library_conversation_reader_state = dataclasses.replace(
                    state,
                    error="Couldn't load this conversation. Try again.",
                    loading=False,
                    unavailable=False,
                )
                self._sync_library_conversation_reader()
            return
        if not self._conversation_reader_bootstrap_is_current(
            conversation_id, bootstrap_generation
        ):
            return
        state = self._library_conversation_reader_state
        if response is None:
            self._library_conversation_reader_state = dataclasses.replace(
                state,
                error="Conversation unavailable.",
                loading=False,
                unavailable=True,
            )
            self._sync_library_conversation_reader()
            return
        version = response.get("version") if isinstance(response, Mapping) else None
        if type(version) is not int or version < 0:
            self._library_conversation_reader_state = dataclasses.replace(
                state,
                error="Conversation detail returned no authoritative version.",
                loading=False,
            )
            self._sync_library_conversation_reader()
            return
        selected, request = select_conversation(state, conversation_id, version=version)
        self._library_conversation_reader_state = selected
        await self._load_library_conversation_reader(
            request,
            initial_response=response,
        )

    def _conversation_reader_bootstrap_is_current(
        self,
        conversation_id: str,
        bootstrap_generation: int,
    ) -> bool:
        """Fence a version bootstrap to its mounted Conversations selection."""
        state = self._library_conversation_reader_state
        return (
            self._library_conversation_reader_mounted_authority
            and self._library_selected_row_id == LIBRARY_ROW_BROWSE_CONVERSATIONS
            and state.selected_id == conversation_id
            and state.selected_version is None
            and state.generation == bootstrap_generation
        )

    async def _load_library_conversation_reader(
        self,
        request: ConversationReaderRequest,
        *,
        initial_response: Mapping[str, Any] | None = None,
    ) -> None:
        """Settle bounded pages and body continuations incrementally off-loop."""
        service = self._conversation_reader_service()
        read = getattr(service, "get_library_conversation_messages", None)
        if not callable(read):
            self._library_conversation_reader_state = settle_conversation_error(
                self._library_conversation_reader_state,
                request,
                "Conversation detail is unavailable.",
            )
            self._sync_library_conversation_reader()
            return

        response = initial_response
        while self._conversation_reader_request_is_current(request):
            try:
                if response is None:
                    response = await self._run_library_service_call(
                        read,
                        request.conversation_id,
                        message_offset=request.message_offset,
                        message_limit=request.message_limit,
                        max_chars=LIBRARY_CONVERSATION_READER_MAX_CHARS,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                response = None
                if self._conversation_reader_request_is_current(request):
                    self._library_conversation_reader_state = settle_conversation_error(
                        self._library_conversation_reader_state,
                        request,
                        "Couldn't load this conversation. Try again.",
                    )
                    self._sync_library_conversation_reader()
                return
            if not self._conversation_reader_request_is_current(request):
                return
            if response is None:
                self._library_conversation_reader_state = (
                    settle_conversation_unavailable(
                        self._library_conversation_reader_state, request
                    )
                )
                self._sync_library_conversation_reader()
                return
            try:
                before_settle = self._library_conversation_reader_state
                previous_loaded_generation = before_settle.loaded_generation
                settled = settle_conversation_page(
                    before_settle,
                    request,
                    response,
                )
            except (TypeError, ValueError):
                before_settle = self._library_conversation_reader_state
                settled = settle_conversation_error(
                    before_settle,
                    request,
                    "Conversation detail was invalid. Try again.",
                )
            self._library_conversation_reader_state = settled
            request_content_loaded = (
                settled.loaded_id == request.conversation_id
                and settled.loaded_generation == request.generation
            )
            if (
                settled is not before_settle
                and settled.error is None
                and request_content_loaded
            ):
                metadata = dict(
                    self._library_conversation_reader_selected_metadata
                    if previous_loaded_generation != request.generation
                    else self._library_conversation_reader_loaded_metadata
                )
                title = response.get("title")
                if isinstance(title, str) and title.strip():
                    metadata["title"] = title.strip()
                metadata["version"] = request.version
                self._library_conversation_reader_loaded_metadata = metadata
            self._sync_library_conversation_reader()
            await asyncio.sleep(0)
            if settled.error:
                return

            for message in tuple(settled.messages):
                while (
                    not message.complete
                    and self._conversation_reader_request_is_current(request)
                ):
                    try:
                        continuation = await self._run_library_service_call(
                            read,
                            request.conversation_id,
                            message_offset=request.message_offset,
                            message_limit=request.message_limit,
                            max_chars=LIBRARY_CONVERSATION_READER_MAX_CHARS,
                            message_id=message.message_id,
                            char_start=len(message.text),
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        self._library_conversation_reader_state = (
                            settle_conversation_error(
                                self._library_conversation_reader_state,
                                request,
                                "Couldn't finish loading a saved message. Try again.",
                            )
                        )
                        self._sync_library_conversation_reader()
                        return
                    before = self._library_conversation_reader_state
                    if continuation is None:
                        self._library_conversation_reader_state = (
                            settle_conversation_unavailable(before, request)
                        )
                        self._sync_library_conversation_reader()
                        return
                    self._library_conversation_reader_state = (
                        settle_conversation_continuation(
                            before,
                            request,
                            continuation,
                        )
                    )
                    if self._library_conversation_reader_state is before:
                        self._library_conversation_reader_state = (
                            settle_conversation_error(
                                before,
                                request,
                                "Saved message continuation was invalid. Try again.",
                            )
                        )
                        self._sync_library_conversation_reader()
                        return
                    self._sync_library_conversation_reader()
                    await asyncio.sleep(0)
                    message = next(
                        candidate
                        for candidate in self._library_conversation_reader_state.messages
                        if candidate.message_id == message.message_id
                    )

            state = self._library_conversation_reader_state
            if state.complete:
                return
            next_offset = len(state.messages)
            if next_offset <= request.message_offset:
                self._library_conversation_reader_state = settle_conversation_error(
                    state,
                    request,
                    "Conversation loading made no progress. Try again.",
                )
                self._sync_library_conversation_reader()
                return
            request = dataclasses.replace(request, message_offset=next_offset)
            response = None

    def _mirror_library_conversation_reader_preference(
        self,
        key: Literal["library_open", "items_open"],
        value: bool,
    ) -> None:
        """Mirror one optimistic Conversations pane choice into app config."""
        app_config = getattr(self.app_instance, "app_config", None)
        if not isinstance(app_config, dict):
            return
        library_config = app_config.setdefault("library", {})
        if not isinstance(library_config, dict):
            library_config = {}
            app_config["library"] = library_config
        section_name = "reader" if key == "library_open" else "conversations_reader"
        section = library_config.setdefault(section_name, {})
        if not isinstance(section, dict):
            section = {}
            library_config[section_name] = section
        section[key] = value

    def _retry_library_conversation_reader(self) -> None:
        """Start one generation-fenced reader retry when selection is known."""
        if (
            self._library_conversations_select_mode
            or self._library_conversation_reader_state.bulk_active
        ):
            return
        try:
            state, request = retry_conversation(self._library_conversation_reader_state)
        except ValueError:
            selected = self._library_conversation_reader_state.selected_id
            if selected:
                self._start_library_conversation_reader_selection(selected)
            return
        self._library_conversation_reader_state = state
        self._sync_library_conversation_reader()
        self.run_worker(
            self._load_library_conversation_reader(request),
            exclusive=True,
            group="library_conversation_reader",
        )

    def _sync_library_conversation_reader_layout_from_shell(
        self,
        priority: Literal["library", "items"] | None = None,
    ) -> None:
        """Resolve the settled Conversations shell and patch it in place."""
        try:
            shell = self.query_one(
                "#library-conversations-reader-shell", LibraryAdaptiveReaderShell
            )
        except (NoMatches, QueryError):
            return
        width = shell.region.width
        if not self._library_adaptive_reader_allocation_is_current(shell):
            return
        previous = self._library_conversation_reader_layout
        if (
            previous.reader_width == 0
            and previous.library_width == 0
            and previous.items_width == 0
        ):
            previous = None
        layout = resolve_adaptive_reader_layout(
            width,
            self._library_conversation_reader_preferences,
            LIBRARY_CONVERSATION_READER_PROFILE,
            previous=previous,
            priority=priority,
        )
        shell.sync_layout(layout)
        self._library_conversation_reader_layout = layout
        self._sync_library_conversation_reader()

    @on(Button.Pressed, "#library-conversation-reader-read")
    def show_library_conversation_reader_read(self, event: Button.Pressed) -> None:
        """Show the retained saved transcript."""
        event.stop()
        self._library_conversation_reader_state = set_conversation_reader_mode(
            self._library_conversation_reader_state, "read"
        )
        self._sync_library_conversation_reader()

    @on(Button.Pressed, "#library-conversation-reader-info")
    def show_library_conversation_reader_info(self, event: Button.Pressed) -> None:
        """Show truthful local conversation metadata."""
        event.stop()
        self._library_conversation_reader_state = set_conversation_reader_mode(
            self._library_conversation_reader_state, "info"
        )
        self._sync_library_conversation_reader()

    @on(Input.Submitted, "#library-conversation-reader-find")
    def find_in_library_conversation(self, event: Input.Submitted) -> None:
        """Find only after every saved message body is fully hydrated."""
        event.stop()
        self._library_conversation_reader_state = set_conversation_find_query(
            self._library_conversation_reader_state,
            event.value,
        )
        self._sync_library_conversation_reader()
        state = self._library_conversation_reader_state
        if not state.loading and not state.complete and state.selected_id:
            self._retry_library_conversation_reader()
            state = self._library_conversation_reader_state
        self._library_conversation_find_focus_intent = (
            (
                state.generation,
                self._library_notes_focus_intent_generation,
                state.find_query,
            )
            if state.find_query
            else None
        )
        if state.find_complete and state.find_matches:
            self._finish_library_conversation_find_focus()

    @on(LibraryConversationReader.MessagesSynced)
    def library_conversation_reader_messages_synced(
        self,
        event: LibraryConversationReader.MessagesSynced,
    ) -> None:
        """Revalidate deferred Find focus after current transcript rows mount."""
        event.stop()
        state = self._library_conversation_reader_state
        if (
            event.reader_generation != state.generation
            or event.find_query != state.find_query
        ):
            self._library_conversation_find_focus_intent = None
            return
        self._finish_library_conversation_find_focus()

    @on(Button.Pressed, "#library-conversation-reader-retry")
    def retry_library_conversation_reader(self, event: Button.Pressed) -> None:
        """Retry the selected detail with a fresh pure-state generation."""
        event.stop()
        self._retry_library_conversation_reader()

# --- BEGIN generated conversations-state shims (delete wholesale at cleanup) ---
# task 7: exposes every `LibraryConversationsState` field under its original
# `_library_conversation*`/`_library_conversations_*` name on THIS controller
# too, reading/writing through the injected `conversations_state_accessor`
# instead of a direct `self._conversations_state` attribute (this class has
# none) -- same generator shape as the shim block `LibraryScreen` carries
# (task 6), attached programmatically so the class body gains no
# `FunctionDef`s.
#
# task 8: `CONVERSATIONS_PLURAL_STATE_FIELDS` now imported from
# `library_conversations_state` (the dataclass's own module) instead of
# kept as a local literal -- task 7's own fix round 1 flagged this set's
# independent copy here and on `LibraryScreen` as a concrete drift risk; a
# third controller (task 8's `LibraryConversationsController`) importing
# the same shared constant closes that gap for good. See that module's
# docstring for the full note.
for _rc_field in dataclasses.fields(LibraryConversationsState):
    _rc_prefix = (
        "_library_conversations_"
        if _rc_field.name in CONVERSATIONS_PLURAL_STATE_FIELDS
        else "_library_conversation_"
    )
    setattr(
        LibraryConversationReaderController,
        _rc_prefix + _rc_field.name,
        property(
            lambda self, _n=_rc_field.name: getattr(
                self._conversations_state_accessor(), _n
            ),
            lambda self, value, _n=_rc_field.name: setattr(
                self._conversations_state_accessor(), _n, value
            ),
        ),
    )
del _rc_field, _rc_prefix
# --- END generated conversations-state shims ---
