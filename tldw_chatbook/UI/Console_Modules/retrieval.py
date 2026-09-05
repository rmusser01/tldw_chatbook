"""Controller-owned Console retrieval and cached inspector policy."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
import re
from typing import Any

from loguru import logger

from ...config import get_cli_setting
from ...Event_Handlers.Chat_Events.chat_events_console_dictionaries import (
    console_attachable_dictionaries,
    console_attached_dictionaries,
    handle_console_dictionary_attach,
    handle_console_dictionary_detach,
)
from ...Widgets.Persona_Widgets.dictionary_picker import DictionaryPicker

from ...Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore
from ...Chat.console_display_state import (
    ConsoleDisplayRow,
    ConsoleInspectorAction,
    ConsoleRetrievalScopeState,
)
from ...Chat.console_live_work import ConsoleLiveWorkLaunch
from ...Chat.console_turn_context import ConsoleTurnExecutionContext
from ...Chat.rag_scope import (
    RagScope,
    read_conversation_scope,
    write_conversation_scope,
)
from ...Chat.scope_picker_listers import (
    build_keyword_tag_lister,
    build_media_source_lister,
    build_notes_source_lister,
)
from ...Event_Handlers.Chat_Events.chat_rag_events import (
    capture_console_staged_evidence_for_chat,
    resolve_effective_scope_for_chat,
    resolve_scope_for_session,
)
from ...Library.library_rag_service import (
    LibraryRagSearchOutcome,
    LibraryRagSearchRequest,
    run_library_rag_search,
    scope_empty_recovery_state,
)
from ...Library.library_rag_state import library_rag_source_scope_summary
from ...Widgets.Console.console_library_search_modal import (
    CONSOLE_LIBRARY_SEARCH_QUERY_MAX_CHARS,
    CONSOLE_RAG_SOURCE_SUMMARY_PREFIX,
    ConsoleLibrarySearchModal,
    ConsoleLibrarySearchResult,
    sanitize_console_library_rag_query,
)
from ...Widgets.Console.console_retrieval_scope_row import (
    console_retrieval_scope_label,
)
from ..Views.RAGSearch.search_handoff import (
    build_library_rag_console_live_work_payload,
    build_library_rag_evidence_bundle,
)

logger = logger.bind(module="ConsoleRetrievalController")

# Roleplay P1h: bounds passed to `Chat_Dictionary_Lib.apply_active_chatdicts_to_text`
# for the native Console send-path applier (`_console_chat_dictionary_applier`).
_CHATDICT_MAX_TOKENS = 500
_CHATDICT_STRATEGY = "sorted_evenly"

CONSOLE_LIBRARY_RAG_RECOVERY_COPY = "Review citations before sending."
AUTO_RAG_QUERY_MAX_CHARS = CONSOLE_LIBRARY_SEARCH_QUERY_MAX_CHARS


def source_mentions_rag(source: Any) -> bool:
    """Return whether a source label contains a standalone RAG token."""
    return "rag" in re.split(r"[^a-z0-9]+", str(source or "").lower())


def launch_has_rag_source_payload(launch: ConsoleLiveWorkLaunch) -> bool:
    """Return whether a launch carries at least one source-bearing field."""
    return any(
        str(launch.payload.get(key) or "").strip()
        for key in (
            "source_id",
            "source_count",
            "citation_count",
            "chunk_id",
            "query",
            "result_id",
        )
    )


def _run_dictionary_summary_off_thread(
    service: Any,
    conversation_id: Any,
    character_id: Any,
) -> Any:
    """Run the async dictionary summary on a private worker-thread loop."""
    return asyncio.run(
        service.summarize_active_dictionaries(
            conversation_id, character_id, mode="local"
        )
    )


class ConsoleRetrievalController:
    """Own non-DOM retrieval state and behavior for the Console."""

    def __init__(
        self,
        *,
        app_instance: Any,
        active_native_session: Callable[[], ConsoleChatSession | None],
        current_conversation_id: Callable[[], str | None],
        clear_evidence_sent_notice: Callable[[], None],
        consume_pending_launch: Callable[[], ConsoleLiveWorkLaunch | None],
        release_consumed_launch: Callable[[ConsoleLiveWorkLaunch, Any], None],
        is_mounted: Callable[[], bool],
        sync_retrieval_scope_row: Callable[[], None],
        sync_control_bar: Callable[[], None],
        request_control_bar_sync: Callable[[], None],
        dictionary_scope_service: Callable[[], Any],
        finish_dictionary_dialog: Callable[[], None],
        finish_worldbook_dialog: Callable[[], None],
        set_library_rag_source_scope: Callable[[Any], None],
        set_library_rag_query: Callable[[str], None],
        run_library_rag_action: Callable[[], None],
        push_screen: Callable[..., Any],
        library_rag_source_scope: Callable[[], tuple[str, ...]],
        library_rag_top_k: Callable[[], int],
        pending_launch: Callable[[], ConsoleLiveWorkLaunch | None],
        set_pending_launch: Callable[[ConsoleLiveWorkLaunch | None], None],
        set_pending_auto_open: Callable[[bool], None],
        set_evidence_sent_notice: Callable[[int | None], None],
        sync_pending_launch_surfaces: Callable[[], bool],
        refresh_screen: Callable[[], None],
        has_staged_evidence: Callable[[], bool],
        chat_store: Callable[[], ConsoleChatStore | None] | None = None,
        composer_draft: Callable[[], str | None] | None = None,
        library_rag_query: Callable[[], str] | None = None,
    ) -> None:
        """Bind explicit late-bound screen edges and initialize owned state."""
        self.app_instance = app_instance
        self._active_native_session = active_native_session
        self._current_conversation_id = current_conversation_id
        self._clear_evidence_sent_notice = clear_evidence_sent_notice
        self._consume_pending_launch = consume_pending_launch
        self._release_consumed_launch = release_consumed_launch
        self._is_mounted = is_mounted
        self._sync_retrieval_scope_row = sync_retrieval_scope_row
        self._sync_control_bar = sync_control_bar
        self._request_control_bar_sync = request_control_bar_sync
        self._dictionary_scope_service = dictionary_scope_service
        self._finish_dictionary_dialog = finish_dictionary_dialog
        self._finish_worldbook_dialog = finish_worldbook_dialog
        self._set_library_rag_source_scope = set_library_rag_source_scope
        self._set_library_rag_query = set_library_rag_query
        self._run_library_rag_action = run_library_rag_action
        self._composer_draft = composer_draft or (lambda: None)
        self._library_rag_query = library_rag_query or (lambda: "")
        self._push_screen = push_screen
        self._library_rag_source_scope = library_rag_source_scope
        self._library_rag_top_k = library_rag_top_k
        self._pending_launch = pending_launch
        self._set_pending_launch = set_pending_launch
        self._set_pending_auto_open = set_pending_auto_open
        self._set_evidence_sent_notice = set_evidence_sent_notice
        self._sync_pending_launch_surfaces = sync_pending_launch_surfaces
        self._refresh_screen = refresh_screen
        self._has_staged_evidence = has_staged_evidence
        self._chat_store = chat_store or (lambda: None)

        self._console_retrieval_scope_cache: dict[str, RagScope | None] = {}
        self._console_effective_scope_cache: dict[str, ConsoleRetrievalScopeState] = {}
        self._active_dictionaries_summary: dict | None = None
        self._last_console_dictionary_scope_ids: (
            tuple[str | None, int | None] | None
        ) = None
        self._active_world_books_summary: dict | None = None
        self._last_console_world_book_scope_ids: tuple[str | None] | None = None

    async def _capture_console_staged_rag(
        self,
        draft: str,
        turn_context: ConsoleTurnExecutionContext | None = None,
    ) -> Any:
        """Capture one explicitly staged manual evidence launch."""
        self._clear_evidence_sent_notice()
        launch = self._consume_pending_launch()
        result = await capture_console_staged_evidence_for_chat(
            self.app_instance,
            launch,
            user_message=draft,
        )
        context = getattr(result, "context", None)
        if launch is not None and isinstance(context, str) and context.strip():
            self._release_consumed_launch(launch, result)
        return result

    def _snapshot_console_staged_evidence(self) -> ConsoleLiveWorkLaunch | None:
        """Snapshot the exact live launch admitted with a prepared send."""

        # Every admitted send supersedes the prior send's transient receipt,
        # including the common no-staging fast path where capture is skipped.
        self._clear_evidence_sent_notice()
        return self._pending_launch()

    async def _capture_frozen_console_staged_rag(
        self,
        draft: str,
        turn_context: ConsoleTurnExecutionContext | None,
        launch: ConsoleLiveWorkLaunch | None,
    ) -> Any:
        """Capture an admitted launch without consulting newer staged state."""

        del turn_context
        self._clear_evidence_sent_notice()
        result = await capture_console_staged_evidence_for_chat(
            self.app_instance,
            launch,
            user_message=draft,
        )
        return result

    def _release_frozen_console_staged_rag(
        self, launch: ConsoleLiveWorkLaunch | None, result: Any
    ) -> None:
        """Release one exact frozen launch only after turn acceptance."""

        context = getattr(result, "context", None)
        if launch is not None and isinstance(context, str) and context.strip():
            self._release_consumed_launch(launch, result)

    def _build_console_retrieval_scope_state(self) -> ConsoleRetrievalScopeState:
        """Return the cached effective scope for the active session."""
        session = self._active_native_session()
        if session is None:
            return ConsoleRetrievalScopeState.unscoped()
        cache_key = session.persisted_conversation_id or session.id
        cached = self._console_effective_scope_cache.get(cache_key)
        if cached is not None:
            return cached
        if session.persisted_conversation_id is None:
            return ConsoleRetrievalScopeState.from_scope(session.rag_scope_holder.scope)
        return ConsoleRetrievalScopeState.unscoped()

    def _console_retrieval_scope_run_recipe_count(self) -> int | None:
        """Return the effective scoped-item count, or ``None`` when unscoped."""
        state = self._build_console_retrieval_scope_state()
        if state.is_empty:
            return 0
        return state.item_count if state.is_scoped else None

    async def _resolve_console_effective_scope_state(
        self, session: ConsoleChatSession
    ) -> ConsoleRetrievalScopeState:
        """Resolve and cache the conversation/workspace effective scope."""
        resolution = await resolve_scope_for_session(self.app_instance, session)
        if session.persisted_conversation_id is not None:
            self._console_retrieval_scope_cache[session.persisted_conversation_id] = (
                resolution.conv_scope
            )
        state = ConsoleRetrievalScopeState.from_effective(
            resolution.effective,
            conv_item_count=(
                len(resolution.conv_scope.items)
                if resolution.conv_scope is not None
                else None
            ),
            ws_item_count=(
                len(resolution.ws_scope.items)
                if resolution.ws_scope is not None
                else None
            ),
        )
        cache_key = session.persisted_conversation_id or session.id
        self._console_effective_scope_cache[cache_key] = state
        return state

    async def _refresh_console_effective_scope_and_sync(
        self, session: ConsoleChatSession
    ) -> None:
        """Resolve effective scope, then refresh mounted screen projections."""
        await self._resolve_console_effective_scope_state(session)
        if self._is_mounted():
            self._sync_retrieval_scope_row()
            self._sync_control_bar()

    async def _warm_console_effective_scope_cache_if_stale(self) -> None:
        """Warm a restored persisted session's effective-scope cache once."""
        session = self._active_native_session()
        if session is None or session.persisted_conversation_id is None:
            return
        cache_key = session.persisted_conversation_id
        if cache_key in self._console_effective_scope_cache:
            return
        try:
            await self._refresh_console_effective_scope_and_sync(session)
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to warm retrieval scope cache for conversation {}",
                cache_key,
            )

    @staticmethod
    async def _read_console_retrieval_scope(
        db: Any, conversation_id: str
    ) -> RagScope | None:
        """Read a stored scope off-loop except for thread-local memory DBs."""
        if getattr(db, "is_memory_db", False):
            return read_conversation_scope(db, conversation_id)
        return await asyncio.to_thread(read_conversation_scope, db, conversation_id)

    @staticmethod
    async def _write_console_retrieval_scope(
        db: Any, conversation_id: str, scope: RagScope | None
    ) -> None:
        """Write a stored scope off-loop except for thread-local memory DBs."""
        if getattr(db, "is_memory_db", False):
            write_conversation_scope(db, conversation_id, scope)
            return
        await asyncio.to_thread(write_conversation_scope, db, conversation_id, scope)

    def _console_scope_picker_listers(self) -> tuple[Any, Any, Any]:
        """Build the real media, notes, and keyword listers."""
        user_id = getattr(self.app_instance, "notes_user_id", None) or "default_user"
        return (
            build_media_source_lister(self.app_instance),
            build_notes_source_lister(self.app_instance, user_id=user_id),
            build_keyword_tag_lister(self.app_instance),
        )

    async def _apply_console_retrieval_scope_save(
        self,
        session: ConsoleChatSession,
        scope: RagScope | None,
    ) -> None:
        """Persist and publish a scope inside the store transition boundary."""

        store = self._chat_store()
        if store is None:
            return
        with store.fork_source_transition(session.id):
            await self._apply_console_retrieval_scope_save_transition(
                store, session, scope
            )

    async def _apply_console_retrieval_scope_save_transition(
        self,
        store: ConsoleChatStore,
        session: ConsoleChatSession,
        scope: RagScope | None,
    ) -> None:
        """Persist or session-hold a picker result, then refresh effective scope."""
        if session.persisted_conversation_id is not None:
            conversation_id = session.persisted_conversation_id
            db = getattr(self.app_instance, "chachanotes_db", None)
            if db is None:
                self.app_instance.notify(
                    "Couldn't save scope: no database is available.", severity="error"
                )
                return
            from ...DB.ChaChaNotes_DB import ConflictError

            try:
                await self._write_console_retrieval_scope(db, conversation_id, scope)
            except ConflictError:
                logger.opt(exception=True).warning(
                    "Retrieval scope write conflict for conversation {}",
                    conversation_id,
                )
                self.app_instance.notify(
                    "Couldn't save scope: the conversation was modified "
                    "concurrently — try again.",
                    severity="error",
                )
                return
            except ValueError:
                logger.opt(exception=True).warning(
                    "Retrieval scope write target missing for conversation {}",
                    conversation_id,
                )
                self.app_instance.notify(
                    "Couldn't save scope: this conversation no longer exists.",
                    severity="error",
                )
                return
            except Exception:
                logger.opt(exception=True).warning(
                    "Failed to write retrieval scope for conversation {}",
                    conversation_id,
                )
                self.app_instance.notify(
                    "Couldn't save scope: conversation metadata is corrupted.",
                    severity="error",
                )
                return
            after = await self._read_console_retrieval_scope(db, conversation_id)
            if scope is not None and after is None:
                self.app_instance.notify(
                    "Couldn't save scope: conversation metadata is corrupted.",
                    severity="error",
                )
                return
            self._console_retrieval_scope_cache[conversation_id] = after
        store.set_session_rag_scope(session.id, scope)
        await self._refresh_console_effective_scope_and_sync(session)

    def _apply_console_library_search_choice(
        self, result: ConsoleLibrarySearchResult | None
    ) -> None:
        """Store one-shot search values and optionally run retrieval."""
        if result is None:
            return
        self._set_library_rag_source_scope(result.source_types)
        self._set_library_rag_query(sanitize_console_library_rag_query(result.query))
        if result.run:
            self._run_library_rag_action()

    def open_library_search(self) -> None:
        """Open one-shot Library search from the current Console state."""
        prefill = self._composer_draft()
        self._push_screen(
            ConsoleLibrarySearchModal(
                query=prefill if prefill is not None else self._library_rag_query(),
                source_types=self._library_rag_source_scope(),
                item_scope_summary=console_retrieval_scope_label(
                    self._build_console_retrieval_scope_state()
                ),
            ),
            callback=self._apply_console_library_search_choice,
        )

    def _apply_console_rag_settings_choice(
        self, result: ConsoleLibrarySearchResult | None
    ) -> None:
        """Compatibility entry point for callers using the former name."""
        self._apply_console_library_search_choice(result)

    def _console_rag_source_status(
        self,
        pending_launch: ConsoleLiveWorkLaunch | None,
        sent_source_count: int | None = None,
    ) -> str:
        """Return the Inspector's staged/sent Library source status."""
        if pending_launch is None:
            if sent_source_count:
                count = int(sent_source_count)
                noun = "source" if count == 1 else "sources"
                return f"sent with the last message · {count} {noun}"
            return "not staged"
        if source_mentions_rag(pending_launch.source):
            launch_status = str(pending_launch.status or "").strip().lower()
            if launch_status in {"blocked", "failed", "unavailable"}:
                return "unavailable"
            if launch_status == "empty":
                return "no results"
            if launch_status == "searching":
                return "retrieving from Library Search/RAG"
            if launch_has_rag_source_payload(pending_launch):
                return "staged from Library Search/RAG"
            return "missing source"
        return "not requested"

    def _active_console_dictionary_scope_ids(
        self,
    ) -> tuple[str | None, int | None]:
        """Return the active native conversation and character scope IDs."""
        return self._current_conversation_id(), None

    async def refresh_active_dictionaries_summary(self) -> None:
        """Refresh the DB-backed dictionary summary cache off the UI loop."""
        conversation_id, character_id = self._active_console_dictionary_scope_ids()
        service = self._dictionary_scope_service()
        if service is None or (conversation_id is None and character_id is None):
            self._active_dictionaries_summary = {"dictionaries": []}
        else:
            try:
                summary = await asyncio.to_thread(
                    _run_dictionary_summary_off_thread,
                    service,
                    conversation_id,
                    character_id,
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not summarize active chat dictionaries for the Console "
                    "inspector."
                )
                summary = {"dictionaries": []}
            self._active_dictionaries_summary = (
                summary if isinstance(summary, dict) else {"dictionaries": []}
            )
        self._request_control_bar_sync()

    async def _refresh_active_dictionaries_summary_if_scope_changed(self) -> None:
        """Refresh dictionary summary only when its native scope changes."""
        scope_ids = self._active_console_dictionary_scope_ids()
        if scope_ids == self._last_console_dictionary_scope_ids:
            return
        self._last_console_dictionary_scope_ids = scope_ids
        await self.refresh_active_dictionaries_summary()

    def _console_dictionary_inspector_rows(self) -> tuple[ConsoleDisplayRow, ...]:
        """Project cached dictionary data into inspector rows without I/O."""
        conversation_id, character_id = self._active_console_dictionary_scope_ids()
        if conversation_id is None and character_id is None:
            return (ConsoleDisplayRow("No active chat", ""),)
        dictionaries = (self._active_dictionaries_summary or {}).get(
            "dictionaries"
        ) or []
        if not dictionaries:
            return (ConsoleDisplayRow("No dictionaries in play", ""),)
        rows = []
        for entry in dictionaries:
            if not isinstance(entry, dict):
                continue
            value = (
                "from conversation"
                if entry.get("source") == "conversation"
                else "from character"
            )
            if entry.get("shadowed"):
                value += " (shadowed)"
            if not entry.get("enabled", True):
                value += " (disabled)"
            rows.append(ConsoleDisplayRow(str(entry.get("name") or "Unnamed"), value))
        return tuple(rows)

    def _active_console_world_book_scope_ids(self) -> tuple[str | None]:
        """Return the active native conversation's world-book scope."""
        return (self._current_conversation_id(),)

    async def refresh_active_world_books_summary(self) -> None:
        """Refresh the DB-backed world-book summary cache off the UI loop."""
        conversation_id = self._current_conversation_id()
        db = getattr(self.app_instance, "chachanotes_db", None)
        if not conversation_id or db is None:
            self._active_world_books_summary = {"world_books": []}
            self._request_control_bar_sync()
            return
        try:
            from ...Character_Chat.world_info_resolver import (
                summarize_active_world_books,
            )

            summary = await asyncio.to_thread(
                summarize_active_world_books, db, conversation_id, None
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Could not summarize active world books for the Console inspector."
            )
            summary = {"world_books": []}
        self._active_world_books_summary = (
            summary if isinstance(summary, dict) else {"world_books": []}
        )
        self._request_control_bar_sync()

    async def _refresh_active_world_books_summary_if_scope_changed(self) -> None:
        """Refresh world-book summary only when its native scope changes."""
        scope_ids = self._active_console_world_book_scope_ids()
        if scope_ids == self._last_console_world_book_scope_ids:
            return
        self._last_console_world_book_scope_ids = scope_ids
        await self.refresh_active_world_books_summary()

    def _console_world_book_inspector_rows(self) -> tuple[ConsoleDisplayRow, ...]:
        """Project cached world-book data into inspector rows without I/O."""
        if self._current_conversation_id() is None:
            return (ConsoleDisplayRow("No active chat", ""),)
        books = (self._active_world_books_summary or {}).get("world_books") or []
        if not books:
            return (ConsoleDisplayRow("No world books in play", ""),)
        rows = []
        for entry in books:
            if not isinstance(entry, dict):
                continue
            count = entry.get("entry_count")
            value = f"{count} entries" if isinstance(count, int) else "0 entries"
            if not entry.get("enabled", True):
                value += " (disabled)"
            rows.append(ConsoleDisplayRow(str(entry.get("name") or "Unnamed"), value))
        return tuple(rows)

    def _console_dictionary_inspector_actions(
        self,
    ) -> tuple[ConsoleInspectorAction, ...]:
        """Build dictionary attach/detach actions from cached state."""
        conversation_id, _ = self._active_console_dictionary_scope_ids()
        dictionaries = (self._active_dictionaries_summary or {}).get(
            "dictionaries"
        ) or []
        has_attached = any(
            isinstance(entry, dict) and entry.get("source") == "conversation"
            for entry in dictionaries
        )
        return (
            ConsoleInspectorAction(
                "console-inspector-dictionaries-attach",
                "Attach dictionary…",
                enabled=bool(conversation_id),
                disabled_reason="Start or load a conversation first",
            ),
            ConsoleInspectorAction(
                "console-inspector-dictionaries-detach",
                "Detach dictionary…",
                enabled=has_attached,
            ),
        )

    def _console_world_book_inspector_actions(
        self,
    ) -> tuple[ConsoleInspectorAction, ...]:
        """Build world-book attach/detach actions from cached state."""
        conversation_id = self._current_conversation_id()
        has_attached = bool((self._active_world_books_summary or {}).get("world_books"))
        return (
            ConsoleInspectorAction(
                "console-inspector-worldbooks-attach",
                "Attach world book…",
                enabled=bool(conversation_id),
                disabled_reason="Start or load a conversation first",
            ),
            ConsoleInspectorAction(
                "console-inspector-worldbooks-detach",
                "Detach world book…",
                enabled=has_attached,
            ),
        )

    def _console_library_rag_scope_label(self) -> str:
        """Return the readiness card's normalized Library source summary."""
        return library_rag_source_scope_summary(
            self._library_rag_source_scope(), prefix=CONSOLE_RAG_SOURCE_SUMMARY_PREFIX
        )

    def _stage_console_library_rag_launch(
        self,
        launch: ConsoleLiveWorkLaunch,
        *,
        allow_recompose: bool = True,
    ) -> None:
        """Stage a launch and refresh mounted pending-launch projections."""
        self._set_pending_launch(launch)
        self._set_evidence_sent_notice(None)
        if not self._sync_pending_launch_surfaces() and allow_recompose:
            self._refresh_screen()

    async def _resolve_console_library_rag_scope(
        self, request: LibraryRagSearchRequest
    ) -> tuple[LibraryRagSearchRequest, LibraryRagSearchOutcome | None]:
        """Apply the active effective item scope to a Library request."""
        effective_scope = await resolve_effective_scope_for_chat(self.app_instance)
        if effective_scope.state == "empty":
            return request, LibraryRagSearchOutcome(
                status="empty",
                recovery_state=scope_empty_recovery_state(effective_scope.cause),
            )
        if effective_scope.state == "scoped":
            return replace(request, scope=effective_scope), None
        return request, None

    async def _execute_console_library_rag_search(
        self, request: LibraryRagSearchRequest
    ) -> None:
        """Resolve scope, execute Library retrieval, and apply the outcome."""
        scoped_request, short_circuit = await self._resolve_console_library_rag_scope(
            request
        )
        outcome = short_circuit or await run_library_rag_search(
            self.app_instance, scoped_request
        )
        await self._apply_console_library_rag_search_outcome(scoped_request, outcome)

    async def _apply_console_library_rag_search_outcome(
        self,
        request: LibraryRagSearchRequest,
        outcome: Any,
    ) -> None:
        """Convert a Library result or recovery into staged live work."""
        if not self._is_mounted():
            return
        if outcome.results:
            result = outcome.results[0]
            launch_payload = build_library_rag_console_live_work_payload(
                result, query=request.query
            )
            launch_payload["evidence_bundle"] = build_library_rag_evidence_bundle(
                outcome.results, query=request.query
            ).to_payload()
            launch_payload["requested_top_k"] = request.top_k
            launch_payload["search_mode"] = request.mode
            self._stage_console_library_rag_launch(
                ConsoleLiveWorkLaunch.from_values(
                    source="Library Search/RAG",
                    title=result.title,
                    payload=launch_payload,
                    status="staged",
                    recovery=CONSOLE_LIBRARY_RAG_RECOVERY_COPY,
                    action_label="Review evidence in Console",
                )
            )
            return

        recovery_state = outcome.recovery_state
        recovery_copy = (
            recovery_state.visible_copy
            if recovery_state is not None
            else "Library Search/RAG did not return usable evidence."
        )
        self._set_pending_auto_open(True)
        self._stage_console_library_rag_launch(
            ConsoleLiveWorkLaunch.from_values(
                source="Library Search/RAG",
                title="Library Search/RAG retrieval",
                payload={
                    "query": request.query,
                    "source_scope": ", ".join(request.source_types),
                },
                status=outcome.status or "blocked",
                recovery=recovery_copy,
                action_label="Resolve Library search setup",
            )
        )

    def _console_chat_dictionary_applier(
        self, conversation_id: str | None, text: str
    ) -> str:
        """Bound applier handed to the native Console controller: apply the
        active CONVERSATION chat dictionaries to a send's text (never raises).

        Resolves the db lazily (at call time), so a controller built before the
        db is ready still works. Conversation-only: ``char_data`` is ``None``
        (native sessions carry no character card yet).
        """
        db = getattr(self.app_instance, "chachanotes_db", None)
        if db is None or not conversation_id or not isinstance(text, str):
            return text
        from ...Character_Chat import Chat_Dictionary_Lib as cdl

        return cdl.apply_active_chatdicts_to_text(
            db,
            conversation_id,
            None,
            text,
            max_tokens=_CHATDICT_MAX_TOKENS,
            strategy=_CHATDICT_STRATEGY,
        )

    def _console_world_info_applier(
        self, conversation_id: str | None, message_text: str, history: list
    ) -> str:
        """Bound applier handed to the native Console controller: inject the
        active CONVERSATION world-info into a send's text (never raises).

        Resolves the db lazily. Conversation-only: ``char_data`` is ``None``
        (native sessions carry no character card). Honors the same
        ``[character_chat] enable_world_info`` gate as the legacy send path
        (`Event_Handlers/Chat_Events/chat_events.py`).
        """
        db = getattr(self.app_instance, "chachanotes_db", None)
        if (
            db is None
            or not conversation_id
            or not isinstance(message_text, str)
            or not get_cli_setting("character_chat", "enable_world_info", True)
        ):
            return message_text
        from ...Character_Chat.world_info_resolver import apply_world_info_to_message

        return apply_world_info_to_message(
            db, conversation_id, None, message_text, history or []
        )

    async def _console_dictionary_attach_worker(self) -> None:
        """Pick and attach a chat dictionary to the active Console conversation.

        Mirrors P1f's ``_character_dictionary_attach_worker``
        (``UI/Screens/personas_screen.py``) structurally: every await is
        individually guarded so no exception escapes the worker boundary --
        an uncaught worker exception kills the whole app under
        ``run_worker(exit_on_error=True)``.
        """
        try:
            conversation_id = self._current_conversation_id()
            if not conversation_id:
                self.app_instance.notify(
                    "Start or load a conversation first.", severity="warning"
                )
                return
            db = getattr(self.app_instance, "chachanotes_db", None)
            try:
                rows = await asyncio.to_thread(
                    console_attachable_dictionaries, db, conversation_id
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not load dictionaries for the Console attach picker."
                )
                return
            if not rows:
                self.app_instance.notify(
                    "No more dictionaries to attach.", severity="information"
                )
                return
            try:
                picked = await self.app_instance.push_screen_wait(
                    DictionaryPicker(rows)
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not show the Console dictionary picker."
                )
                return
            if not picked:
                return
            await handle_console_dictionary_attach(
                self.app_instance, conversation_id, picked
            )
            # Always resync after an attempted attach (spec AC5: ConflictError
            # -> notify + refresh): on success the summary gains the dict; on a
            # ConflictError the DB changed under us and the cache must re-read
            # the current truth rather than stay stale until the next switch.
            await self.refresh_active_dictionaries_summary()
        finally:
            self._finish_dictionary_dialog()

    async def _console_worldbook_attach_worker(self) -> None:
        """Pick and attach a world book to the active Console conversation.

        Mirrors :meth:`_console_dictionary_attach_worker`: every await is
        individually guarded so no exception escapes the worker boundary --
        an uncaught worker exception kills the whole app under
        ``run_worker(exit_on_error=True)``.
        """
        try:
            conversation_id = self._current_conversation_id()
            if not conversation_id:
                self.app_instance.notify(
                    "Start or load a conversation first.", severity="warning"
                )
                return
            db = getattr(self.app_instance, "chachanotes_db", None)
            if db is None:
                return
            from ...Character_Chat.world_book_manager import WorldBookManager
            from ...Widgets.Persona_Widgets.world_book_picker import WorldBookPicker

            def _attachable() -> list[dict]:
                mgr = WorldBookManager(db)
                attached_ids = {
                    b.get("id")
                    for b in mgr.get_world_books_for_conversation(
                        str(conversation_id), enabled_only=False
                    )
                }
                return [
                    {"world_book_id": int(b.get("id")), "name": str(b.get("name"))}
                    for b in (mgr.list_world_books(include_disabled=False) or [])
                    if b.get("id") not in attached_ids
                ]

            try:
                rows = await asyncio.to_thread(_attachable)
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not load world books for the Console attach picker."
                )
                return
            if not rows:
                self.app_instance.notify(
                    "No more world books to attach.", severity="information"
                )
                return
            try:
                picked = await self.app_instance.push_screen_wait(WorldBookPicker(rows))
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not show the Console world-book picker."
                )
                return
            if not picked:
                return
            try:
                await asyncio.to_thread(
                    WorldBookManager(db).associate_world_book_with_conversation,
                    str(conversation_id),
                    int(picked),
                )
            except Exception as exc:
                logger.opt(exception=True).warning("Could not attach the world book.")
                self.app_instance.notify(f"Attach failed: {exc}", severity="error")
                return
            await self.refresh_active_world_books_summary()
        finally:
            self._finish_worldbook_dialog()

    async def _console_worldbook_detach_worker(self) -> None:
        """Pick and detach a world book from the active Console conversation.

        Analogous to :meth:`_console_worldbook_attach_worker`.
        """
        try:
            conversation_id = self._current_conversation_id()
            if not conversation_id:
                self.app_instance.notify(
                    "Start or load a conversation first.", severity="warning"
                )
                return
            db = getattr(self.app_instance, "chachanotes_db", None)
            if db is None:
                return
            from ...Character_Chat.world_book_manager import WorldBookManager
            from ...Widgets.Persona_Widgets.world_book_picker import WorldBookPicker

            def _attached() -> list[dict]:
                mgr = WorldBookManager(db)
                return [
                    {"world_book_id": int(b.get("id")), "name": str(b.get("name"))}
                    for b in mgr.get_world_books_for_conversation(
                        str(conversation_id), enabled_only=False
                    )
                ]

            try:
                rows = await asyncio.to_thread(_attached)
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not load world books for the Console detach picker."
                )
                return
            if not rows:
                self.app_instance.notify(
                    "No world books attached to this conversation.",
                    severity="information",
                )
                return
            try:
                picked = await self.app_instance.push_screen_wait(
                    WorldBookPicker(
                        rows, title="Detach world book", confirm_label="Detach"
                    )
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not show the Console world-book picker."
                )
                return
            if not picked:
                return
            try:
                await asyncio.to_thread(
                    WorldBookManager(db).disassociate_world_book_from_conversation,
                    str(conversation_id),
                    int(picked),
                )
            except Exception as exc:
                logger.opt(exception=True).warning("Could not detach the world book.")
                self.app_instance.notify(f"Detach failed: {exc}", severity="error")
                return
            await self.refresh_active_world_books_summary()
        finally:
            self._finish_worldbook_dialog()

    async def _console_dictionary_detach_worker(self) -> None:
        """Pick and detach a chat dictionary from the active Console conversation.

        Analogous to :meth:`_console_dictionary_attach_worker`, over
        ``console_attached_dictionaries``/``handle_console_dictionary_detach``.
        """
        try:
            conversation_id = self._current_conversation_id()
            if not conversation_id:
                self.app_instance.notify(
                    "Start or load a conversation first.", severity="warning"
                )
                return
            db = getattr(self.app_instance, "chachanotes_db", None)
            try:
                rows = await asyncio.to_thread(
                    console_attached_dictionaries, db, conversation_id
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not load dictionaries for the Console detach picker."
                )
                return
            if not rows:
                self.app_instance.notify(
                    "No dictionaries attached to this conversation.",
                    severity="information",
                )
                return
            try:
                picked = await self.app_instance.push_screen_wait(
                    DictionaryPicker(
                        rows, title="Detach dictionary", confirm_label="Detach"
                    )
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not show the Console dictionary picker."
                )
                return
            if not picked:
                return
            await handle_console_dictionary_detach(
                self.app_instance, conversation_id, picked
            )
            # Always resync after an attempted detach (spec AC5: ConflictError
            # -> notify + refresh); see _console_dictionary_attach_worker.
            await self.refresh_active_dictionaries_summary()
        finally:
            self._finish_dictionary_dialog()
