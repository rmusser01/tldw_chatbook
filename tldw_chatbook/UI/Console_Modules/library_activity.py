"""Controller for minimized Console Library activity capture and review."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from loguru import logger

from tldw_chatbook.Chat.console_library_activity_buffer import (
    LibraryActivityFlushResult,
)
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnExecutionContext
from tldw_chatbook.Chat.library_activity import (
    LibraryActivityEvent,
    LibraryActivityView,
)


class ConsoleLibraryActivityController:
    """Own activity capture bindings and the selected-turn UI projection.

    The controller owns no DOM. Every mutable dependency is a named late-bound
    callable so session switches and test replacements remain visible.
    ``app_instance`` is the justified snapshot exception: the app identity is
    stable for the controller's lifetime while its service attributes remain
    live reads during provider construction.

    Args:
        app_instance: Stable application identity providing live services.
        ensure_store: Late-bound Console store accessor.
        transcript: Late-bound transcript accessor.
        inspector_rail: Late-bound Inspector accessor.
        citation_counts: Late-bound assistant citation-count mapping.
        reveal_inspector: Callback that exposes the Inspector region.
        sync_native_ui: Async native Console synchronization callback.
        notify: User-notification callback.
        run_worker: Starts a coroutine as a screen worker. With one, a
            file-backed conversation's durable rows are read off the UI loop
            (TASK-33620.15.4); without one, or for an in-memory or custom
            store, the projection reads them synchronously as before.
        presentation_paused: True while the Console must not render (a
            maintenance or local storage pause); a landing read then waits
            for the resume's replayed sync.
        torn_down: True once the screen is closing; a read landing after it
            publishes nothing.
    """

    def __init__(
        self,
        *,
        app_instance: Any,
        ensure_store: Callable[[], Any],
        transcript: Callable[[], Any | None],
        inspector_rail: Callable[[], Any | None],
        citation_counts: Callable[[], Mapping[str, int]],
        reveal_inspector: Callable[[], None],
        sync_native_ui: Callable[[], Awaitable[None]],
        notify: Callable[..., None],
        run_worker: Callable[[Any], Any] | None = None,
        presentation_paused: Callable[[], bool] | None = None,
        torn_down: Callable[[], bool] | None = None,
    ) -> None:
        self.app_instance = app_instance
        self._ensure_store = ensure_store
        self._transcript = transcript
        self._inspector_rail = inspector_rail
        self._citation_counts = citation_counts
        self._reveal_inspector = reveal_inspector
        self._sync_native_ui = sync_native_ui
        self._notify = notify
        self.run_worker = run_worker
        self.presentation_paused = presentation_paused or (lambda: False)
        self.torn_down = torn_down or (lambda: False)
        #: (database, conversation id, activity revision) and the durable rows
        #: a worker read for it; and the key of the newest read requested.
        self._durable: tuple[tuple, tuple[Any, ...]] | None = None
        self._durable_requested: tuple | None = None
        self.counts: dict[str, int] = {}
        self.view = LibraryActivityView(selected_turn_id=None, actions=())
        self.flush_result = LibraryActivityFlushResult(
            "saved", saved_count=0, pending_count=0
        )
        self._projection_token: (
            tuple[str | None, int, tuple[str, ...], str | None, int] | None
        ) = None

    def capture_kwargs(self, turn_context: object) -> dict[str, object]:
        """Return provider capture bindings for a production turn context.

        Args:
            turn_context: Candidate immutable Console turn context.

        Returns:
            Provider keyword arguments, or an empty mapping for test/legacy
            contexts.

        Raises:
            RuntimeError: If the production context has no active user opener.
        """
        if not isinstance(turn_context, ConsoleTurnExecutionContext):
            return {}
        store = self._ensure_store()
        session_id = turn_context.session_id
        turn_id = store.current_library_activity_turn_id(session_id)

        def capture(event: LibraryActivityEvent) -> None:
            store.capture_library_activity(session_id, turn_id, event)

        return {
            "activity_attempt_id": str(turn_context.library_authority.attempt_id),
            "activity_sink": capture,
        }

    def build_provider(
        self, turn_context: ConsoleTurnExecutionContext | None
    ) -> Any | None:
        """Resolve the run-pinned Direct or RAG Library provider.

        Args:
            turn_context: Immutable production turn context, if available.

        Returns:
            Configured provider, or ``None`` without a turn context.

        Raises:
            RuntimeError: If production capture cannot bind an owning turn.
        """
        if turn_context is None:
            return None
        # One builder, shared with the runtime-owned factory. The runtime's
        # copy is what `ensure_chat_controller` actually binds, so a second
        # body here is a copy that silently rots (TASK-32892 P0-1).
        from tldw_chatbook.Chat.console_runtime import _library_provider_for_app

        return _library_provider_for_app(
            self.app_instance,
            turn_context,
            **self.capture_kwargs(turn_context),
        )

    def invalidate_projection(self) -> None:
        """Force the next synchronization to rebuild its store projection."""
        self._projection_token = None

    def selected_message_id(self) -> str | None:
        """Return the selected native message after owner normalization.

        Returns:
            Native owner message ID, or ``None`` without a selection.
        """
        transcript = self._transcript()
        if transcript is None:
            return None
        selected = transcript.selected_message_id
        if selected is None:
            return None
        return transcript.thinking_owner_message_id(selected) or selected

    def selected_citation_count(self) -> int:
        """Return the selected answer's cached citation count.

        Returns:
            Non-negative number of cited sources.
        """
        selected = self.selected_message_id()
        return max(0, self._citation_counts().get(selected or "", 0))

    def sync_projection(self) -> None:
        """Refresh the selected-turn projection when its stable token moves."""
        store = self._ensure_store()
        session_id = store.active_session_id
        selected = self.selected_message_id()
        if session_id is None:
            self.counts = {}
            self.view = LibraryActivityView(selected_turn_id=None, actions=())
            self.invalidate_projection()
            return
        revision = store.library_activity_revision(session_id)
        active_path = tuple(store.active_path_message_ids(session_id))
        citation_count = self._citation_counts().get(selected or "", 0)
        token = (session_id, revision, active_path, selected, citation_count)
        if token == self._projection_token:
            return
        source = self._worker_source(store, session_id)
        if source is None:
            snapshot = store.library_activity_snapshot(session_id, selected)
        else:
            key, landed = (*source, revision), self._durable
            if landed is not None and landed[0] == key:
                rows: tuple[Any, ...] = landed[1]
            else:
                self._request_durable_rows(key)
                if landed is not None and landed[0][:2] == source:
                    # Rows read before this revision may predate a flush that
                    # emptied the pending buffer: keep what is shown until
                    # the newer rows land (within one read).
                    return
                rows = ()
            snapshot = store.library_activity_snapshot(
                session_id, selected, durable_rows=rows
            )
        view, counts, flush_result = snapshot
        self.view = view
        self.counts = dict(counts)
        self.flush_result = flush_result
        self._projection_token = token
        rail = self._inspector_rail()
        if rail is not None:
            rail.sync_library_activity(
                view,
                citation_count=citation_count,
                flush_result=flush_result,
            )

    def _worker_source(self, store: Any, session_id: str) -> tuple | None:
        """Return the (database, conversation) a worker may read, or None.

        Only the stock persistence over a file-backed ChaChaNotes database:
        an in-memory database's connection is per thread, and a custom store
        keeps its own read. A session with no persisted conversation has no
        durable rows to read.
        """
        if self.run_worker is None:
            return None
        from tldw_chatbook.Chat.chat_persistence_service import (
            ChatPersistenceService,
        )
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

        persistence = getattr(store, "persistence", None)
        database = getattr(persistence, "db", None)
        if (
            type(persistence) is not ChatPersistenceService
            or type(database) is not CharactersRAGDB
            or database.is_memory_db
        ):
            return None
        session = next(
            (item for item in store.sessions() if item.id == session_id), None
        )
        conversation_id = getattr(session, "persisted_conversation_id", None)
        return None if conversation_id is None else (database, conversation_id)

    def _request_durable_rows(self, key: tuple) -> None:
        """Start one worker read for ``key`` unless it is already in flight."""
        if self._durable_requested == key:
            return
        self._durable_requested = key
        self.run_worker(self._read_durable_rows(key))

    async def _read_durable_rows(self, key: tuple) -> None:
        """Read ``key``'s durable rows on a worker thread, then render them.

        Only the newest requested read publishes, so a read overtaken by a
        newer revision or source never overwrites it; nor does one landing
        after teardown. A local storage pause neither renders nor caches
        (the projection defers through one, and the next projection reads
        again); any other failure degrades as the synchronous read did, to
        no durable rows and the same warning.
        """
        try:
            rows = await self._durable_read(key)
        finally:
            newest = self._durable_requested == key
            if newest:
                self._durable_requested = None
        if not newest or self.torn_down():
            return
        self.invalidate_projection()
        if rows is None:
            return
        self._durable = (key, rows)
        if self.presentation_paused():
            return  # The resume's replayed sync renders them.
        transcript = self._transcript()
        if transcript is not None:
            self.sync_transcript(transcript)
        else:
            self.sync_projection()

    @staticmethod
    async def _durable_read(key: tuple) -> tuple[Any, ...] | None:
        """Return the rows, () after a failed read, or None during a pause."""
        from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

        from .view_workers import owned_display_read

        database, conversation_id, _revision = key
        try:
            return tuple(
                await owned_display_read(
                    database, database.get_trajectory_rows, conversation_id
                )
            )
        except RecoveryRequired as error:
            if error.args == ("storage_locally_paused",):
                return None
            logger.warning("library_activity_projection_read_failed")
            return ()
        except Exception:  # noqa: BLE001 - payload-free review degradation
            logger.warning("library_activity_projection_read_failed")
            return ()

    def sync_transcript(self, transcript: Any) -> dict[str, int]:
        """Publish native-assistant footer counts and return the visible map.

        Args:
            transcript: Mounted Console transcript widget.

        Returns:
            Positive activity counts keyed by native assistant message ID.
        """
        self.sync_projection()
        visible = {
            message_id: count
            for message_id, count in self.counts.items()
            if type(count) is int and count > 0
        }
        transcript.set_library_activity_counts(visible)
        return visible

    def open_selected(self, button: Any) -> None:
        """Select an assistant's owner, reveal Inspector, and focus activity.

        Args:
            button: Transcript activity affordance carrying a native message ID.
        """
        message_id = getattr(button, "native_message_id", None)
        if type(message_id) is not str or not message_id:
            return
        transcript = self._transcript()
        if transcript is None:
            return
        transcript.select_message(message_id)
        self._reveal_inspector()
        self.sync_projection()
        rail = self._inspector_rail()
        if rail is not None:
            rail.request_library_activity_focus()

    async def retry(self) -> None:
        """Retry the retained store-owned activity batch once.

        Returns:
            None after synchronizing projection and notification state.
        """
        store = self._ensure_store()
        session_id = store.active_session_id
        if session_id is None:
            return
        result = await asyncio.to_thread(store.retry_library_activity, session_id)
        self.invalidate_projection()
        self.sync_projection()
        await self._sync_native_ui()
        if result.status == "saved":
            self._notify("Library activity saved.")
        elif result.warning:
            self._notify(result.warning, severity="warning")


__all__ = ["ConsoleLibraryActivityController"]
