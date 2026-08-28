"""Controller for minimized Console Library activity capture and review."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from tldw_chatbook.Chat.console_library_activity_buffer import (
    LibraryActivityFlushResult,
)
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnExecutionContext
from tldw_chatbook.Chat.library_activity import LibraryActivityEvent, LibraryActivityView


class ConsoleLibraryActivityController:
    """Own activity capture bindings and the selected-turn UI projection.

    The controller owns no DOM. Every mutable dependency is a named late-bound
    callable so session switches and test replacements remain visible.
    ``app_instance`` is the justified snapshot exception: the app identity is
    stable for the controller's lifetime while its service attributes remain
    live reads during provider construction.
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
    ) -> None:
        self.app_instance = app_instance
        self._ensure_store = ensure_store
        self._transcript = transcript
        self._inspector_rail = inspector_rail
        self._citation_counts = citation_counts
        self._reveal_inspector = reveal_inspector
        self._sync_native_ui = sync_native_ui
        self._notify = notify
        self.counts: dict[str, int] = {}
        self.view = LibraryActivityView(selected_turn_id=None, actions=())
        self.flush_result = LibraryActivityFlushResult(
            "saved", saved_count=0, pending_count=0
        )
        self._projection_token: tuple[
            str | None, int, tuple[str, ...], str | None, int
        ] | None = None

    def capture_kwargs(self, turn_context: object) -> dict[str, object]:
        """Return provider capture bindings for a production turn context."""
        if not isinstance(turn_context, ConsoleTurnExecutionContext):
            return {}
        store = self._ensure_store()
        session_id = turn_context.session_id
        turn_id = store.current_library_activity_turn_id(session_id)

        def capture(event: LibraryActivityEvent) -> None:
            store.admit_library_activity(session_id, turn_id, event)
            store.flush_library_activity(session_id)

        return {
            "activity_attempt_id": str(turn_context.library_authority.attempt_id),
            "activity_sink": capture,
        }

    def build_provider(
        self, turn_context: ConsoleTurnExecutionContext | None
    ) -> Any | None:
        """Resolve the run-pinned Direct or RAG Library provider."""
        if turn_context is None:
            return None
        app = self.app_instance
        activity_kwargs = self.capture_kwargs(turn_context)
        if not turn_context.library_authority.direct_library_tools:
            from tldw_chatbook.Agents.library_rag_tool_provider import (
                LibraryRagToolProvider,
            )

            return LibraryRagToolProvider(
                getattr(app, "library_rag_search_service", None),
                **activity_kwargs,
            )
        from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
        from tldw_chatbook.Library.local_library_tool_service import (
            LocalLibraryToolService,
        )

        media_chunk_service = None
        media_reading_service = getattr(app, "local_media_reading_service", None)
        media_db = getattr(app, "media_db", None) or getattr(
            media_reading_service, "media_db", None
        )
        if media_db is not None or media_reading_service is not None:
            from tldw_chatbook.Chunking.chunking_interop_library import (
                get_chunking_service,
            )
            from tldw_chatbook.Library.local_media_chunk_tool_service import (
                LocalMediaChunkToolService,
            )

            media_chunk_service = LocalMediaChunkToolService(
                media_db,
                media_reading_service,
                template_interop=(
                    get_chunking_service(media_db) if media_db is not None else None
                ),
                policy_enforcer=getattr(app, "service_policy_enforcer", None),
            )
        service = LocalLibraryToolService(
            media_service=media_reading_service,
            notes_service=getattr(app, "notes_service", None),
            prompt_service=getattr(app, "local_prompt_service", None),
            skills_service=getattr(app, "local_skills_service", None),
            conversation_service=getattr(
                app, "local_chat_conversation_service", None
            ),
            collections_service=getattr(
                app, "local_library_collections_service", None
            ),
            media_chunk_service=media_chunk_service,
            notes_scope_service=getattr(app, "notes_scope_service", None),
            policy_enforcer=getattr(app, "service_policy_enforcer", None),
        )
        return LibraryToolProvider(service, **activity_kwargs)

    def invalidate_projection(self) -> None:
        """Force the next synchronization to rebuild its store projection."""
        self._projection_token = None

    def selected_message_id(self) -> str | None:
        """Return the selected native message after owner normalization."""
        transcript = self._transcript()
        if transcript is None:
            return None
        selected = transcript.selected_message_id
        if selected is None:
            return None
        return transcript.thinking_owner_message_id(selected) or selected

    def selected_citation_count(self) -> int:
        """Return the selected answer's cached citation count."""
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
        view, counts, flush_result = store.library_activity_snapshot(
            session_id, selected
        )
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

    def sync_transcript(self, transcript: Any) -> dict[str, int]:
        """Publish native-assistant footer counts and return the visible map."""
        self.sync_projection()
        visible = {
            message_id: count
            for message_id, count in self.counts.items()
            if type(count) is int and count > 0
        }
        transcript.set_library_activity_counts(visible)
        return visible

    def open_selected(self, button: Any) -> None:
        """Select an assistant's owner, reveal Inspector, and focus activity."""
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
        """Retry the retained store-owned activity batch once."""
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
