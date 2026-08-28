"""Saved-conversations controller for the Personas workbench screen.

Owns the conversations feature block extracted from ``PersonasScreen``:
listing a character's saved conversations, opening one in the read-only
center view, and the Send-to-Console-draft / Open-in-Library actions. The
screen keeps the compose chrome, ``_show_center``, ``_stage_handoff``, and
the thin ``@on`` handlers that delegate here, mirroring the
``CCPCharacterHandler`` pattern (a class holding a reference to its screen).
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Any

from loguru import logger
from textual.css.query import QueryError
from textual.widgets import Button

from ...Character_Chat.Character_Chat_Lib import (
    retrieve_conversation_messages_for_ui,
)
from ...Constants import (
    CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID,
    LIBRARY_MODE_CONVERSATIONS,
    LIBRARY_NAV_CONTEXT_CONVERSATION_ID,
    LIBRARY_NAV_CONTEXT_MODE,
    PERSONAS_CONVERSATIONS_PAGE_SIZE,
    TAB_CHAT,
    TAB_LIBRARY,
)
from ...Widgets.Persona_Widgets.personas_conversation_transcript_widget import (
    PersonasConversationTranscriptWidget,
)
from ...Widgets.Persona_Widgets.personas_inspector_pane import PersonasInspectorPane
from ..Navigation.main_navigation import NavigateToScreen

if TYPE_CHECKING:
    from ..Screens.personas_screen import PersonasScreen


logger = logger.bind(module="PersonasConversationsController")

#: The read-only transcript view (PersonasConversationTranscriptWidget's own id).
_CONVERSATION_VIEW_ID = "#personas-conversation-transcript-view"

#: Cap on the transcript text staged into a Console handoff body.
_HANDOFF_TRANSCRIPT_CHAR_LIMIT = 6000

#: One extra fetched row is the pagination sentinel.
_CONVERSATIONS_FETCH_LIMIT = PERSONAS_CONVERSATIONS_PAGE_SIZE + 1
#: Duplicate-only seek continuations allowed before yielding to another user action.
_CONVERSATIONS_MAX_AUTO_HOPS = 4


class PersonasConversationsController:
    """Handles the saved-conversations region for ``PersonasScreen``."""

    def __init__(self, screen: "PersonasScreen") -> None:
        self.screen = screen
        # Conversations listed for the selected character (id -> title) and
        # the conversation currently open in the read-only center view.
        self._conversation_rows: dict[str, str] = {}
        self._open_character_id: str | None = None
        self._open_conversation_id: str | None = None
        self._open_conversation_title: str = ""
        self._open_conversation_transcript: str = ""
        self._open_conversation_truncated: bool = False
        # Id of the conversation whose transcript actually finished loading;
        # set only by show_conversation_view so the Send-to-Console-draft
        # handler can tell an in-flight load from a completed one.
        self._loaded_conversation_id: str | None = None
        self._failed_conversation_id: str | None = None
        self._preview_attempt: object | None = None
        self._resume_in_flight_attempts: dict[str, object] = {}
        self._list_character_id: str | None = None
        self._loaded_conversation_ids: set[str] = set()
        self._next_conversation_cursor: tuple[Any, str] | None = None
        self._has_more_conversations = False
        self._conversation_list_phase: str | None = None
        self._conversation_list_attempt: object | None = None
        self._conversation_attempt_boundaries: list[
            tuple[Any, str] | None
        ] = []

    def reset(self) -> None:
        try:
            self.screen.query_one(
                PersonasInspectorPane
            ).invalidate_conversation_render()
        except QueryError:
            pass
        self._reset_conversation_browse()
        self.close_conversation_preview()
        self._resume_in_flight_attempts = {}

    # ===== Listing =====

    async def load_conversations(self, character_id: str) -> None:
        """Reset browsing for ``character_id`` and schedule its first page."""
        self._reset_conversation_browse(str(character_id))
        attempt = self._claim_conversation_page(initial=True)
        if attempt is None:
            return
        rendered = False
        try:
            rendered = await self.screen.query_one(
                PersonasInspectorPane
            ).show_conversations_loading(attempt)
        except Exception:
            logger.opt(exception=True).warning(
                "Could not render the conversations loading state."
            )
        if rendered and self._owns_conversation_page(
            str(character_id), None, attempt
        ):
            self._schedule_conversation_page(initial=True, attempt=attempt)
        elif not rendered:
            await self._recover_owned_conversation_render_failure(
                str(character_id), None, True, attempt
            )

    async def request_older_conversations(self) -> None:
        """Handle the actionable Load/Retry tail without starting duplicates."""
        if self._conversation_list_attempt is not None:
            return
        phase = self._conversation_list_phase
        if phase == "initial-retry":
            initial = True
        elif phase == "append-retry" or (
            phase == "ready" and self._has_more_conversations
        ):
            initial = False
        else:
            return

        attempt = self._claim_conversation_page(initial=initial)
        if attempt is None:
            return
        character_id = self._list_character_id
        cursor = self._next_conversation_cursor
        if character_id is None:
            return
        rendered = False
        try:
            inspector = self.screen.query_one(PersonasInspectorPane)
            if initial:
                rendered = await inspector.show_conversations_loading(attempt)
            else:
                rendered = await inspector.show_older_conversations_loading(attempt)
        except Exception:
            logger.opt(exception=True).warning(
                "Could not render the conversations loading state."
            )
        if rendered and self._owns_conversation_page(
            character_id, cursor, attempt
        ):
            self._schedule_conversation_page(initial=initial, attempt=attempt)
        elif not rendered:
            await self._recover_owned_conversation_render_failure(
                character_id, cursor, initial, attempt
            )

    def _reset_conversation_browse(self, character_id: str | None = None) -> None:
        """Invalidate an active list read and clear memory-only browse state."""
        self._conversation_list_attempt = None
        self._list_character_id = character_id
        self._conversation_rows = {}
        self._loaded_conversation_ids = set()
        self._next_conversation_cursor = None
        self._has_more_conversations = False
        self._conversation_list_phase = None
        self._conversation_attempt_boundaries = []

    def _claim_conversation_page(self, *, initial: bool) -> object | None:
        character_id = self._list_character_id
        if character_id is None or self._conversation_list_attempt is not None:
            return None
        attempt = object()
        self._conversation_list_attempt = attempt
        self._conversation_list_phase = (
            "initial-loading" if initial else "append-loading"
        )
        self._conversation_attempt_boundaries = [self._next_conversation_cursor]
        return attempt

    def _schedule_conversation_page(self, *, initial: bool, attempt: object) -> None:
        character_id = self._list_character_id
        if character_id is None or self._conversation_list_attempt is not attempt:
            return
        cursor = self._next_conversation_cursor
        self.screen.run_worker(
            partial(
                self._load_conversations_sync,
                character_id,
                cursor,
                initial,
                attempt,
            ),
            thread=True,
            exclusive=True,
            group="personas-conversations",
        )

    def _load_conversations_sync(
        self,
        character_id: str,
        cursor: tuple[Any, str] | None,
        initial: bool,
        attempt: object,
    ) -> None:
        """Fetch one saved-conversation page off the UI thread."""
        try:
            db = self.screen._character_db()
            if cursor is None:
                records = db.get_conversations_for_character(
                    int(character_id), limit=_CONVERSATIONS_FETCH_LIMIT
                )
            else:
                records = db.get_conversations_for_character(
                    int(character_id),
                    limit=_CONVERSATIONS_FETCH_LIMIT,
                    before_last_modified=cursor[0],
                    before_id=cursor[1],
                )
            records = tuple(records or ())
        except Exception:
            logger.opt(exception=True).warning(
                f"Could not list conversations for character {character_id}.",
            )
            self.screen.app.call_from_thread(
                self.apply_conversation_page_failure,
                character_id,
                cursor,
                initial,
                attempt,
            )
            return
        self.screen.app.call_from_thread(
            self.apply_conversation_page,
            character_id,
            cursor,
            initial,
            attempt,
            records,
        )

    def _owns_conversation_page(
        self,
        character_id: str,
        cursor: tuple[Any, str] | None,
        attempt: object,
    ) -> bool:
        """Return whether a list continuation still owns the visible context."""
        screen = self.screen
        return bool(
            screen.is_mounted
            and screen.state.active_mode == "characters"
            and screen.state.runtime_source == "local"
            and screen.state.selected_entity_kind == "character"
            and str(screen.state.selected_entity_id) == str(character_id)
            and self._list_character_id == str(character_id)
            and self._next_conversation_cursor == cursor
            and self._conversation_list_attempt is attempt
        )

    def _owns_conversation_retry(
        self,
        character_id: str,
        cursor: tuple[Any, str] | None,
        initial: bool,
    ) -> bool:
        """Return whether a released attempt still owns its retry context."""
        screen = self.screen
        return bool(
            screen.is_mounted
            and screen.state.active_mode == "characters"
            and screen.state.runtime_source == "local"
            and screen.state.selected_entity_kind == "character"
            and str(screen.state.selected_entity_id) == str(character_id)
            and self._list_character_id == str(character_id)
            and self._next_conversation_cursor == cursor
            and self._conversation_list_attempt is None
            and self._conversation_list_phase
            == ("initial-retry" if initial else "append-retry")
        )

    async def _recover_owned_conversation_render_failure(
        self,
        character_id: str,
        cursor: tuple[Any, str] | None,
        initial: bool,
        attempt: object,
        *,
        restore_append_rows: bool = False,
    ) -> None:
        """Release one owned attempt and make a bounded retry presentation."""
        if not self._owns_conversation_page(character_id, cursor, attempt):
            return
        self._conversation_list_attempt = None
        self._conversation_list_phase = "initial-retry" if initial else "append-retry"
        preserved_rows = (
            tuple(self._conversation_rows.items())
            if restore_append_rows and not initial
            else None
        )
        try:
            rendered = await self.screen.query_one(
                PersonasInspectorPane
            ).show_conversations_failure(
                initial=initial,
                render_attempt=attempt,
                preserved_rows=preserved_rows,
            )
        except Exception:
            rendered = False
            logger.bind(
                character_id=str(character_id),
                cursor=cursor,
                phase=self._conversation_list_phase,
                operation="render-owned-retry",
            ).opt(exception=True).warning(
                "Could not render the conversations retry state."
            )
        if rendered or not self._owns_conversation_retry(
            character_id, cursor, initial
        ):
            return
        try:
            rendered = await self.screen.query_one(
                PersonasInspectorPane
            ).show_conversations_failure(
                initial=initial,
                preserved_rows=preserved_rows,
            )
        except Exception:
            logger.bind(
                character_id=str(character_id),
                cursor=cursor,
                phase=self._conversation_list_phase,
                operation="render-fallback-retry",
            ).opt(exception=True).error(
                "Conversation retry presentation failed after bounded fallback."
            )
            return
        if not rendered and self._owns_conversation_retry(
            character_id, cursor, initial
        ):
            logger.bind(
                character_id=str(character_id),
                cursor=cursor,
                phase=self._conversation_list_phase,
                operation="render-fallback-retry",
            ).error(
                "Conversation retry presentation was rejected after bounded fallback."
            )

    async def apply_conversation_page_failure(
        self,
        character_id: str,
        cursor: tuple[Any, str] | None,
        initial: bool,
        attempt: object,
    ) -> None:
        """Render a retry tail when the exact failed attempt still owns it."""
        await self._recover_owned_conversation_render_failure(
            character_id, cursor, initial, attempt
        )

    async def apply_conversation_page(
        self,
        character_id: str,
        cursor: tuple[Any, str] | None,
        initial: bool,
        attempt: object,
        records: tuple[object, ...],
    ) -> None:
        """Validate, present, and then commit one still-owned page."""
        if not self._owns_conversation_page(character_id, cursor, attempt):
            return

        durable_page: list[tuple[str, str, Any]] = []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            raw_id = record.get("id")
            last_modified = record.get("last_modified")
            if raw_id is None or last_modified is None:
                continue
            conversation_id = str(raw_id).strip()
            if not conversation_id:
                continue
            durable_page.append(
                (
                    conversation_id,
                    str(record.get("title") or "Untitled conversation"),
                    last_modified,
                )
            )
            if len(durable_page) == PERSONAS_CONVERSATIONS_PAGE_SIZE:
                break

        raw_cursor = (
            (durable_page[-1][2], durable_page[-1][0])
            if durable_page
            else None
        )
        accepted: list[tuple[str, str, Any]] = []
        page_ids: set[str] = set()
        for conversation_id, title, last_modified in durable_page:
            if (
                conversation_id in self._loaded_conversation_ids
                or conversation_id in page_ids
            ):
                continue
            page_ids.add(conversation_id)
            accepted.append((conversation_id, title, last_modified))

        has_more = len(records) > PERSONAS_CONVERSATIONS_PAGE_SIZE
        if not accepted and has_more:
            advances = raw_cursor is not None and not any(
                raw_cursor == boundary
                for boundary in self._conversation_attempt_boundaries
            )
            auto_hops = len(self._conversation_attempt_boundaries) - 1
            if advances and auto_hops < _CONVERSATIONS_MAX_AUTO_HOPS:
                self._next_conversation_cursor = raw_cursor
                self._conversation_attempt_boundaries.append(raw_cursor)
                self._schedule_conversation_page(initial=initial, attempt=attempt)
                return
            if not advances:
                logger.warning(
                    f"Conversation page for character {character_id} contained "
                    "no new rows and no advancing durable boundary; treating it as "
                    "exhausted."
                )
                has_more = False

        rows = tuple((conversation_id, title) for conversation_id, title, _ in accepted)
        if initial:
            proposed_rows = dict(rows)
            proposed_ids = {
                conversation_id for conversation_id, _ in rows
            }
            proposed_cursor = raw_cursor
        else:
            proposed_rows = dict(self._conversation_rows)
            proposed_rows.update(rows)
            proposed_ids = set(self._loaded_conversation_ids)
            proposed_ids.update(conversation_id for conversation_id, _ in rows)
            proposed_cursor = raw_cursor or self._next_conversation_cursor

        rendered = False
        try:
            inspector = self.screen.query_one(PersonasInspectorPane)
            if initial:
                rendered = await inspector.show_conversations(
                    rows,
                    empty_copy="No saved conversations.",
                    has_more=has_more,
                    render_attempt=attempt,
                )
            else:
                rendered = await inspector.append_conversations(
                    rows, has_more=has_more, render_attempt=attempt
                )
        except Exception:
            logger.opt(exception=True).warning(
                "Could not render the conversations panel."
            )
        if not rendered:
            await self._recover_owned_conversation_render_failure(
                character_id,
                cursor,
                initial,
                attempt,
                restore_append_rows=not initial,
            )
            return
        if not self._owns_conversation_page(character_id, cursor, attempt):
            return
        self._conversation_rows = proposed_rows
        self._loaded_conversation_ids = proposed_ids
        self._next_conversation_cursor = proposed_cursor
        self._has_more_conversations = has_more
        self._conversation_list_phase = "ready"
        self._conversation_list_attempt = None

    # ===== Read-only view =====

    async def open_conversation(self, conversation_id: str) -> None:
        """Row-selected continuation: open the conversation read-only.

        The transcript view opens IMMEDIATELY with a loading placeholder so
        the click has instant feedback; the message worker's continuation
        replaces it with the content (or a newer selection supersedes it).
        """
        screen = self.screen
        preview_attempt = object()
        self._preview_attempt = preview_attempt
        self._open_character_id = (
            str(screen.state.selected_entity_id).strip()
            if screen.state.selected_entity_id is not None
            else None
        )
        screen._edit_mode = "view"
        self._open_conversation_id = conversation_id
        self._open_conversation_title = self._conversation_rows.get(
            conversation_id, "Untitled conversation"
        )
        self._open_conversation_transcript = ""
        self._open_conversation_truncated = False
        self._loaded_conversation_id = None
        self._failed_conversation_id = None
        target_id = str(conversation_id).strip()
        self._set_resume_button_busy(target_id in self._resume_in_flight_attempts)
        try:
            view = screen.query_one(PersonasConversationTranscriptWidget)
            view.set_title(self._open_conversation_title or "Conversation")
            rendered = await view.show_loading(preview_attempt)
            if not rendered or not self._owns_preview(conversation_id, preview_attempt):
                return
            screen._show_center(_CONVERSATION_VIEW_ID)
            # Sync header title and console actions for the open transcript.
            screen._sync_title_and_console_actions()
        except QueryError:
            logger.warning("Conversation transcript widget is not mounted.")
        if not self._owns_preview(conversation_id, preview_attempt):
            return
        self.load_conversation_messages(
            conversation_id,
            screen.state.selected_entity_name or "Character",
            preview_attempt,
        )

    def load_conversation_messages(
        self,
        conversation_id: str,
        character_name: str,
        preview_attempt: object,
    ) -> None:
        """Schedule the transcript fetch on the screen's worker pool.

        Args:
            conversation_id: Durable conversation selected for preview.
            character_name: Historical speaker label used while shaping messages.
            preview_attempt: Exact ownership token for this preview load.
        """
        self.screen.run_worker(
            partial(
                self._load_conversation_messages_sync,
                conversation_id,
                character_name,
                preview_attempt,
            ),
            thread=True,
            exclusive=True,
            group="personas-conversation-view",
        )

    def _load_conversation_messages_sync(
        self,
        conversation_id: str,
        character_name: str,
        preview_attempt: object,
    ) -> None:
        """Fetch and shape the conversation's messages off the UI thread."""
        try:
            history = (
                retrieve_conversation_messages_for_ui(
                    self.screen._character_db(),
                    conversation_id,
                    character_name,
                    None,
                    limit=200,
                )
                or []
            )
        except Exception:
            logger.opt(exception=True).warning(
                f"Could not load messages for conversation {conversation_id}.",
            )
            self.screen.app.call_from_thread(
                self.show_conversation_error, conversation_id, preview_attempt
            )
            return
        messages: list[dict] = []
        transcript_lines: list[str] = []
        for user_message, bot_message in history:
            if user_message:
                messages.append({"role": "user", "content": user_message})
                transcript_lines.append(f"User: {user_message}")
            if bot_message:
                messages.append({"role": "assistant", "content": bot_message})
                transcript_lines.append(f"{character_name}: {bot_message}")
        full_transcript = "\n".join(transcript_lines)
        truncated = len(full_transcript) > _HANDOFF_TRANSCRIPT_CHAR_LIMIT
        transcript = full_transcript[:_HANDOFF_TRANSCRIPT_CHAR_LIMIT]
        self.screen.app.call_from_thread(
            self.show_conversation_view,
            conversation_id,
            messages,
            transcript,
            truncated,
            preview_attempt,
            # The on-screen transcript names speakers ("You"/the character),
            # matching the staged handoff body built above.
            {"user": "You", "assistant": character_name},
        )

    async def show_conversation_view(
        self,
        conversation_id: str,
        messages: list[dict],
        transcript: str,
        truncated: bool,
        preview_attempt: object,
        speaker_names: dict[str, str] | None = None,
    ) -> None:
        """Display the read-only transcript when this continuation still owns it.

        Args:
            conversation_id: Durable conversation being rendered.
            messages: Transcript messages shaped for the read-only widget.
            transcript: Bounded plain-text transcript for the draft handoff.
            truncated: Whether the draft handoff transcript was truncated.
            preview_attempt: Exact ownership token for this preview load.
            speaker_names: Optional role-to-display-name mapping.
        """
        if not self._owns_preview(conversation_id, preview_attempt):
            return
        screen = self.screen
        try:
            view = screen.query_one(PersonasConversationTranscriptWidget)
        except QueryError:
            logger.warning("Conversation transcript widget is not mounted.")
            return
        rendered = await view.load_messages(
            messages,
            speaker_names=speaker_names,
            render_attempt=preview_attempt,
        )
        if not rendered or not self._owns_preview(conversation_id, preview_attempt):
            return
        self._open_conversation_transcript = transcript
        self._open_conversation_truncated = truncated
        self._loaded_conversation_id = conversation_id
        screen._show_center(_CONVERSATION_VIEW_ID)
        # Sync header title and actions without moving focus away from the
        # conversations list the user is browsing.
        screen._sync_title_and_console_actions()

    async def show_conversation_error(
        self, conversation_id: str, preview_attempt: object
    ) -> None:
        """Display a recoverable error for the current preview only.

        Args:
            conversation_id: Durable conversation whose preview failed.
            preview_attempt: Exact ownership token for this preview load.
        """
        if not self._owns_preview(conversation_id, preview_attempt):
            return
        screen = self.screen
        try:
            view = screen.query_one(PersonasConversationTranscriptWidget)
        except QueryError:
            return
        rendered = await view.show_error(preview_attempt)
        if not rendered or not self._owns_preview(conversation_id, preview_attempt):
            return
        self._loaded_conversation_id = None
        self._failed_conversation_id = conversation_id
        self._open_conversation_transcript = ""
        self._open_conversation_truncated = False
        screen._show_center(_CONVERSATION_VIEW_ID)
        screen._sync_title_and_console_actions()

    def close_conversation_preview(self) -> None:
        """Invalidate the open preview so delayed continuations lose ownership."""
        preview_attempt = self._preview_attempt
        try:
            self.screen.query_one(
                PersonasConversationTranscriptWidget
            ).invalidate_render(preview_attempt)
        except QueryError:
            pass
        self._preview_attempt = None
        self._open_character_id = None
        self._open_conversation_id = None
        self._open_conversation_title = ""
        self._open_conversation_transcript = ""
        self._open_conversation_truncated = False
        self._loaded_conversation_id = None
        self._failed_conversation_id = None

    def _owns_preview(self, conversation_id: str, preview_attempt: object) -> bool:
        """Return whether an async continuation still owns the open preview."""
        screen = self.screen
        return bool(
            self._preview_attempt is preview_attempt
            and self._open_character_id is not None
            and self._open_character_id
            == (
                str(screen.state.selected_entity_id).strip()
                if screen.state.selected_entity_id is not None
                else None
            )
            and self._open_conversation_id == conversation_id
            and screen.is_mounted
            and screen.state.active_mode == "characters"
            and screen.state.selected_entity_kind == "character"
        )

    # ===== Conversation actions =====

    def continue_in_console(self) -> None:
        """Stage the open conversation's transcript into Console."""
        screen = self.screen
        conversation_id = self._open_conversation_id
        if not conversation_id:
            screen._notify(
                "Open a conversation before sending it to the Console draft.",
                "warning",
            )
            return
        if self._loaded_conversation_id != conversation_id:
            # The transcript worker has not delivered this conversation yet
            # (or a newer selection superseded the loaded one).
            if self._failed_conversation_id == conversation_id:
                screen._notify("Conversation preview couldn't load.", "warning")
            else:
                screen._notify("Conversation is still loading.", "warning")
            return
        character_name = screen.state.selected_entity_name or "Character"
        title = self._open_conversation_title or "Untitled conversation"
        staged = screen._stage_handoff(
            item_type="character-conversation",
            title=f"{character_name}: {title}",
            body=self._open_conversation_transcript or "",
            body_truncated=self._open_conversation_truncated,
            source_id=conversation_id,
            extra_metadata={"conversation_id": conversation_id},
        )
        if staged:
            screen._notify("Conversation staged in Console.", "information")

    def resume_in_console(self) -> None:
        """Navigate to Console with only the open saved-conversation ID."""
        target_id = str(self._open_conversation_id or "").strip()
        if not target_id or target_id not in self._conversation_rows:
            self.screen._notify(
                "This conversation is no longer available. Refresh conversations and "
                "try again.",
                "warning",
            )
            return
        if target_id in self._resume_in_flight_attempts:
            return
        attempt = object()
        self._resume_in_flight_attempts[target_id] = attempt
        self._set_resume_button_busy(True)
        self.screen.post_message(
            NavigateToScreen(
                TAB_CHAT,
                {CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID: target_id},
            )
        )
        self.screen.set_timer(
            1.0, partial(self._restore_resume_button, target_id, attempt)
        )

    def _set_resume_button_busy(self, busy: bool) -> None:
        """Paint the shared Resume button's local source-side state."""
        try:
            button = self.screen.query_one("#personas-conversation-resume", Button)
        except QueryError:
            return
        button.disabled = busy
        button.label = "Opening Console…" if busy else "Resume chat"

    def _restore_resume_button(self, target_id: str, attempt: object) -> None:
        """Release the exact target when navigation leaves Roleplay mounted."""
        screen = self.screen
        if (
            self._resume_in_flight_attempts.get(target_id) is not attempt
            or not screen.is_mounted
            or screen.app.screen is not screen
        ):
            return
        del self._resume_in_flight_attempts[target_id]
        if str(self._open_conversation_id or "").strip() == target_id:
            self._set_resume_button_busy(False)

    def open_in_library(self) -> None:
        """Route the open conversation to Library.

        Returns:
            None. Posts a navigation message when a conversation is open;
            otherwise warns the user and leaves the current screen in place.
        """
        conversation_id = str(self._open_conversation_id or "").strip()
        if not conversation_id:
            self.screen._notify(
                "Open a conversation before opening it in Library.", "warning"
            )
            return
        self.screen.post_message(
            NavigateToScreen(
                TAB_LIBRARY,
                {
                    LIBRARY_NAV_CONTEXT_MODE: LIBRARY_MODE_CONVERSATIONS,
                    LIBRARY_NAV_CONTEXT_CONVERSATION_ID: conversation_id,
                },
            )
        )
