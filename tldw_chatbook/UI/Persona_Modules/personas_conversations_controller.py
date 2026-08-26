"""Saved-conversations controller for the Personas workbench screen.

Owns the conversations feature block extracted from ``PersonasScreen``:
listing a character's saved conversations, opening one in the read-only
center view, and the Send-to-Console-draft / Open-in-Library actions. The
screen keeps the compose chrome, ``_show_center``, ``_stage_handoff``, and
the thin ``@on`` handlers that delegate here, mirroring the
``CCPCharacterHandler`` pattern (a class holding a reference to its screen).
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from loguru import logger
from textual.css.query import QueryError
from textual.widgets import Button

from ...Character_Chat.Character_Chat_Lib import (
    list_character_conversations,
    retrieve_conversation_messages_for_ui,
)
from ...Constants import (
    CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID,
    LIBRARY_MODE_CONVERSATIONS,
    LIBRARY_NAV_CONTEXT_CONVERSATION_ID,
    LIBRARY_NAV_CONTEXT_MODE,
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


class PersonasConversationsController:
    """Handles the saved-conversations region for ``PersonasScreen``."""

    def __init__(self, screen: "PersonasScreen") -> None:
        self.screen = screen
        # Conversations listed for the selected character (id -> title) and
        # the conversation currently open in the read-only center view.
        self._conversation_rows: dict[str, str] = {}
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

    def reset(self) -> None:
        self._conversation_rows = {}
        self._open_conversation_id = None
        self._open_conversation_title = ""
        self._open_conversation_transcript = ""
        self._open_conversation_truncated = False
        self._loaded_conversation_id = None
        self._failed_conversation_id = None
        self._preview_attempt = None
        self._resume_in_flight_attempts = {}

    # ===== Listing =====

    def load_conversations(self, character_id: str) -> None:
        """Schedule the conversation listing on the screen's worker pool."""
        self.screen.run_worker(
            partial(self._load_conversations_sync, character_id),
            thread=True,
            exclusive=True,
            group="personas-conversations",
        )

    def _load_conversations_sync(self, character_id: str) -> None:
        """List the character's saved conversations off the UI thread."""
        try:
            records = (
                list_character_conversations(
                    self.screen._character_db(), int(character_id), limit=20
                )
                or []
            )
        except Exception:
            logger.opt(exception=True).warning(
                f"Could not list conversations for character {character_id}.",
            )
            records = []
        rows = tuple(
            (str(record.get("id")), str(record.get("title") or "Untitled conversation"))
            for record in records
            if record.get("id") is not None
        )
        self.screen.app.call_from_thread(
            self.apply_conversation_rows, character_id, rows
        )

    async def apply_conversation_rows(
        self, character_id: str, rows: tuple[tuple[str, str], ...]
    ) -> None:
        """UI-thread continuation: render rows unless the selection moved on."""
        screen = self.screen
        if not screen.is_mounted or screen.state.active_mode != "characters":
            return
        if screen.state.selected_entity_kind != "character" or str(
            screen.state.selected_entity_id
        ) != str(character_id):
            return
        self._conversation_rows = dict(rows)
        try:
            # An empty result (including a tolerated listing failure) shows
            # readable empty-state copy rather than a silently blank panel.
            await screen.query_one(PersonasInspectorPane).show_conversations(
                rows, empty_copy="No saved conversations."
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Could not render the conversations panel."
            )

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
            await view.show_loading()
            if not self._owns_preview(conversation_id, preview_attempt):
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
        """Schedule the transcript fetch on the screen's worker pool."""
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
        """UI-thread continuation: display the read-only transcript view."""
        if not self._owns_preview(conversation_id, preview_attempt):
            return
        screen = self.screen
        try:
            view = screen.query_one(PersonasConversationTranscriptWidget)
        except QueryError:
            logger.warning("Conversation transcript widget is not mounted.")
            return
        await view.load_messages(messages, speaker_names=speaker_names)
        if not self._owns_preview(conversation_id, preview_attempt):
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
        """Display a recoverable error for the current preview only."""
        if not self._owns_preview(conversation_id, preview_attempt):
            return
        screen = self.screen
        try:
            view = screen.query_one(PersonasConversationTranscriptWidget)
        except QueryError:
            return
        await view.show_error()
        if not self._owns_preview(conversation_id, preview_attempt):
            return
        self._loaded_conversation_id = None
        self._failed_conversation_id = conversation_id
        self._open_conversation_transcript = ""
        self._open_conversation_truncated = False
        screen._show_center(_CONVERSATION_VIEW_ID)
        screen._sync_title_and_console_actions()

    def close_conversation_preview(self) -> None:
        """Invalidate the open preview so delayed continuations lose ownership."""
        self._preview_attempt = None
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
