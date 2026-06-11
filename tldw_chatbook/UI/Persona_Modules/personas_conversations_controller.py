"""Saved-conversations controller for the Personas workbench screen.

Owns the conversations feature block extracted from ``PersonasScreen``:
listing a character's saved conversations, opening one in the read-only
center view, and the Continue-in-Console / Open-in-Library actions. The
screen keeps the compose chrome, ``_show_center``, ``_stage_handoff``, and
the thin ``@on`` handlers that delegate here, mirroring the
``CCPCharacterHandler`` pattern (a class holding a reference to its screen).
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from loguru import logger
from textual.css.query import QueryError

from ...Character_Chat.Character_Chat_Lib import (
    list_character_conversations,
    retrieve_conversation_messages_for_ui,
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
        # set only by show_conversation_view so the Continue-in-Console
        # handler can tell an in-flight load from a completed one.
        self._loaded_conversation_id: str | None = None

    def reset(self) -> None:
        self._conversation_rows = {}
        self._open_conversation_id = None
        self._open_conversation_title = ""
        self._open_conversation_transcript = ""
        self._open_conversation_truncated = False
        self._loaded_conversation_id = None

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
            records = list_character_conversations(
                self.screen._character_db(), int(character_id), limit=20
            ) or []
        except Exception:
            logger.warning(
                f"Could not list conversations for character {character_id}.",
                exc_info=True,
            )
            records = []
        rows = tuple(
            (str(record.get("id")), str(record.get("title") or "Untitled conversation"))
            for record in records
            if record.get("id") is not None
        )
        self.screen.app.call_from_thread(self.apply_conversation_rows, character_id, rows)

    async def apply_conversation_rows(
        self, character_id: str, rows: tuple[tuple[str, str], ...]
    ) -> None:
        """UI-thread continuation: render rows unless the selection moved on."""
        screen = self.screen
        if not screen.is_mounted or screen.state.active_mode != "characters":
            return
        if (
            screen.state.selected_entity_kind != "character"
            or str(screen.state.selected_entity_id) != str(character_id)
        ):
            return
        self._conversation_rows = dict(rows)
        try:
            await screen.query_one(PersonasInspectorPane).show_conversations(rows)
        except Exception:
            logger.warning("Could not render the conversations panel.", exc_info=True)

    # ===== Read-only view =====

    async def open_conversation(self, conversation_id: str) -> None:
        """Row-selected continuation: open the conversation read-only."""
        screen = self.screen
        screen._edit_mode = "view"
        self._open_conversation_id = conversation_id
        self._open_conversation_title = self._conversation_rows.get(
            conversation_id, "Untitled conversation"
        )
        self._open_conversation_transcript = ""
        self._open_conversation_truncated = False
        self._loaded_conversation_id = None
        self.load_conversation_messages(
            conversation_id, screen.state.selected_entity_name or "Character"
        )

    def load_conversation_messages(self, conversation_id: str, character_name: str) -> None:
        """Schedule the transcript fetch on the screen's worker pool."""
        self.screen.run_worker(
            partial(self._load_conversation_messages_sync, conversation_id, character_name),
            thread=True,
            exclusive=True,
            group="personas-conversation-view",
        )

    def _load_conversation_messages_sync(
        self, conversation_id: str, character_name: str
    ) -> None:
        """Fetch and shape the conversation's messages off the UI thread."""
        try:
            history = retrieve_conversation_messages_for_ui(
                self.screen._character_db(), conversation_id, character_name, None, limit=200
            ) or []
        except Exception:
            logger.warning(
                f"Could not load messages for conversation {conversation_id}.",
                exc_info=True,
            )
            history = []
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
            self.show_conversation_view, conversation_id, messages, transcript, truncated
        )

    async def show_conversation_view(
        self, conversation_id: str, messages: list[dict], transcript: str, truncated: bool
    ) -> None:
        """UI-thread continuation: display the read-only transcript view."""
        screen = self.screen
        if not screen.is_mounted or screen.state.active_mode != "characters":
            return
        if (
            screen.state.selected_entity_kind != "character"
            or self._open_conversation_id != conversation_id
        ):
            # The selection or the requested conversation changed mid-flight.
            return
        self._open_conversation_transcript = transcript
        self._open_conversation_truncated = truncated
        self._loaded_conversation_id = conversation_id
        try:
            view = screen.query_one(PersonasConversationTranscriptWidget)
        except QueryError:
            logger.warning("Conversation transcript widget is not mounted.")
            return
        view.set_title(self._open_conversation_title or "Conversation")
        await view.load_messages(messages)
        screen._show_center(_CONVERSATION_VIEW_ID)

    # ===== Conversation actions =====

    def continue_in_console(self) -> None:
        """Stage the open conversation's transcript into Console."""
        screen = self.screen
        conversation_id = self._open_conversation_id
        if not conversation_id:
            screen._notify("Open a conversation before continuing in Console.", "warning")
            return
        if self._loaded_conversation_id != conversation_id:
            # The transcript worker has not delivered this conversation yet
            # (or a newer selection superseded the loaded one).
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

    def open_in_library(self) -> None:
        """Route to the Library conversations screen."""
        # "conversation" is the Library conversations route in screen_registry.
        self.screen.post_message(NavigateToScreen("conversation"))
