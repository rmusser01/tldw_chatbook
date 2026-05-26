"""Console-native chat session surface."""

from __future__ import annotations

import asyncio
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Static

from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


CONSOLE_CLOSE_TAB_BUTTON_WIDTH = 3
CONSOLE_CLOSE_TAB_BUTTON_HEIGHT = 1
CONSOLE_NEW_TAB_BUTTON_WIDTH = 3
CONSOLE_NEW_TAB_BUTTON_HEIGHT = 1
CONSOLE_RENAME_TAB_BUTTON_WIDTH = 8
CONSOLE_SESSION_RENAME_INPUT_WIDTH = 24
CONSOLE_SESSION_TAB_DISPLAY_CHARS = 19
CONSOLE_SESSION_TAB_WIDTH = 21
CONSOLE_TRANSCRIPT_TITLE = "Transcript / Event Stream"


class ConsoleSessionSurface(Vertical):
    """Host Console transcript/event stream sessions without legacy chat chrome."""

    def __init__(self, app_instance: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self._session_sync_lock = asyncio.Lock()
        self._renaming_session_id: str | None = None

    def compose(self) -> ComposeResult:
        title = Static(
            CONSOLE_TRANSCRIPT_TITLE,
            id="console-transcript-title",
            classes="destination-section console-transcript-title",
        )
        title.styles.height = 1
        title.styles.min_height = 1
        yield title

        tab_strip = Horizontal(
            id="console-native-tab-strip",
            classes="console-session-tab-strip",
        )
        tab_strip.styles.height = 1
        tab_strip.styles.min_height = 1
        tab_strip.styles.max_height = 1
        with tab_strip:
            yield self._build_new_tab_button()
        yield ChatTaskCards(id="console-task-surface")
        yield ConsoleTranscript(id="console-native-transcript")

    def _build_new_tab_button(self) -> Button:
        """Return the compact symbolic Console new-session control."""
        button = Button("+", id="console-new-chat-tab", compact=True)
        button.tooltip = "New Console tab"
        button.styles.width = CONSOLE_NEW_TAB_BUTTON_WIDTH
        button.styles.min_width = CONSOLE_NEW_TAB_BUTTON_WIDTH
        button.styles.max_width = CONSOLE_NEW_TAB_BUTTON_WIDTH
        button.styles.height = CONSOLE_NEW_TAB_BUTTON_HEIGHT
        button.styles.min_height = CONSOLE_NEW_TAB_BUTTON_HEIGHT
        button.styles.max_height = CONSOLE_NEW_TAB_BUTTON_HEIGHT
        return button

    @staticmethod
    def rename_input_id(session_id: str) -> str:
        """Return the stable inline rename input ID for a session."""
        return f"console-session-rename-input-{session_id}"

    def start_renaming_session(self, session_id: str) -> None:
        """Enter inline rename mode for a Console session tab."""
        self._renaming_session_id = session_id

    def cancel_renaming_session(self) -> None:
        """Exit inline rename mode without changing the session title."""
        self._renaming_session_id = None

    def focus_rename_input(self) -> None:
        """Focus the active inline rename input when it is mounted."""
        if self._renaming_session_id is None:
            return
        try:
            rename_input = self.query_one(
                f"#{self.rename_input_id(self._renaming_session_id)}",
                Input,
            )
            rename_input.focus()
            rename_input.select_all()
        except Exception:
            return

    @classmethod
    def _display_title(cls, title: str) -> str:
        """Return a tab label that preserves space for close/rename controls."""
        normalized_title = title.strip() or "Untitled"
        if len(normalized_title) <= CONSOLE_SESSION_TAB_DISPLAY_CHARS:
            return normalized_title
        visible_chars = CONSOLE_SESSION_TAB_DISPLAY_CHARS - 3
        return f"{normalized_title[:visible_chars].rstrip()}..."

    def _build_session_tab_button(
        self,
        session: ConsoleChatSession,
        *,
        active: bool,
    ) -> Button:
        """Build a stable-width Console session tab title button."""
        classes = "console-session-tab"
        if active:
            classes = f"{classes} console-session-tab-active"
        button = Button(
            self._display_title(session.title),
            id=f"console-session-tab-{session.id}",
            classes=classes,
            compact=True,
        )
        button.tooltip = session.title
        button.styles.width = CONSOLE_SESSION_TAB_WIDTH
        button.styles.min_width = CONSOLE_SESSION_TAB_WIDTH
        button.styles.max_width = CONSOLE_SESSION_TAB_WIDTH
        button.styles.height = 1
        button.styles.min_height = 1
        button.styles.max_height = 1
        return button

    def _build_rename_input(self, session: ConsoleChatSession) -> Input:
        """Build the active tab's inline rename editor."""
        rename_input = Input(
            value=session.title,
            id=self.rename_input_id(session.id),
            classes="console-session-rename-input",
        )
        rename_input.tooltip = "Enter to save, Esc to cancel"
        rename_input.styles.width = CONSOLE_SESSION_RENAME_INPUT_WIDTH
        rename_input.styles.min_width = CONSOLE_SESSION_RENAME_INPUT_WIDTH
        rename_input.styles.max_width = CONSOLE_SESSION_RENAME_INPUT_WIDTH
        rename_input.styles.height = 1
        rename_input.styles.min_height = 1
        rename_input.styles.max_height = 1
        return rename_input

    def _build_close_tab_button(self, session: ConsoleChatSession) -> Button:
        """Build the compact close control for a Console session tab."""
        close_button = Button(
            "x",
            id=f"console-close-session-tab-{session.id}",
            classes="console-session-close-button",
            compact=True,
        )
        close_button.tooltip = "Close Console tab"
        close_button.styles.width = CONSOLE_CLOSE_TAB_BUTTON_WIDTH
        close_button.styles.min_width = CONSOLE_CLOSE_TAB_BUTTON_WIDTH
        close_button.styles.max_width = CONSOLE_CLOSE_TAB_BUTTON_WIDTH
        close_button.styles.height = CONSOLE_CLOSE_TAB_BUTTON_HEIGHT
        close_button.styles.min_height = CONSOLE_CLOSE_TAB_BUTTON_HEIGHT
        close_button.styles.max_height = CONSOLE_CLOSE_TAB_BUTTON_HEIGHT
        return close_button

    def _build_rename_tab_button(self, session: ConsoleChatSession) -> Button:
        """Build the active tab rename affordance."""
        rename_button = Button(
            "Rename",
            id=f"console-rename-session-tab-{session.id}",
            classes="console-session-rename-button",
            compact=True,
        )
        rename_button.tooltip = "Rename Console tab"
        rename_button.styles.width = CONSOLE_RENAME_TAB_BUTTON_WIDTH
        rename_button.styles.min_width = CONSOLE_RENAME_TAB_BUTTON_WIDTH
        rename_button.styles.max_width = CONSOLE_RENAME_TAB_BUTTON_WIDTH
        rename_button.styles.height = 1
        rename_button.styles.min_height = 1
        rename_button.styles.max_height = 1
        return rename_button

    def _desired_tab_child_ids(
        self,
        *,
        sessions: list[ConsoleChatSession],
        active_session_id: str | None,
    ) -> list[str]:
        """Return the expected child ID sequence for the session tab strip."""
        desired_ids: list[str] = []
        for session in sessions:
            if session.id == self._renaming_session_id:
                desired_ids.append(self.rename_input_id(session.id))
            else:
                desired_ids.append(f"console-session-tab-{session.id}")
            desired_ids.append(f"console-close-session-tab-{session.id}")
            if (
                session.id == active_session_id
                and session.id != self._renaming_session_id
            ):
                desired_ids.append(f"console-rename-session-tab-{session.id}")
        desired_ids.append("console-new-chat-tab")
        return desired_ids

    def _update_existing_tab_strip(
        self,
        *,
        tab_strip: Horizontal,
        sessions: list[ConsoleChatSession],
        active_session_id: str | None,
    ) -> None:
        """Update labels, tooltips, and active state without stealing focus."""
        session_by_id = {session.id: session for session in sessions}
        for child in tab_strip.children:
            child_id = child.id or ""
            if child_id.startswith("console-session-tab-"):
                session_id = child_id.removeprefix("console-session-tab-")
                session = session_by_id.get(session_id)
                if session is None or not isinstance(child, Button):
                    continue
                child.label = self._display_title(session.title)
                child.tooltip = session.title
                child.set_class(
                    session.id == active_session_id,
                    "console-session-tab-active",
                )
            elif child_id.startswith("console-rename-session-tab-") and isinstance(
                child,
                Button,
            ):
                child.tooltip = "Rename Console tab"

    async def sync_sessions(
        self,
        *,
        sessions: list[ConsoleChatSession],
        active_session_id: str | None,
    ) -> None:
        """Render native Console session tabs from controller-owned state."""
        async with self._session_sync_lock:
            session_ids = {session.id for session in sessions}
            if (
                self._renaming_session_id is not None
                and (
                    self._renaming_session_id not in session_ids
                    or self._renaming_session_id != active_session_id
                )
            ):
                self.cancel_renaming_session()
            tab_strip = self.query_one("#console-native-tab-strip", Horizontal)
            desired_ids = self._desired_tab_child_ids(
                sessions=sessions,
                active_session_id=active_session_id,
            )
            existing_ids = [child.id for child in tab_strip.children]
            if existing_ids == desired_ids:
                self._update_existing_tab_strip(
                    tab_strip=tab_strip,
                    sessions=sessions,
                    active_session_id=active_session_id,
                )
                return

            for child in list(tab_strip.children):
                await child.remove()
            for session in sessions:
                is_active = session.id == active_session_id
                if session.id == self._renaming_session_id:
                    await tab_strip.mount(self._build_rename_input(session))
                else:
                    await tab_strip.mount(
                        self._build_session_tab_button(session, active=is_active)
                    )
                await tab_strip.mount(self._build_close_tab_button(session))
                if is_active and session.id != self._renaming_session_id:
                    await tab_strip.mount(self._build_rename_tab_button(session))
            await tab_strip.mount(self._build_new_tab_button())

    def sync_inline_guidance(self, *, visible: bool, copy: str = "") -> None:
        """Keep guidance out of the persistent transcript title."""
        try:
            title = self.query_one("#console-transcript-title", Static)
        except Exception:
            return
        title.update(CONSOLE_TRANSCRIPT_TITLE)
