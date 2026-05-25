"""Console-native chat session surface."""

from __future__ import annotations

import asyncio
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


CONSOLE_CLOSE_TAB_BUTTON_WIDTH = 3
CONSOLE_CLOSE_TAB_BUTTON_HEIGHT = 1
CONSOLE_NEW_TAB_BUTTON_WIDTH = 3
CONSOLE_NEW_TAB_BUTTON_HEIGHT = 1
CONSOLE_TRANSCRIPT_TITLE = "Transcript / Event Stream"


class ConsoleSessionSurface(Vertical):
    """Host Console transcript/event stream sessions without legacy chat chrome."""

    def __init__(self, app_instance: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self._session_sync_lock = asyncio.Lock()

    def compose(self) -> ComposeResult:
        title = Static(
            CONSOLE_TRANSCRIPT_TITLE,
            id="console-transcript-title",
            classes="destination-section",
        )
        title.styles.height = 1
        title.styles.min_height = 1
        yield title

        tab_strip = Horizontal(id="console-native-tab-strip")
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

    async def sync_sessions(
        self,
        *,
        sessions: list[ConsoleChatSession],
        active_session_id: str | None,
    ) -> None:
        """Render native Console session tabs from controller-owned state."""
        async with self._session_sync_lock:
            tab_strip = self.query_one("#console-native-tab-strip", Horizontal)
            desired_ids = []
            for session in sessions:
                desired_ids.extend(
                    (
                        f"console-session-tab-{session.id}",
                        f"console-close-session-tab-{session.id}",
                    )
                )
            desired_ids.append("console-new-chat-tab")
            existing_ids = [child.id for child in tab_strip.children]
            if existing_ids == desired_ids:
                for child in tab_strip.children:
                    if child.id and child.id.startswith("console-session-tab-"):
                        child.set_class(
                            child.id == f"console-session-tab-{active_session_id}",
                            "console-session-tab-active",
                        )
                return

            for child in list(tab_strip.children):
                await child.remove()
            for session in sessions:
                classes = "console-session-tab"
                if session.id == active_session_id:
                    classes = f"{classes} console-session-tab-active"
                await tab_strip.mount(
                    Button(
                        session.title,
                        id=f"console-session-tab-{session.id}",
                        classes=classes,
                    )
                )
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
                await tab_strip.mount(close_button)
            await tab_strip.mount(self._build_new_tab_button())

    def sync_inline_guidance(self, *, visible: bool, copy: str = "") -> None:
        """Render compact first-run guidance without adding a separate row."""
        try:
            title = self.query_one("#console-transcript-title", Static)
        except Exception:
            return
        guidance = " ".join(str(copy or "").split())
        if visible and guidance:
            title.update(f"{CONSOLE_TRANSCRIPT_TITLE} | {guidance}")
            return
        title.update(CONSOLE_TRANSCRIPT_TITLE)
