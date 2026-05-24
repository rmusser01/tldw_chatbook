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


class ConsoleSessionSurface(Vertical):
    """Host Console transcript/event stream sessions without legacy chat chrome."""

    def __init__(self, app_instance: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self._session_sync_lock = asyncio.Lock()

    def compose(self) -> ComposeResult:
        yield Static(
            "Transcript / Event Stream",
            id="console-transcript-title",
            classes="destination-section",
        )
        with Horizontal(id="console-native-tab-strip"):
            yield Button("New tab", id="console-new-chat-tab")
        yield ChatTaskCards(id="console-task-surface")
        yield ConsoleTranscript(id="console-native-transcript")

    async def sync_sessions(
        self,
        *,
        sessions: list[ConsoleChatSession],
        active_session_id: str | None,
    ) -> None:
        """Render native Console session tabs from controller-owned state."""
        async with self._session_sync_lock:
            tab_strip = self.query_one("#console-native-tab-strip", Horizontal)
            desired_ids = [f"console-session-tab-{session.id}" for session in sessions]
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
            await tab_strip.mount(Button("New tab", id="console-new-chat-tab"))
