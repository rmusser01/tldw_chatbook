"""Console-native chat session surface."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import Static

from tldw_chatbook.Widgets.Chat_Widgets.chat_tab_container import ChatTabContainer
from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards


class ConsoleSessionSurface(Vertical):
    """Host Console transcript/event stream sessions without legacy chat chrome."""

    def __init__(self, app_instance: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.tab_container: ChatTabContainer | None = None

    def compose(self) -> ComposeResult:
        yield Static(
            "Transcript / Event Stream",
            id="console-transcript-title",
            classes="destination-section",
        )
        yield ChatTaskCards(id="console-task-surface")
        self.tab_container = ChatTabContainer(self.app_instance, id="console-chat-tabs")
        self.tab_container.enhanced_mode = True
        yield self.tab_container

    def get_tab_container(self) -> ChatTabContainer | None:
        """Return the mounted Console tab container when available."""
        if self.tab_container is not None:
            return self.tab_container
        try:
            self.tab_container = self.query_one("#console-chat-tabs", ChatTabContainer)
        except NoMatches:
            return None
        return self.tab_container
