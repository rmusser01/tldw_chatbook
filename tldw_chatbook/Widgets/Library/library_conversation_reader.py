"""Permanent read-only work pane for saved Library conversations."""

from __future__ import annotations

from typing import Any, Mapping

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_conversation_reader_state import (
    ConversationMessageView,
    ConversationReaderState,
)
from tldw_chatbook.Library.library_shell_state import library_disabled_action_label


def _open_console_disabled_tooltip(state: ConversationReaderState) -> str | None:
    """Describe the current reason the retained transcript cannot be opened."""
    if state.loaded_actions_eligible:
        return None
    if state.bulk_active:
        return "Finish bulk selection before opening a conversation in Console."
    if state.loading:
        return "Wait for the selected conversation to finish loading."
    if state.unavailable:
        return "The selected conversation is unavailable."
    if state.error:
        return "Try again before opening the selected conversation in Console."
    if (
        state.selected_id != state.loaded_id
        or state.selected_version != state.loaded_version
        or state.generation != state.loaded_generation
    ):
        return "The selected conversation does not match the retained transcript."
    return "Wait for the complete selected transcript before opening it in Console."


class LibraryConversationReader(Vertical):
    """Render one retained Conversations Read/Info pane from pure state."""

    class MessagesSynced(Message):
        """Notify the controller that current transcript rows are mounted."""

        def __init__(self, reader_generation: int, find_query: str) -> None:
            self.reader_generation = reader_generation
            self.find_query = find_query
            super().__init__()

    def __init__(
        self,
        state: ConversationReaderState,
        *,
        metadata: Mapping[str, Any] | None = None,
        loaded_metadata: Mapping[str, Any] | None = None,
        selected_metadata: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.loaded_metadata = dict(loaded_metadata or metadata or {})
        self.selected_metadata = dict(selected_metadata or {})
        self._message_sync_generation = 0

    def compose(self) -> ComposeResult:
        """Compose stable controls and the initially available transcript."""
        eligible = self.state.loaded_actions_eligible
        yield Static("Conversation reader", classes="destination-section", markup=False)
        with Horizontal(classes="ds-toolbar library-conversation-reader-modes"):
            read = Button(
                "Read",
                id="library-conversation-reader-read",
                classes="library-canvas-action",
                compact=True,
            )
            read.set_class(self.state.mode == "read", "-selected")
            yield read
            info = Button(
                "Info",
                id="library-conversation-reader-info",
                classes="library-canvas-action",
                compact=True,
            )
            info.set_class(self.state.mode == "info", "-selected")
            yield info
        yield Static(
            self._status_text(),
            id="library-conversation-reader-status",
            markup=False,
        )
        find = Input(
            value=self.state.find_query,
            placeholder="Find in complete transcript…",
            id="library-conversation-reader-find",
        )
        find.display = self.state.mode == "read"
        yield find
        messages = VerticalScroll(id="library-conversation-reader-messages")
        messages.display = self.state.mode == "read"
        with messages:
            for message in self.state.messages:
                yield self._message_widget(message)
        info_body = Static(
            self._metadata_text(),
            id="library-conversation-reader-info-body",
            markup=False,
        )
        info_body.display = self.state.mode == "info"
        yield info_body
        with Horizontal(classes="ds-toolbar library-conversation-reader-actions"):
            open_console = Button(
                library_disabled_action_label("Open in Console", not eligible),
                id="library-conversation-open-console",
                classes="library-canvas-action",
                compact=True,
            )
            open_console.disabled = not eligible
            open_console.tooltip = _open_console_disabled_tooltip(self.state)
            yield open_console
            retry = Button(
                "Try again",
                id="library-conversation-reader-retry",
                classes="library-canvas-action",
                compact=True,
            )
            retry.display = bool(self.state.error or self.state.unavailable)
            yield retry

    def on_mount(self) -> None:
        """Project initial labels and visibility without replacing this widget."""
        self.sync_state(
            self.state,
            loaded_metadata=self.loaded_metadata,
            selected_metadata=self.selected_metadata,
        )

    @staticmethod
    def _message_copy(message: ConversationMessageView) -> str:
        heading = " · ".join(
            value for value in (message.sender, message.timestamp) if value
        )
        return f"{heading}\n{message.text}" if heading else message.text

    @classmethod
    def _message_widget(cls, message: ConversationMessageView) -> Static:
        row = Static(
            cls._message_copy(message),
            classes="library-conversation-reader-message",
            markup=False,
        )
        row.message_id = message.message_id
        row.can_focus = True
        return row

    def _metadata_text(self) -> str:
        loaded = self.state.loaded_id is not None
        title = str(self.loaded_metadata.get("title") or "Unknown title")
        conversation_id = self.state.loaded_id or "unknown"
        version = self.state.loaded_version
        workspace = str(
            self.loaded_metadata.get("workspace")
            or self.loaded_metadata.get("workspace_name")
            or "unassigned"
        )
        updated = str(
            self.loaded_metadata.get("last_modified")
            or self.loaded_metadata.get("updated_at")
            or self.loaded_metadata.get("updated")
            or "unknown"
        )
        raw_keywords = self.loaded_metadata.get("keywords")
        if isinstance(raw_keywords, (list, tuple)):
            keywords = ", ".join(str(value) for value in raw_keywords if str(value))
        else:
            keywords = ""
        return "\n".join(
            (
                f"Title: {title}",
                f"Conversation ID: {conversation_id}",
                f"Version: {version if version is not None else 'unknown'}",
                f"Messages: {self.state.message_total if loaded else 'unknown'}",
                f"Workspace: {workspace}",
                f"Updated: {updated}",
                f"Keywords: {keywords or 'unknown'}",
                "Authority: local saved conversation",
            )
        )

    def _status_text(self) -> str:
        state = self.state
        list_status = str(self.loaded_metadata.get("_list_status") or "").strip()
        if list_status:
            return list_status
        list_summary = str(self.loaded_metadata.get("_list_summary") or "").strip()

        selected_title = str(
            self.selected_metadata.get("title")
            or state.selected_id
            or "selected conversation"
        )
        loaded_title = str(
            self.loaded_metadata.get("title")
            or state.loaded_id
            or "loaded conversation"
        )

        def with_list_summary(copy: str) -> str:
            if state.find_query:
                find_copy = (
                    f"Find: {len(state.find_matches)} exact "
                    f"{'match' if len(state.find_matches) == 1 else 'matches'}."
                    if state.find_complete
                    else "Searching complete transcript…"
                )
                copy = f"{copy} · {find_copy}"
            return f"{list_summary} · {copy}" if list_summary else copy

        if state.bulk_active:
            if state.bulk_loaded_preview_selected is True:
                preview = "The retained transcript is included and remains read-only."
            elif state.bulk_loaded_preview_selected is False:
                preview = (
                    "The retained transcript is not included and remains read-only."
                )
            else:
                preview = "No transcript is retained; Read and Info remain available."
            return with_list_summary(
                f"Bulk selection: {state.bulk_selected_count} conversations. {preview}"
            )
        if state.unavailable:
            copy = state.error or "Conversation unavailable."
            if state.selected_id:
                copy = f"{copy} Selected {selected_title} ({state.selected_id})."
            if state.loaded_id and state.loaded_id != state.selected_id:
                copy += f" Showing {loaded_title} ({state.loaded_id})."
            return with_list_summary(copy)
        if state.error:
            if state.loaded_id:
                copy = state.error
                if state.selected_id:
                    copy += f" Selected {selected_title} ({state.selected_id})."
                copy += f" Showing {loaded_title} ({state.loaded_id})."
                return with_list_summary(copy)
            selected = state.selected_id
            if selected:
                copy = f"{state.error} Selected {selected_title} ({selected})."
                if state.loaded_id and state.loaded_id != selected:
                    copy += f" Showing {loaded_title} ({state.loaded_id})."
                return with_list_summary(copy)
            return with_list_summary(state.error)
        if state.loading:
            selected = state.selected_id or "selected conversation"
            if state.loaded_id and state.loaded_id != selected:
                return with_list_summary(
                    f"Loading {selected_title} ({selected}); showing "
                    f"{loaded_title} ({state.loaded_id}) until ready."
                )
            return with_list_summary(f"Loading {selected_title} ({selected})…")
        if state.loaded_id:
            suffix = "complete" if state.complete else "loading more"
            return with_list_summary(
                f"Loaded {state.loaded_id} · {len(state.messages)} of "
                f"{state.message_total} messages · {suffix}."
            )
        return with_list_summary("Select a conversation to read it here.")

    def sync_state(
        self,
        state: ConversationReaderState,
        *,
        metadata: Mapping[str, Any] | None = None,
        loaded_metadata: Mapping[str, Any] | None = None,
        selected_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Patch state, labels, and progressive message rows in place."""
        self.state = state
        if loaded_metadata is not None or metadata is not None:
            self.loaded_metadata = dict(loaded_metadata or metadata or {})
        if selected_metadata is not None:
            self.selected_metadata = dict(selected_metadata)
        if not self.is_mounted:
            return

        read = self.query_one("#library-conversation-reader-read", Button)
        info = self.query_one("#library-conversation-reader-info", Button)
        read.set_class(state.mode == "read", "-selected")
        info.set_class(state.mode == "info", "-selected")
        self.query_one("#library-conversation-reader-status", Static).update(
            self._status_text()
        )
        find = self.query_one("#library-conversation-reader-find", Input)
        if find.value != state.find_query and not find.has_focus:
            find.value = state.find_query
        find.display = state.mode == "read"
        messages = self.query_one(
            "#library-conversation-reader-messages", VerticalScroll
        )
        messages.display = state.mode == "read"
        info_body = self.query_one("#library-conversation-reader-info-body", Static)
        info_body.update(self._metadata_text())
        info_body.display = state.mode == "info"

        eligible = state.loaded_actions_eligible
        open_console = self.query_one("#library-conversation-open-console", Button)
        open_console.disabled = not eligible
        open_console.label = library_disabled_action_label(
            "Open in Console", not eligible
        )
        open_console.tooltip = _open_console_disabled_tooltip(state)
        retry = self.query_one("#library-conversation-reader-retry", Button)
        retry.display = bool(state.error or state.unavailable)

        self._message_sync_generation += 1
        self.call_later(self._sync_messages, self._message_sync_generation)

    async def _sync_messages(self, generation: int) -> None:
        """Mount or patch stable message rows after the current compose settles."""
        if generation != self._message_sync_generation or not self.is_mounted:
            return
        container = self.query_one(
            "#library-conversation-reader-messages", VerticalScroll
        )
        mounted = {
            str(getattr(row, "message_id", "")): row
            for row in container.children
            if getattr(row, "message_id", None)
        }
        desired_ids = {message.message_id for message in self.state.messages}
        stale = [
            row for message_id, row in mounted.items() if message_id not in desired_ids
        ]
        if stale:
            await container.remove_children(stale)
        for message in self.state.messages:
            row = mounted.get(message.message_id)
            if row is None or not row.is_mounted:
                await container.mount(self._message_widget(message))
            else:
                row.update(self._message_copy(message))
        if generation == self._message_sync_generation and self.is_mounted:
            self.post_message(
                self.MessagesSynced(self.state.generation, self.state.find_query)
            )

    def focus_find_match(self, message_id: str) -> bool:
        """Focus and reveal one stable message reference."""
        for row in self.query(".library-conversation-reader-message"):
            if getattr(row, "message_id", None) == message_id:
                row.scroll_visible(animate=False)
                row.focus(scroll_visible=False)
                return True
        return False
