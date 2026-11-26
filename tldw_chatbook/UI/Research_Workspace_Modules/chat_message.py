"""Literal Research transcript rows and stored provenance."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, ListItem, Static, TextArea

from ...Research_Workspace.chat_models import validate_chat_draft
from ...Widgets.modal_dismissal import SafeModalDismissMixin


def research_message_context(row: Mapping[str, Any]) -> Mapping[str, Any]:
    context = row.get("rag_context")
    if isinstance(context, Mapping):
        settings = context.get("settings_snapshot")
        if not isinstance(settings, Mapping):
            settings = {}
        return {**settings, "citations": context.get("citations", ())}
    try:
        details = json.loads(row.get("metadata_json") or "{}").get("research_chat", {})
        return details if isinstance(details, Mapping) else {}
    except (TypeError, ValueError, AttributeError):
        return {}


def context_rows(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence):
        return ()
    return tuple(row for row in value if isinstance(row, Mapping))


class ResearchChatMessage(ListItem):
    def __init__(self, row: Mapping[str, Any]) -> None:
        super().__init__(classes="research-chat-message")
        self.row = dict(row)

    def compose(self) -> ComposeResult:
        role = self.row.get("role") or self.row.get("sender")
        context = research_message_context(self.row)
        heading = "You" if role == "user" else "Assistant"
        if (
            self.row.get("assistant_generation_state") == "stopped"
            or context.get("generation_state") == "cancelled"
        ):
            heading += " · Stopped"
        if context.get("requested_mode") == "auto":
            heading += (
                " · Retrieved" if context.get("retrieved") else " · Did not retrieve"
            )
        yield Static(heading, classes="research-chat-message-heading", markup=False)
        yield Static(
            str(self.row.get("content") or ""),
            classes="research-chat-message-content",
            markup=False,
        )


class ResearchChatMessageEditor(SafeModalDismissMixin, ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    AUTO_FOCUS = "#research-chat-edit-body"
    SAFE_MODAL_CONTENT = "#research-chat-edit-dialog"

    def __init__(self, body: str) -> None:
        super().__init__()
        self.body = body

    def compose(self) -> ComposeResult:
        with Vertical(id="research-chat-edit-dialog", classes="dialog research-wide-dialog"):
            yield Static("Edit message", classes="dialog-title")
            yield TextArea(self.body, id="research-chat-edit-body")
            yield Static("", id="research-chat-edit-status", markup=False)
            with Horizontal(classes="research-chat-row"):
                yield Button(
                    "Save",
                    id="research-chat-edit-save",
                    variant="primary",
                    classes="research-chat-action",
                )
                yield Button(
                    "Cancel",
                    id="research-chat-edit-cancel",
                    classes="research-chat-action",
                )

    @on(Button.Pressed, "#research-chat-edit-save")
    def save(self) -> None:
        body = self.query_one(TextArea).text
        try:
            validate_chat_draft(body)
            if not body.strip():
                raise ValueError("Write a message before saving.")
        except ValueError as exc:
            self.query_one("#research-chat-edit-status", Static).update(str(exc))
            return
        self.dismiss(body)

    @on(Button.Pressed, "#research-chat-edit-cancel")
    async def cancel(self) -> None:
        await self.request_safe_cancel(source="button")
