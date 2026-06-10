"""Selected-item inspector pane for the Personas workbench."""

from __future__ import annotations

import re

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Static

from .personas_pane_messages import ConversationRowSelected

_UNSAVED_TOOLTIP = "Save before using this action; the selection has unsaved edits."

_ID_SAFE = re.compile(r"[^a-zA-Z0-9_-]")


class PersonasInspectorPane(Vertical):
    """Identity, validation, conversations, readiness, and actions."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._has_selection = False
        self._is_unsaved = False
        self._selected_kind: str | None = None
        self._conversation_lookup: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Static("Inspector", classes="destination-section personas-column-title")
        yield Static("Selected: none", id="personas-selected-name")
        yield Static("Type: -", id="personas-selected-kind")
        yield Static("Authority: Local", id="personas-selected-authority")
        yield Static("Validation: OK", id="personas-validation-summary")
        yield Static("Conversations", classes="destination-section")
        yield VerticalScroll(id="personas-conversations-list")
        yield Static("Readiness", classes="destination-section")
        yield Static("Console: Blocked - select an item", id="personas-readiness-console")
        with Vertical(id="personas-inspector-actions"):
            yield Button("Attach to Console", id="personas-attach-to-console", disabled=True)
            yield Button("Start Chat", id="personas-start-chat", disabled=True)
            yield Button("Export JSON", id="personas-export-json", disabled=True)
            yield Button("Export PNG", id="personas-export-png", disabled=True)
            yield Button(
                "Delete",
                id="personas-delete",
                disabled=True,
                classes="personas-destructive",
            )

    def show_selection(self, *, name: str, kind: str, authority: str) -> None:
        self._has_selection = True
        self._selected_kind = kind
        self.query_one("#personas-selected-name", Static).update(f"Selected: {name}")
        self.query_one("#personas-selected-kind", Static).update(f"Type: {kind}")
        self.query_one("#personas-selected-authority", Static).update(f"Authority: {authority}")
        self._apply_action_state()

    async def clear_selection(self) -> None:
        self._has_selection = False
        self._is_unsaved = False
        self._selected_kind = None
        self.query_one("#personas-selected-name", Static).update("Selected: none")
        self.query_one("#personas-selected-kind", Static).update("Type: -")
        self.query_one("#personas-selected-authority", Static).update("Authority: Local")
        await self.show_conversations(())
        self.show_validation(())
        self._apply_action_state()

    def set_unsaved(self, is_unsaved: bool) -> None:
        self._is_unsaved = is_unsaved
        self._apply_action_state()

    def show_validation(self, errors: tuple[str, ...]) -> None:
        summary = self.query_one("#personas-validation-summary", Static)
        if errors:
            summary.update("Validation errors:\n" + "\n".join(errors))
        else:
            summary.update("Validation: OK")

    async def show_conversations(self, rows: tuple[tuple[str, str], ...]) -> None:
        """Render (conversation_id, title) rows; empty tuple clears the panel."""
        container = self.query_one("#personas-conversations-list", VerticalScroll)
        await container.remove_children()
        self._conversation_lookup = {}
        buttons: list[Button] = []
        seen: set[str] = set()
        for conversation_id, title in rows:
            dom_id = f"personas-conversation-row-{_ID_SAFE.sub('-', str(conversation_id))}"
            if dom_id in seen:
                suffix = 2
                while f"{dom_id}-{suffix}" in seen:
                    suffix += 1
                dom_id = f"{dom_id}-{suffix}"
            seen.add(dom_id)
            self._conversation_lookup[dom_id] = conversation_id
            buttons.append(
                Button(title, id=dom_id, classes="personas-conversation-row")
            )
        if buttons:
            await container.mount_all(buttons)

    def _apply_action_state(self) -> None:
        selected = self._has_selection
        unsaved = self._is_unsaved
        readiness = self.query_one("#personas-readiness-console", Static)
        if not selected:
            readiness.update("Console: Blocked - select an item")
        elif unsaved:
            readiness.update("Console: Blocked - unsaved edits")
        else:
            readiness.update("Console: Ready")
        enabled = selected and not unsaved
        tooltip = _UNSAVED_TOOLTIP if (selected and unsaved) else None
        for button_id in (
            "#personas-attach-to-console",
            "#personas-start-chat",
            "#personas-export-json",
        ):
            button = self.query_one(button_id, Button)
            button.disabled = not enabled
            button.tooltip = tooltip
        png_button = self.query_one("#personas-export-png", Button)
        png_button.disabled = not (enabled and self._selected_kind == "character")
        png_button.tooltip = tooltip
        self.query_one("#personas-delete", Button).disabled = not selected

    @on(Button.Pressed, ".personas-conversation-row")
    def _conversation_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        conversation_id = self._conversation_lookup.get(str(event.button.id or ""))
        if conversation_id is not None:
            self.post_message(ConversationRowSelected(conversation_id))
