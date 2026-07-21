"""Console Inspector "Retrieval scope" row (task-9)."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_display_state import ConsoleRetrievalScopeState

ROW_ID = "console-retrieval-scope-row"
LABEL_ID = "console-retrieval-scope-label"
NARROW_BTN_ID = "console-retrieval-scope-narrow"
EDIT_BTN_ID = "console-retrieval-scope-edit"
CLEAR_BTN_ID = "console-retrieval-scope-clear"
#: Shared class for the row's modal-opening buttons ("Narrow…" when
#: unscoped, "Edit" when scoped) -- only one is ever mounted at a time, so
#: the screen wires a single handler to this class rather than to both ids.
OPEN_BUTTON_CLASS = "console-retrieval-scope-open-btn"
CLEAR_BUTTON_CLASS = "console-retrieval-scope-clear-btn"

UNSCOPED_LABEL = "Scope: everything"
NARROW_LABEL = "Narrow…"
EDIT_LABEL = "Edit"
CLEAR_LABEL = "Clear"


class ConsoleRetrievalScopeRow(Horizontal):
    """Compact Inspector row summarizing the active conversation's RAG scope.

    Sits directly below the Sources (staged-context) tray in the Inspector
    rail body as its own sibling widget -- never a row mounted inside
    ``ConsoleRunInspector`` or a button folded into the staged-context tray
    (design spec section 4: "A separate row, not a button inside the tray:
    staged-vs-scope mechanism boundary stays visible"). Renders purely from
    a ``ConsoleRetrievalScopeState`` snapshot: no DB access at
    compose/recompose time (task-9's zero-DB-on-recompose contract) -- the
    owning screen reads the actual scope on modal open and after save
    (both off the UI loop) and pushes the result in via ``sync_state``.

    Two states:

    - Unscoped: "Scope: everything" + a "Narrow…" button that opens the
      picker modal.
    - Scoped: "Scope: N items" + "Edit" (reopens the picker, seeded with
      the current selection) and "Clear" (clears the scope) buttons.
    """

    def __init__(self, state: ConsoleRetrievalScopeState, **kwargs: Any) -> None:
        """Initialize the row.

        Args:
            state: Display-state snapshot to render.
            **kwargs: Additional Textual widget arguments (``id``, etc.).
        """
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        if self.state.is_scoped:
            label = f"Scope: {self.state.item_count} items"
        else:
            label = UNSCOPED_LABEL
        label_widget = Static(
            label,
            id=LABEL_ID,
            classes="console-retrieval-scope-label",
            markup=False,
        )
        # Set inline (not CSS-only): the lightweight ``ConsoleHarness`` test
        # apps many Console tests use never load the bundled stylesheet
        # (only widget ``DEFAULT_CSS``), so a plain ``Static``'s width would
        # otherwise default to filling the WHOLE row and push the
        # button(s) off past the row's own bounds -- mirrors
        # ``_frame_console_region``'s own inline-Python-styles-over-CSS
        # discipline for exactly this reason.
        label_widget.styles.width = "1fr"
        label_widget.styles.min_width = 0
        yield label_widget
        if self.state.is_scoped:
            yield self._action_button(
                EDIT_LABEL, EDIT_BTN_ID, f"console-retrieval-scope-action {OPEN_BUTTON_CLASS}"
            )
            yield self._action_button(
                CLEAR_LABEL,
                CLEAR_BTN_ID,
                f"console-retrieval-scope-action {CLEAR_BUTTON_CLASS}",
            )
        else:
            yield self._action_button(
                NARROW_LABEL,
                NARROW_BTN_ID,
                f"console-retrieval-scope-action {OPEN_BUTTON_CLASS}",
            )

    @staticmethod
    def _action_button(label: str, button_id: str, classes: str) -> Button:
        button = Button(label, id=button_id, classes=classes, compact=True)
        button.styles.width = "auto"
        return button

    def sync_state(self, state: ConsoleRetrievalScopeState) -> None:
        """Refresh the mounted row from a new display-state snapshot.

        Equality-guarded like the other Console tray widgets (e.g.
        ``ConsoleStagedContextTray``): a real change (unscoped<->scoped, or
        a changed item count) recomposes only this row, never the owning
        screen.

        Args:
            state: Display-state snapshot to render.
        """
        if state == self.state:
            return
        self.state = state
        self.refresh(recompose=True)
