"""Console Inspector "Retrieval scope" row (task-9)."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_display_state import ConsoleRetrievalScopeState
from tldw_chatbook.Chat.rag_scope import SCOPE_EMPTY_NOTICE_TEMPLATE, SCOPE_REASON_EMPTY

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

    Three states -- the same three ``ConsoleRetrievalScopeState`` renders
    the header's "Scope" chip (task-10, ``ConsoleControlBar._scope_chip_
    render``) also handles, since row and chip are one state, two
    renderers and must never diverge:

    - Unscoped: "Scope: everything" + a "Narrow…" button that opens the
      picker modal.
    - Scoped: "Scope: N items" + "Edit" (reopens the picker, seeded with
      the current selection) and "Clear" (clears the scope) buttons.
    - Empty (``is_empty``): the configured scope resolves to nothing to
      retrieve from -- every item in an active scope has since been
      deleted, OR (task-13, Phase 3) a conversation/workspace intersection
      has no overlap. Renders "Scope: empty" with the cause folded into the
      label's tooltip via ``SCOPE_EMPTY_NOTICE_TEMPLATE``, same wording the
      chip's tooltip uses, plus the "Narrow…" button (an empty scope still
      lets the user pick something that actually resolves). Reached from
      the real ``ChatScreen._build_console_retrieval_scope_state`` via the
      off-loop-resolved ``self._console_effective_scope_cache`` (Phase 3
      wired the conversation/workspace intersection into the display path;
      the builder itself stays zero-DB, reading only that cache).
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
        # task-10 review finding 1: EMPTY must be checked ahead of
        # ``is_scoped`` -- an EMPTY snapshot has ``is_scoped=False`` (see
        # ``ConsoleRetrievalScopeState.empty()``), so an ``is_scoped``-only
        # branch here would silently render it as the plain "everything"
        # default, diverging from the header chip's ``_scope_chip_render``
        # (which checks ``is_empty`` explicitly). Row and chip render the
        # exact same state snapshot and must agree.
        if self.state.is_empty:
            label = "Scope: empty"
        elif self.state.is_scoped:
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
        if self.state.is_empty:
            # Cause folded into the label's tooltip, same wording and same
            # ``SCOPE_EMPTY_NOTICE_TEMPLATE`` the chip's tooltip uses
            # (``ConsoleControlBar._scope_chip_render``) -- reuses the
            # row's existing label widget rather than a second Static,
            # since the row's own frame is pinned to a single text line
            # (``#console-retrieval-scope-row`` is ``max-height: 1``).
            cause = self.state.cause or SCOPE_REASON_EMPTY
            label_widget.tooltip = SCOPE_EMPTY_NOTICE_TEMPLATE.format(cause=cause)
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
