"""Library RAG settings: the RAG chip's enable-and-customize modal.

"RAG: off" in the Console status strip is not a latent toggle -- the chip
reads "on" once retrieved Library evidence is staged for the next send.
This modal is where a user makes that happen: it owns the retrieval query
(the same state the visible "Run Library RAG" actions read) and runs the
retrieval that stages evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Click
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static


@dataclass(frozen=True)
class ConsoleRagSettingsResult:
    """Outcome of the Library RAG settings modal.

    Attributes:
        query: The retrieval query as typed (the screen sanitizes it).
        run: Whether to run Library retrieval now with that query.
    """

    query: str
    run: bool


class ConsoleRagSettingsModal(ModalScreen["ConsoleRagSettingsResult | None"]):
    """Edit the Library RAG query and optionally run retrieval now."""

    DEFAULT_CSS = """
    ConsoleRagSettingsModal {
        align: center middle;
    }

    #console-rag-settings {
        width: 64;
        height: auto;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    .console-rag-settings-status {
        margin: 0 0 1 0;
    }

    #console-rag-settings-query {
        margin: 0 0 1 0;
    }

    .console-rag-settings-scope {
        color: $text-muted;
        margin: 0 0 1 0;
    }

    .console-rag-settings-actions {
        height: 3;
    }

    .console-rag-settings-actions Button {
        margin: 0 2 0 0;
    }

    .console-rag-settings-hint {
        color: $text-muted;
        margin: 1 0 0 0;
    }
    """

    BINDINGS = [("escape", "dismiss_modal", "Cancel")]

    def __init__(
        self,
        *,
        query: str = "",
        scope_label: str = "",
        rag_active: bool = False,
        staged_title: str = "",
        **kwargs: Any,
    ) -> None:
        """Initialize the modal.

        Args:
            query: Prefill for the retrieval query input.
            scope_label: Read-only source-scope line (e.g. ``"Scope: notes,
                media, conversations"``).
            rag_active: Whether RAG currently reads "on" (staged evidence).
            staged_title: Title of the staged evidence when ``rag_active``,
                for honest status copy.
            **kwargs: Forwarded to ``ModalScreen``.
        """
        super().__init__(**kwargs)
        self._query = query
        self._scope_label = scope_label
        self._rag_active = rag_active
        self._staged_title = staged_title

    def _status_copy(self) -> str:
        """Return honest RAG-state copy for the top of the modal."""
        if self._rag_active:
            staged = f" ({self._staged_title})" if self._staged_title else ""
            return (
                f"RAG is on: retrieved Library evidence{staged} is staged "
                "for your next send. Running again replaces it."
            )
        return (
            "RAG is off. It turns on once you run retrieval and Library "
            "evidence is staged for your next send."
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="console-rag-settings"):
            yield Static("Library RAG", classes="console-modal-header")
            yield Static(
                self._status_copy(),
                classes="console-rag-settings-status",
                markup=False,
            )
            yield Input(
                value=self._query,
                placeholder="What should Library retrieval look for?",
                id="console-rag-settings-query",
            )
            if self._scope_label:
                yield Static(
                    self._scope_label,
                    id="console-rag-settings-scope",
                    classes="console-rag-settings-scope",
                    markup=False,
                )
            with Horizontal(classes="console-rag-settings-actions"):
                yield Button(
                    "Run Library RAG",
                    id="console-rag-settings-run",
                    variant="primary",
                    disabled=not self._query.strip(),
                )
                yield Button("Cancel", id="console-rag-settings-cancel")
            yield Static(
                "Enter runs retrieval. Esc or a click outside closes "
                "without changes.",
                classes="console-rag-settings-hint",
                markup=False,
            )

    def _current_query(self) -> str:
        return self.query_one("#console-rag-settings-query", Input).value

    @on(Input.Changed, "#console-rag-settings-query")
    def _sync_run_availability(self, event: Input.Changed) -> None:
        """Keep the Run action gated on a non-blank query."""
        event.stop()
        run_button = self.query_one("#console-rag-settings-run", Button)
        run_button.disabled = not str(event.value or "").strip()

    @on(Input.Submitted, "#console-rag-settings-query")
    def _submit_query(self, event: Input.Submitted) -> None:
        """Enter in the query input runs retrieval, matching the hint copy."""
        event.stop()
        query = self._current_query()
        if not query.strip():
            return
        self.dismiss(ConsoleRagSettingsResult(query=query, run=True))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "console-rag-settings-run":
            event.stop()
            self.dismiss(
                ConsoleRagSettingsResult(query=self._current_query(), run=True)
            )
            return
        if button_id == "console-rag-settings-cancel":
            event.stop()
            self.dismiss(None)

    def on_click(self, event: Click) -> None:
        """Dismiss with no changes when a click lands on the backdrop.

        Same contract as the composer ☰ menu: containment is tested against
        the modal box's region, and a click that carries no screen
        coordinates (synthesized clicks under textual-web) keeps the modal
        open rather than guessing.

        Args:
            event: The screen-level click, carrying absolute coordinates.
        """
        screen_x = getattr(event, "screen_x", None)
        screen_y = getattr(event, "screen_y", None)
        if screen_x is None or screen_y is None:
            return
        box = self.query_one("#console-rag-settings")
        if box.region.contains(screen_x, screen_y):
            return
        event.stop()
        self.dismiss(None)

    def action_dismiss_modal(self) -> None:
        self.dismiss(None)
