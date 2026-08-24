"""Library search settings: the Library-search chip's enable-and-customize modal.

"Library search: off" in the Console status strip is not a latent toggle --
the chip reads "on" once retrieved Library evidence is staged for the next
send. This modal is where a user makes that happen: it owns the search query
(the same state the visible "Search Library" actions read) and runs the
retrieval that stages evidence.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Sequence

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_rag_state import (
    LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES,
    LIBRARY_RAG_SOURCE_TYPES,
    library_rag_source_scope_summary,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin


#: The Console's Library RAG source-type default: exactly the three kinds
#: retrieval has always run over from this screen (prompts OFF). It is the
#: ONE default -- `chat_screen.CONSOLE_LIBRARY_RAG_SOURCE_SCOPE` is this
#: same tuple -- so making the row editable (RAG-44) changes nothing until
#: a user actually touches a toggle.
CONSOLE_RAG_DEFAULT_SOURCE_TYPES: tuple[str, ...] = (
    "notes",
    "media",
    "conversations",
)
#: The Console's leading noun for the source-scope line. NOT "Scope": the
#: Console already spends that word on the retrieval ITEM scope ("Scope: 2
#: items" in the rail and status strip), which is a different concept --
#: this line says which KINDS of sources retrieval reads.
CONSOLE_RAG_SOURCE_SUMMARY_PREFIX = "Sources"
CONSOLE_RAG_SOURCE_TOGGLE_ID_PREFIX = "console-rag-settings-source-"
CONSOLE_RAG_SOURCE_TOGGLE_CLASS = "console-rag-settings-source-toggle"
_SOURCE_TYPE_LABELS = dict(LIBRARY_RAG_SOURCE_TYPES)

def normalize_console_rag_source_types(value: Any) -> tuple[str, ...]:
    """Return a usable Console RAG source-type selection from loose input.

    Every boundary the selection crosses (modal construction, the screen's
    stored attribute, restored screen state) runs through here, so a
    legacy payload, an unknown identifier, or a non-sequence can never
    reach a retrieval request. Unknown values are dropped and the result
    is ordered canonically; an unusable or empty result falls back to
    `CONSOLE_RAG_DEFAULT_SOURCE_TYPES` rather than retrieving over
    nothing.

    Args:
        value: Any candidate selection (sequence of source-type ids).

    Returns:
        A canonical, non-empty tuple of known source-type identifiers.
    """
    if isinstance(value, str) or not isinstance(value, Iterable):
        return CONSOLE_RAG_DEFAULT_SOURCE_TYPES
    try:
        candidates = {str(item).strip().lower() for item in value}
    except TypeError:
        return CONSOLE_RAG_DEFAULT_SOURCE_TYPES
    normalized = tuple(
        source_type
        for source_type in LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES
        if source_type in candidates
    )
    return normalized or CONSOLE_RAG_DEFAULT_SOURCE_TYPES


def console_rag_source_toggle_label(source_type: str, selected: bool) -> str:
    """Return one source toggle's visible label ("✓ Notes" / "○ Prompts").

    Mirrors the Library Search canvas's own toggle marker convention
    (`scope_toggle_label`) minus its `(N)` count suffix: the Console has
    no per-source counts to show, and inventing one would be a lie. The
    display label itself comes from Library's one label table, so this
    modal never introduces a second source vocabulary.

    Args:
        source_type: A Library source-type identifier.
        selected: Whether that source is currently in the selection.

    Returns:
        The toggle Button's label text.
    """
    marker = "✓" if selected else "○"
    return f"{marker} {_SOURCE_TYPE_LABELS.get(source_type, source_type)}"


@dataclass(frozen=True)
class ConsoleRagSettingsResult:
    """Outcome of the Library search settings modal.

    Attributes:
        query: The retrieval query as typed (the screen sanitizes it).
        run: Whether to run Library retrieval now with that query.
        source_types: Which KINDS of Library sources retrieval should read
            (RAG-44). Deliberately not the retrieval item scope
            (conversation ∩ workspace), which the Console resolves
            separately and this modal never touches.
    """

    query: str
    run: bool
    source_types: tuple[str, ...] = CONSOLE_RAG_DEFAULT_SOURCE_TYPES


class ConsoleRagSettingsModal(
    SafeModalDismissMixin, ModalScreen["ConsoleRagSettingsResult | None"]
):
    """Edit the Library search query and optionally run the search now."""

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

    .console-rag-settings-sources {
        height: auto;
        margin: 0 0 1 0;
    }

    /* Selected through the container id ON PURPOSE. Textual's own Button
       rules carry `border: tall ...` at class-level specificity
       (`Button.-style-default` and friends); a bare
       `.console-rag-settings-source-toggle` rule ties with them and
       loses, which left each toggle rendering as two border rows with a
       ZERO-height content area -- the label invisible while the button
       still clicked. The id selector outranks them. */
    #console-rag-settings .console-rag-settings-source-toggle {
        height: 1;
        min-height: 1;
        max-height: 1;
        min-width: 0;
        width: auto;
        border: none;
        border-top: none;
        border-bottom: none;
        padding: 0 1;
        margin: 0 1 0 0;
        background: $panel;
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

    SAFE_MODAL_CONTENT = "#console-rag-settings"
    BINDINGS = [("escape", "request_safe_cancel", "Cancel")]

    def __init__(
        self,
        *,
        query: str = "",
        source_types: Sequence[str] = CONSOLE_RAG_DEFAULT_SOURCE_TYPES,
        rag_active: bool = False,
        staged_title: str = "",
        **kwargs: Any,
    ) -> None:
        """Initialize the modal.

        Args:
            query: Prefill for the retrieval query input.
            source_types: The Console's current Library source selection --
                which KINDS of sources retrieval reads. Rendered as one
                toggle per Library source and returned (possibly edited) in
                the result.
            rag_active: Whether RAG currently reads "on" (staged evidence).
            staged_title: Title of the staged evidence when ``rag_active``,
                for honest status copy.
            **kwargs: Forwarded to ``ModalScreen``.
        """
        super().__init__(**kwargs)
        self._query = query
        self._source_types = normalize_console_rag_source_types(source_types)
        self._rag_active = rag_active
        self._staged_title = staged_title

    def _status_copy(self) -> str:
        """Return honest Library-search-state copy for the top of the modal."""
        if self._rag_active:
            staged = f" ({self._staged_title})" if self._staged_title else ""
            return (
                f"Library search is on: retrieved Library evidence{staged} is "
                "staged for your next send. Running again replaces it."
            )
        return (
            "Library search is off. It turns on once you run a search and "
            "Library evidence is staged for your next send."
        )

    def _scope_summary(self) -> str:
        """Return the source-scope line for the current toggle selection.

        The Console's readiness-card label
        (``ChatScreen._console_library_rag_scope_label``) is the same
        builder on the same state, so the two surfaces cannot drift.
        """
        return library_rag_source_scope_summary(
            self._source_types,
            prefix=CONSOLE_RAG_SOURCE_SUMMARY_PREFIX,
        )

    def compose(self) -> ComposeResult:
        """Build the modal: status copy, query input, sources, actions.

        The Run action composes disabled when the prefill is blank and is
        kept in step by the ``Input.Changed`` handler below; the source
        toggles gate it the same way (retrieval over zero source kinds
        would read nothing).

        Yields:
            The modal's child widgets.
        """
        with Vertical(id="console-rag-settings"):
            yield Static("Library search", classes="console-modal-header")
            yield Static(
                self._status_copy(),
                classes="console-rag-settings-status",
                markup=False,
            )
            yield Input(
                value=self._query,
                placeholder="What should the Library search look for?",
                id="console-rag-settings-query",
            )
            yield Static(
                self._scope_summary(),
                id="console-rag-settings-scope",
                classes="console-rag-settings-scope",
                markup=False,
            )
            with Horizontal(classes="console-rag-settings-sources"):
                for source_type in LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES:
                    label = _SOURCE_TYPE_LABELS.get(source_type, source_type)
                    yield Button(
                        console_rag_source_toggle_label(
                            source_type, source_type in self._source_types
                        ),
                        id=f"{CONSOLE_RAG_SOURCE_TOGGLE_ID_PREFIX}{source_type}",
                        classes=CONSOLE_RAG_SOURCE_TOGGLE_CLASS,
                        tooltip=f"Include {label} in Library retrieval.",
                    )
            with Horizontal(classes="console-rag-settings-actions"):
                yield Button(
                    "Search Library",
                    id="console-rag-settings-run",
                    variant="primary",
                    disabled=not self._can_run(self._query),
                )
                yield Button("Cancel", id="console-rag-settings-cancel")
            yield Static(
                "Enter runs the search. Esc or a click outside closes "
                "without changes.",
                classes="console-rag-settings-hint",
                markup=False,
            )

    def _current_query(self) -> str:
        return self.query_one("#console-rag-settings-query", Input).value

    def _can_run(self, query: str) -> bool:
        """Return whether retrieval can run: a query AND a source to read."""
        return bool(str(query or "").strip()) and bool(self._source_types)

    def _run_result(self) -> ConsoleRagSettingsResult:
        return ConsoleRagSettingsResult(
            query=self._current_query(),
            run=True,
            source_types=self._source_types,
        )

    def _refresh_run_availability(self, query: str | None = None) -> None:
        """Re-gate the Run action after a query edit or a source toggle."""
        run_button = self.query_one("#console-rag-settings-run", Button)
        run_button.disabled = not self._can_run(
            self._current_query() if query is None else query
        )

    def _toggle_source_type(self, source_type: str) -> None:
        """Flip one source kind in the selection and re-render what changed.

        Only the toggled button's label, the shared summary line, and the
        Run gate move -- no recompose, so the typed query survives.

        Args:
            source_type: The Library source-type identifier to flip.
        """
        if source_type not in LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES:
            return
        selected = set(self._source_types)
        selected.symmetric_difference_update({source_type})
        self._source_types = tuple(
            candidate
            for candidate in LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES
            if candidate in selected
        )
        toggle = self.query_one(
            f"#{CONSOLE_RAG_SOURCE_TOGGLE_ID_PREFIX}{source_type}", Button
        )
        toggle.label = console_rag_source_toggle_label(
            source_type, source_type in self._source_types
        )
        self.query_one("#console-rag-settings-scope", Static).update(
            self._scope_summary()
        )
        self._refresh_run_availability()

    @on(Input.Changed, "#console-rag-settings-query")
    def _sync_run_availability(self, event: Input.Changed) -> None:
        """Keep the Run action gated on a non-blank query."""
        event.stop()
        self._refresh_run_availability(str(event.value or ""))

    @on(Input.Submitted, "#console-rag-settings-query")
    def _submit_query(self, event: Input.Submitted) -> None:
        """Enter in the query input runs retrieval, matching the hint copy."""
        event.stop()
        if not self._can_run(self._current_query()):
            return
        self.dismiss(self._run_result())

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route source toggles; dismiss with the run result or no changes.

        Args:
            event: The pressed button.
        """
        button_id = event.button.id or ""
        if button_id.startswith(CONSOLE_RAG_SOURCE_TOGGLE_ID_PREFIX):
            event.stop()
            self._toggle_source_type(
                button_id[len(CONSOLE_RAG_SOURCE_TOGGLE_ID_PREFIX) :]
            )
            return
        if button_id == "console-rag-settings-run":
            event.stop()
            self.dismiss(self._run_result())
            return
        if button_id == "console-rag-settings-cancel":
            event.stop()
            await self.request_safe_cancel(source="button")
