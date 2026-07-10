"""Library ingest canvas: local-file ingest form + job queue (render-from-state)."""

from __future__ import annotations

from typing import Any

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Button, Collapsible, Input, Static

from tldw_chatbook.Library.library_ingest_state import (
    QUEUE_EMPTY_COPY,
    LibraryIngestCanvasState,
)


def _toggle_label(*, enabled: bool, text: str) -> str:
    """Return a toggle Button's visible label, ``✓``/``○`` convention."""
    marker = "✓" if enabled else "○"
    return f"{marker} {text}"


class LibraryIngestCanvas(VerticalScroll):
    """Render the Library ingest canvas: the local-file ingest form and its job queue.

    ``VerticalScroll`` root (the L3a clipping lesson -- a plain ``Vertical``
    canvas clips content past the fold); every child is a stacked,
    full-width Button/Input/Static, mirroring ``LibraryNotesCanvas``'s sync
    panel. No ``Select``, and no ``Horizontal`` mixing a 1fr sibling with
    fixed-width action buttons (the known non-rendering failure mode for
    this canvas family).

    Attributes:
        state: The canvas's full display state (built by
            ``build_library_ingest_state``): the form echo, the Start
            gate, and the queue rows to render.
    """

    def __init__(self, state: LibraryIngestCanvasState, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.styles.width = "1fr"
        self.styles.min_width = 40

    def compose(self) -> ComposeResult:
        state = self.state
        yield Static(
            state.header,
            id="library-ingest-header",
            classes="destination-section",
            markup=False,
        )
        if state.server_quiet_line:
            yield Static(
                state.server_quiet_line,
                id="library-ingest-server-line",
                classes="library-ingest-quiet-line",
                markup=False,
            )
        if state.unavailable_line:
            yield Static(
                state.unavailable_line,
                id="library-ingest-unavailable-line",
                classes="library-ingest-quiet-line",
                markup=False,
            )
        yield Input(
            value=state.form.path,
            placeholder="Path to a local file…",
            id="library-ingest-path",
            classes="library-ingest-field",
        )
        yield Button(
            "Browse…",
            id="library-ingest-browse",
            classes="library-canvas-action",
            compact=True,
        )
        yield Input(
            value=state.form.title,
            placeholder="Title",
            id="library-ingest-title",
            classes="library-ingest-field",
        )
        yield Input(
            value=state.form.author,
            placeholder="Author",
            id="library-ingest-author",
            classes="library-ingest-field",
        )
        yield Input(
            value=state.form.keywords,
            placeholder="Keywords (comma-separated)",
            id="library-ingest-keywords",
            classes="library-ingest-field",
        )
        with Collapsible(
            title="Advanced options",
            # Renders from the form echo's `advanced_open` field (not a
            # hardcoded True) so a recompose while the panel is expanded --
            # the analyze/chunk toggle handlers' own, or a registry-listener
            # -driven one -- never re-collapses it out from under the user.
            # The screen's `Collapsible.Toggled` handler keeps this field in
            # sync with the live widget's `collapsed` reactive (mirrors the
            # `#library-rag-history` collapsible's own state-sync pattern).
            collapsed=not state.form.advanced_open,
            id="library-ingest-advanced",
        ):
            yield Button(
                _toggle_label(enabled=state.form.analyze, text="Analyze after ingest"),
                id="library-ingest-analyze-toggle",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                _toggle_label(enabled=state.form.chunk, text="Chunk content"),
                id="library-ingest-chunk-toggle",
                classes="library-canvas-action",
                compact=True,
            )
            yield Input(
                value=state.form.chunk_size,
                id="library-ingest-chunk-size",
                classes="library-ingest-field",
            )
        yield Button(
            "Start ingest",
            id="library-ingest-start",
            classes="library-canvas-action",
            compact=True,
            disabled=not state.start_enabled,
        )
        yield Static(
            state.queue_heading,
            id="library-ingest-queue-heading",
            classes="destination-section",
            markup=False,
        )
        yield Static(
            state.queue_counts_line,
            id="library-ingest-queue-counts",
            markup=False,
        )
        if not state.queue_rows:
            yield Static(
                QUEUE_EMPTY_COPY,
                id="library-ingest-queue-empty",
                markup=False,
            )
            return
        for index, row in enumerate(state.queue_rows):
            # A source filename can contain Rich markup syntax (e.g. a
            # literal "[/bracket]" in the name) -- escape_markup here is
            # what keeps a hostile filename from raising MarkupError at
            # mount time (the L3a lesson; mirrors
            # ``library_rag_history_children``'s escaped Button labels).
            yield Static(
                escape_markup(row.line),
                id=f"library-ingest-row-{index}",
                classes="library-ingest-row",
            )
            if row.can_open:
                yield Button(
                    "Open in Library",
                    id=f"library-ingest-open-{index}",
                    classes="library-canvas-action library-ingest-open",
                    compact=True,
                )
            if row.can_retry:
                yield Button(
                    "Retry",
                    id=f"library-ingest-retry-{index}",
                    classes="library-canvas-action library-ingest-retry",
                    compact=True,
                )
