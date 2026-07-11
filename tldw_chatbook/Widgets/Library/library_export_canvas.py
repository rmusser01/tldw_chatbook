"""Library export canvas: in-canvas chatbook export form (render-from-state)."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_export_state import (
    CHOOSE_DESTINATION_COPY,
    DESTINATION_PLACEHOLDER_COPY,
    EXPORT_BUTTON_COPY,
    EXPORT_HEADER_COPY,
    MEDIA_QUALITY_HELPER_COPY,
    LibraryExportFormState,
)


class LibraryExportCanvas(VerticalScroll):
    """Render the Library export canvas: scope summary + chatbook export form.

    ``VerticalScroll`` root (the L3a clipping lesson -- a plain ``Vertical``
    canvas clips content past the fold); every child is a stacked, full-
    width Button/Input/Static, mirroring ``LibraryIngestCanvas``. No
    ``Select`` -- the media-quality control is a cycle button (label
    ``"quality: {value} ▸"``) like the media canvas's type filter,
    since a plain ``Select`` widget did not render reliably in the
    deployed TUI (see ``handle_library_media_type_filter_pressed``).

    Attributes:
        state: The canvas's full display state (built by
            ``build_library_export_form_state``): the scope summary, the
            form echo, and the Export gate.
    """

    def __init__(self, state: LibraryExportFormState, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.styles.width = "1fr"
        self.styles.min_width = 40

    def compose(self) -> ComposeResult:
        state = self.state
        yield Static(
            EXPORT_HEADER_COPY,
            id="library-export-header",
            classes="destination-section",
            markup=False,
        )
        yield Static(
            state.scope_line,
            id="library-export-scope-line",
            markup=False,
        )
        # Always composed (display-toggled, never conditionally yielded):
        # the empty-scope helper is the one widget whose presence can
        # change when the counts worker lands, and counts landing must
        # update this canvas IN PLACE -- a recompose would destroy a form
        # Input mid-keystroke, dropping keyboard focus (the typed text
        # survives via the screen's form dict; focus does not). See
        # ``LibraryScreen._apply_library_export_counts``.
        empty_line = Static(
            state.empty_scope_line,
            id="library-export-empty-line",
            classes="library-export-quiet-line",
            markup=False,
        )
        empty_line.display = bool(state.empty_scope_line)
        yield empty_line
        yield Input(
            value=state.name,
            placeholder="Export name",
            id="library-export-name",
            classes="library-export-field",
        )
        yield Input(
            value=state.description,
            placeholder="Description (optional)",
            id="library-export-description",
            classes="library-export-field",
        )
        if state.show_media_fields:
            yield Button(
                f"quality: {state.media_quality} ▸",
                id="library-export-quality",
                classes="library-canvas-action",
                compact=True,
            )
            yield Static(
                MEDIA_QUALITY_HELPER_COPY,
                id="library-export-quality-helper",
                classes="library-export-quiet-line",
                markup=False,
            )
        yield Button(
            CHOOSE_DESTINATION_COPY,
            id="library-export-destination",
            classes="library-canvas-action",
            compact=True,
        )
        yield Static(
            state.destination or DESTINATION_PLACEHOLDER_COPY,
            id="library-export-destination-line",
            markup=False,
        )
        if state.overwrite_line:
            yield Static(
                state.overwrite_line,
                id="library-export-overwrite-line",
                classes="library-export-quiet-line",
                markup=False,
            )
        # Always composed (display-toggled, never conditionally yielded) --
        # same reasoning as the empty-scope helper above: a running export's
        # completion (success or failure) updates these two lines IN PLACE
        # (never a recompose), so both must already be mounted for that
        # targeted update to find them. See
        # ``LibraryScreen._update_library_export_canvas_after_run``.
        status_line = Static(
            state.status_line,
            id="library-export-status-line",
            classes="library-export-quiet-line",
            markup=False,
        )
        status_line.display = bool(state.status_line)
        yield status_line
        error_line = Static(
            state.error_line,
            id="library-export-error-line",
            classes="destination-purpose",
            markup=False,
        )
        error_line.display = bool(state.error_line)
        yield error_line
        yield Button(
            EXPORT_BUTTON_COPY,
            id="library-export-submit",
            classes="library-canvas-action",
            compact=True,
            disabled=not state.export_enabled,
        )
