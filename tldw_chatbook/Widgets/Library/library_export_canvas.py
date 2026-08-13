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
    MEDIA_QUALITY_OPTIONS,
    LibraryExportFormState,
    export_button_tooltip,
    media_quality_helper_copy,
)
from tldw_chatbook.Library.library_shell_state import (
    library_choice_label,
    library_choice_tooltip,
    library_disabled_action_label,
)
from tldw_chatbook.Widgets.Library.library_choice_strip import (
    compose_library_choice_strip,
)
from tldw_chatbook.Widgets.Library.library_canvas_sync import (
    PostRecomposeCallback,
)


def apply_library_export_submit_gate(
    submit_button: Button, state: LibraryExportFormState
) -> None:
    """Apply the Export submit gate: disabled, tooltip, AND marker label.

    task-4023 AC#1 (RC-07): the disabled marker lives in the label, so
    every code path that flips ``disabled`` must rebuild the label too --
    ``compose`` below plus the screen's two in-place patchers
    (``_apply_library_export_counts`` / the completion updater), which
    deliberately avoid recomposing the canvas. One shared helper keeps
    the three call sites from drifting (the recompose-discipline rule:
    any conditional a compose branch owns, the in-place updater must own
    too).

    Args:
        submit_button: The mounted ``#library-export-submit`` Button.
        state: The current export form state.

    Returns:
        None.
    """
    submit_button.disabled = not state.export_enabled
    submit_button.label = library_disabled_action_label(
        EXPORT_BUTTON_COPY, submit_button.disabled
    )
    # task-2858 AC#3 (LIB-11): F-018 -- the tooltip always explains either
    # what pressing Export will do or the SAME blocker ``disabled``
    # reflects.
    submit_button.tooltip = export_button_tooltip(state)

class LibraryExportCanvas(PostRecomposeCallback, VerticalScroll):
    """Render the Library export canvas: scope summary + chatbook export form.

    ``VerticalScroll`` root (the L3a clipping lesson -- a plain ``Vertical``
    canvas clips content past the fold); every child is a stacked, full-
    width Button/Input/Static, mirroring ``LibraryIngestCanvas``. No
    ``Select`` -- the media-quality control opens a one-row choice strip
    (task-14902, ``library_choice_strip.py``) with ``✓`` on the active
    option, like the media canvas's type filter, since a plain ``Select``
    widget did not render reliably in the deployed TUI.

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

    def sync_state(self, state: LibraryExportFormState) -> None:
        """Rebuild only the mounted export canvas from a complete snapshot.

        Args:
            state: Complete export form state to render.
        """
        self.state = state
        self.refresh(recompose=True)

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
                # task-14902: a chooser-opener -- press opens the
                # direct-pick strip below (the per-press cycle retired).
                library_choice_label("quality", state.media_quality),
                id="library-export-quality",
                classes="library-canvas-action",
                compact=True,
                tooltip=library_choice_tooltip(
                    "media quality", MEDIA_QUALITY_OPTIONS
                ),
            )
            if state.quality_choices_visible:
                # Unlike the list canvases, the opener stays visible (the
                # form has vertical room and the "quality:" label anchors
                # what the strip's bare values mean) -- so a second press
                # on the opener also closes the strip.
                yield from compose_library_choice_strip(
                    strip_id="library-export-quality-choices",
                    choice_class="library-export-quality-choice",
                    options=tuple(
                        (f"library-export-quality-{value}", value, value)
                        for value in MEDIA_QUALITY_OPTIONS
                    ),
                    active_value=state.media_quality,
                )
            yield Static(
                media_quality_helper_copy(state.media_quality),
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
        # task-2858 AC#3 (LIB-12): the durable "Last export: ..." receipt --
        # always mounted (display-toggled), same discipline as the three
        # quiet lines above, since the screen's in-place completion
        # handler (``_update_library_export_canvas_after_run``) patches
        # this widget's text after a run finishes without a recompose.
        last_export_line = Static(
            state.last_export_line,
            id="library-export-last-line",
            classes="library-export-quiet-line",
            markup=False,
        )
        last_export_line.display = bool(state.last_export_line)
        yield last_export_line
        submit_button = Button(
            EXPORT_BUTTON_COPY,
            id="library-export-submit",
            classes="library-canvas-action",
            compact=True,
        )
        # Disabled + marker label + F-018 reason tooltip in one place,
        # shared with the screen's in-place patchers (see
        # ``apply_library_export_submit_gate``).
        apply_library_export_submit_gate(submit_button, state)
        yield submit_button
        cancel_button = Button(
            "Cancel",
            id="library-export-cancel",
            classes="library-canvas-action",
            compact=True,
        )
        cancel_button.display = bool(state.running)
        yield cancel_button
