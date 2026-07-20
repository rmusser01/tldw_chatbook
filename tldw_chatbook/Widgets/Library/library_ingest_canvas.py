"""Library ingest canvas: local-file ingest form + job queue (render-from-state)."""

from __future__ import annotations

from typing import Any

from rich.markup import escape as escape_markup
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Checkbox, Collapsible, Input, Select, Static

from tldw_chatbook.Library.ingest_capabilities import (
    TypeGroupCapabilities,
    _is_installed,
    get_capabilities,
)
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
    canvas clips content past the fold); every child is stacked full-width,
    mirroring ``LibraryNotesCanvas``'s sync panel. Per-type option panels
    are rendered from ``ingest_capabilities.py`` schemas and post messages
    for all state changes so the screen can persist them.
    """

    class OptionValueChanged(Message):
        """A per-type option value changed."""

        def __init__(self, group: str, name: str, value: Any) -> None:
            super().__init__()
            self.group = group
            self.name = name
            self.value = value

    class OptionPanelToggled(Message):
        """A per-type options panel was expanded or collapsed."""

        def __init__(self, group: str, expanded: bool) -> None:
            super().__init__()
            self.group = group
            self.expanded = expanded

    def __init__(self, state: LibraryIngestCanvasState, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.styles.width = "1fr"
        self.styles.min_width = 40

    def _compose_type_group(
        self,
        group: str,
        cap: TypeGroupCapabilities,
        values: dict[str, Any],
        expanded: bool,
    ) -> Collapsible:
        """Build a collapsible options panel for one detected type group."""
        scope_label = f"These options apply to all {cap.label} files in this selection."
        children: list[Any] = [Static(scope_label, classes="type-group-scope")]
        summary_parts: list[str] = []

        for field in cap.fields:
            value = values.get(field.name, field.default)
            summary_parts.append(f"{field.name}={value}")
            disabled = field.depends_on is not None and not _is_installed(field.depends_on)
            widget_id = f"opt-{group}-{field.name}"

            if field.type == "checkbox":
                children.append(
                    Checkbox(
                        field.label,
                        value=bool(value),
                        id=widget_id,
                        disabled=disabled,
                    )
                )
            elif field.type == "select":
                select_options = [(opt, opt) for opt in field.options]
                select_value = value if value in field.options else field.default
                if select_value not in field.options and field.options:
                    select_value = field.options[0]
                children.append(
                    Select(
                        select_options,
                        value=select_value,
                        id=widget_id,
                        disabled=disabled,
                        allow_blank=False,
                    )
                )
            else:
                children.append(
                    Input(
                        value=str(value),
                        placeholder=field.label,
                        id=widget_id,
                        disabled=disabled,
                    )
                )

        children.append(
            Button(
                "Reset to defaults",
                id=f"opt-{group}-reset",
                classes="library-canvas-action",
                compact=True,
            )
        )

        panel = Vertical(*children, classes="type-group-contents")
        title = f"{cap.label} — {', '.join(summary_parts)}"
        return Collapsible(
            panel,
            title=title,
            collapsed=not expanded,
            id=f"type-group-{group}",
        )

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
            placeholder="Path to a local file or a URL…",
            id="library-ingest-path",
            classes="library-ingest-field",
        )
        yield Button(
            "Browse…",
            id="library-ingest-browse",
            classes="library-canvas-action",
            compact=True,
        )
        # Pre-flight summary replaces the old always-visible supported-types
        # line. All copy is taken straight from ``state``; this widget stays
        # render-only and does not compute pre-flight results itself.
        if state.preflight_checking:
            yield Static(
                "Checking…",
                id="ingest-preflight-status",
                classes="library-ingest-quiet-line",
                markup=False,
            )
        else:
            if state.errors:
                for index, error in enumerate(state.errors):
                    yield Static(
                        escape_markup(error),
                        id=f"ingest-preflight-error-{index}",
                        classes="library-ingest-quiet-line",
                    )
                yield Button(
                    "Retry",
                    id="ingest-preflight-retry",
                    classes="library-canvas-action",
                    compact=True,
                )
            if state.warning_lines:
                for index, warning in enumerate(state.warning_lines):
                    yield Static(
                        f"⚠ {escape_markup(warning)}",
                        id=f"ingest-preflight-warning-{index}",
                        classes="library-ingest-quiet-line",
                    )
            if state.type_breakdown_line:
                yield Static(
                    state.type_breakdown_line,
                    id="ingest-type-breakdown",
                    classes="library-ingest-quiet-line",
                    markup=False,
                )
            if state.estimate_line:
                yield Static(
                    state.estimate_line,
                    id="ingest-estimate",
                    classes="library-ingest-quiet-line",
                    markup=False,
                )
            if state.unsupported_files:
                count = len(state.unsupported_files)
                file_noun = "file" if count == 1 else "files"
                failure_noun = "failure" if count == 1 else "failures"
                yield Static(
                    f"{count} unsupported {file_noun} will be recorded as a {failure_noun}.",
                    id="ingest-unsupported-summary",
                    classes="library-ingest-quiet-line",
                    markup=False,
                )
            if state.type_groups:
                with Horizontal(classes="library-ingest-options-bulk"):
                    yield Button(
                        "Expand all",
                        id="ingest-expand-all",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    yield Button(
                        "Collapse all",
                        id="ingest-collapse-all",
                        classes="library-canvas-action",
                        compact=True,
                    )
                for group in state.type_groups:
                    cap = get_capabilities(group)
                    values = state.form.type_options.get(group, {})
                    expanded = group in state.expanded_type_groups
                    yield self._compose_type_group(group, cap, values, expanded)
        yield Input(
            value=state.form.title,
            placeholder="Title (optional)",
            id="library-ingest-title",
            classes="library-ingest-field",
        )
        yield Input(
            value=state.form.author,
            placeholder="Author (optional)",
            id="library-ingest-author",
            classes="library-ingest-field",
        )
        yield Input(
            value=state.form.keywords,
            placeholder="Keywords, comma-separated (optional)",
            id="library-ingest-keywords",
            classes="library-ingest-field",
        )
        # Always mounted, even with empty text, so the Start button never
        # shifts vertically when the gate line's copy appears/disappears
        # (2026-07 UAT: the button jumped ~2 rows on every gate change,
        # breaking muscle memory). The fixed inline height reserves the
        # line's row when the text is empty (an auto-height empty Static
        # would collapse to 0); the screen's path-changed handler updates
        # the text in place instead of mounting/removing the widget.
        start_quiet_line = Static(
            state.start_quiet_line,
            id="library-ingest-start-quiet-line",
            classes="library-ingest-quiet-line",
            markup=False,
        )
        start_quiet_line.styles.height = 1
        yield start_quiet_line
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
        if state.queue_counts_line:
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
        for index, row in enumerate(state.queue_rows):
            # A source filename can contain Rich markup syntax (e.g. a
            # literal "[/bracket]" in the name) -- escape_markup here is
            # what keeps a hostile filename from raising MarkupError at
            # mount time (the L3a lesson; mirrors
            # ``library_rag_history_children``'s escaped Button labels).
            row_classes = "library-ingest-row"
            has_actions = (
                row.can_open
                or row.can_retry
                or row.can_dismiss
                or bool(row.error_detail)
            )
            if has_actions:
                # A row with action buttons below it gets its own
                # bottom-margin trimmed to 0 (A3) -- the actions row's own
                # ``.library-ingest-row-actions`` margin supplies the "tight
                # gap above, blank line below" spacing instead, so the
                # button(s) read as belonging to THIS row rather than the
                # one below it. Plain rows (queued/running, or a done row
                # with no action) keep their own margin for row-to-row
                # spacing.
                row_classes += " library-ingest-row-with-actions"
            yield Static(
                escape_markup(row.line),
                id=f"library-ingest-row-{index}",
                classes=row_classes,
            )
            if row.progress:
                progress_line = row.progress.get("message") if row.progress else ""
                yield Static(
                    f"{row.state.value} {progress_line}",
                    id=f"library-ingest-progress-{row.job_id}",
                    classes="library-ingest-progress",
                    markup=False,
                )
            # Row-action buttons are keyed by the job's registry-assigned
            # ``job_id`` (e.g. ``"library-ingest-open-ingest-job-3"``), NOT
            # by ``index`` -- unlike the row Static above, these ARE click
            # targets, and the registry mutates asynchronously (runner
            # completions, retry-supersede, new submissions) between a
            # render and a click. An index-keyed id can silently point at a
            # different job by the time it's pressed; a job_id-keyed one
            # can't, because the screen's handlers resolve the job by id
            # from the live registry rather than by re-indexing a rebuilt
            # snapshot (see the PR #591 review's F1 finding).
            #
            # (L5, fix batch F1b) A row's action buttons (Open in Library /
            # Retry / Dismiss -- never more than one of "Open in Library"
            # or the Retry+Dismiss pair applies to the same row, since
            # can_open is DONE-only and can_retry/can_dismiss are
            # FAILED-only) are wrapped in one ``Horizontal`` so a failed
            # row's Retry and Dismiss sit on one line instead of stacking
            # vertically. Both children here are fixed-width compact
            # Buttons -- never a 1fr sibling mixed with a fixed-width one,
            # the known non-rendering failure mode for this canvas family
            # (see the class docstring).
            if has_actions:
                with Horizontal(classes="library-ingest-row-actions"):
                    if row.can_open:
                        yield Button(
                            "Open in Library",
                            id=f"library-ingest-open-{row.job_id}",
                            classes=(
                                "library-canvas-action library-ingest-open "
                                "library-ingest-row-action"
                            ),
                            compact=True,
                        )
                    if row.error_detail:
                        yield Button(
                            "Show details",
                            id=f"library-ingest-details-{row.job_id}",
                            classes=(
                                "library-canvas-action library-ingest-details "
                                "library-ingest-row-action"
                            ),
                            compact=True,
                        )
                    if row.can_retry:
                        yield Button(
                            "Retry",
                            id=f"library-ingest-retry-{row.job_id}",
                            classes=(
                                "library-canvas-action library-ingest-retry "
                                "library-ingest-row-action"
                            ),
                            compact=True,
                        )
                    if row.can_dismiss:
                        yield Button(
                            "Dismiss",
                            id=f"library-ingest-dismiss-{row.job_id}",
                            classes=(
                                "library-canvas-action library-ingest-dismiss "
                                "library-ingest-row-action"
                            ),
                            compact=True,
                        )
        if state.queue_show_clear_finished:
            yield Button(
                "Clear finished",
                id="library-ingest-clear-finished",
                classes="library-canvas-action",
                compact=True,
            )
        with Collapsible(
            title="Recent ingests", collapsed=True, id="library-ingest-recent"
        ):
            for job in state.recent_jobs:
                yield Static(
                    f"{escape_markup(job.source_path)} — {job.state.value}",
                    classes="library-ingest-recent-item",
                    markup=False,
                )

    @on(Checkbox.Changed)
    @on(Select.Changed)
    @on(Input.Changed)
    def _handle_option_value_changed(
        self,
        event: Checkbox.Changed | Select.Changed | Input.Changed,
    ) -> None:
        """Parse an option widget id and bubble the change up as a message."""
        widget = getattr(
            event,
            "checkbox",
            getattr(event, "select", getattr(event, "input", None)),
        )
        if widget is None:
            return
        widget_id = widget.id
        if not widget_id or not widget_id.startswith("opt-"):
            return
        parts = widget_id.split("-")
        if len(parts) < 3 or parts[0] != "opt":
            return
        group = parts[1]
        name = "-".join(parts[2:])
        if name == "reset":
            return
        self.post_message(self.OptionValueChanged(group, name, event.value))

    @on(Collapsible.Expanded)
    @on(Collapsible.Collapsed)
    def _handle_option_panel_toggled(
        self,
        event: Collapsible.Expanded | Collapsible.Collapsed,
    ) -> None:
        """Parse a type-group panel id and bubble expand/collapse up as a message."""
        collapsible = event.collapsible
        widget_id = collapsible.id
        if not widget_id or not widget_id.startswith("type-group-"):
            return
        group = widget_id[len("type-group-"):]
        self.post_message(
            self.OptionPanelToggled(group, expanded=isinstance(event, Collapsible.Expanded))
        )
