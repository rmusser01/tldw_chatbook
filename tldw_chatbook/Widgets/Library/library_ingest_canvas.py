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


def _summarise_option(field: Any, value: Any) -> str:
    """Describe one option for the collapsed panel title, in plain language.

    The title used to be a dump of internal field names and repr'd values
    (``analyze=False, chunk=False, chunk_size=500, chunk_overlap=100``), which
    told a first-time user nothing about what any of it does.
    """
    if field.type == "checkbox":
        return f"{field.label}: {'on' if value else 'off'}"
    return f"{field.label}: {value}"


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

    class ParakeetInstallRequested(Message):
        """The user requested the curated Parakeet v2 installer."""

    def __init__(self, state: LibraryIngestCanvasState, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.styles.width = "1fr"
        self.styles.min_width = 40
        # Value each option widget was last rendered/reported with, keyed by
        # ``(group, field name)``. Seeded by ``_compose_type_group`` so that a
        # widget announcing the value we just gave it is recognised as mount
        # noise rather than a user edit -- see ``_handle_option_value_changed``.
        self._reported_option_values: dict[tuple[str, str], Any] = {}

    def _compose_type_group(
        self,
        group: str,
        cap: TypeGroupCapabilities,
        values: dict[str, Any],
        expanded: bool,
    ) -> Collapsible:
        """Build a collapsible options panel for one detected type group."""
        scope_label = f"Applies to all {cap.label} in this import."
        children: list[Any] = [Static(scope_label, classes="type-group-scope")]
        summary_parts: list[str] = []
        cap_fields_by_name = {f.name: f for f in cap.fields}

        for field in cap.fields:
            value = values.get(field.name, field.default)
            summary_parts.append(_summarise_option(field, value))
            # Two independent reasons a field can be uneditable: its tooling
            # is not installed, or the sibling field that gates it is off.
            disabled = field.depends_on is not None and not _is_installed(
                field.depends_on
            )
            if not disabled and field.enabled_when is not None:
                gate = cap_fields_by_name.get(field.enabled_when)
                gate_value = values.get(
                    field.enabled_when,
                    gate.default if gate is not None else False,
                )
                if field.enabled_when_values:
                    # A select gate: every non-empty choice is truthy, so the
                    # field must name the choices that actually enable it.
                    disabled = gate_value not in field.enabled_when_values
                else:
                    disabled = not bool(gate_value)
            widget_id = f"opt-{group}-{field.name}"

            if field.type == "checkbox":
                self._reported_option_values[(group, field.name)] = bool(value)
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
                self._reported_option_values[(group, field.name)] = select_value
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
                self._reported_option_values[(group, field.name)] = str(value)
                children.append(
                    Input(
                        value=str(value),
                        placeholder=field.label,
                        id=widget_id,
                        disabled=disabled,
                    )
                )

        if group == "audio_video":
            provider = cap_fields_by_name["transcription_provider"]
            provider_value = values.get(
                "transcription_provider", provider.default
            )
            children.append(
                Button(
                    "Install verified Parakeet v2 INT8 (630.6 MiB)…",
                    id="opt-audio_video-install-parakeet-v2",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=provider_value != "parakeet-onnx",
                )
            )

        children.append(
            Button(
                "Reset to defaults",
                id=f"opt-{group}-reset",
                classes="library-canvas-action library-ingest-option-reset",
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
        # State before action: "Imports run on this machine." then the button
        # that changes it. Rendering the button first read as a contradiction
        # top-to-bottom -- "Import on the server / Imports run on this machine"
        # (spotted on screen, not in a test).
        if state.server_quiet_line:
            yield Static(
                state.server_quiet_line,
                id="library-ingest-server-line",
                classes="library-ingest-quiet-line",
                markup=False,
            )
        if state.show_backend_switch:
            # Only offered when a server is actually configured; otherwise there
            # is no choice to make and a dead toggle would be worse than none.
            yield Button(
                "Import on this machine"
                if state.ingest_backend == "server"
                else "Import on the server",
                id="library-ingest-backend-switch",
                classes="library-canvas-action",
                compact=True,
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
        with Horizontal(classes="library-ingest-path-actions"):
            yield Button(
                "Browse…",
                id="library-ingest-browse",
                classes="library-canvas-action",
                compact=True,
            )
            if state.show_clear_path:
                yield Button(
                    "Clear",
                    id="library-ingest-clear-path",
                    classes="library-canvas-action",
                    compact=True,
                )
        for index, line in enumerate(state.intro_lines):
            yield Static(
                line,
                id=f"library-ingest-intro-{index}",
                classes="library-ingest-quiet-line",
                markup=False,
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
                if state.errors_are_path_problem:
                    # Re-running the same analysis on the same bad path fails
                    # identically; the useful action is picking a real one.
                    yield Button(
                        "Choose a file…",
                        id="ingest-preflight-choose",
                        classes="library-canvas-action",
                        compact=True,
                    )
                else:
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
                or row.can_open_on_server
                or row.can_retry
                or row.can_dismiss
                or row.can_cancel
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
                    if row.can_open_on_server:
                        # Its own action rather than a reworded "Open in
                        # Library": that one resolves a LOCAL media row, and a
                        # server ingest has none. The label says where the
                        # content actually is.
                        yield Button(
                            "View on server",
                            # Neither the id prefix nor the class may collide
                            # with "Open in Library": that handler matches
                            # ``.library-ingest-open`` and strips the prefix
                            # ``library-ingest-open-`` to recover a job id, so
                            # an id of ``library-ingest-open-server-<job>``
                            # would be caught by it and parsed into the bogus
                            # job id ``server-<job>``.
                            id=f"library-ingest-view-server-{row.job_id}",
                            classes=(
                                "library-canvas-action "
                                "library-ingest-view-server "
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
                    if row.can_cancel:
                        yield Button(
                            "Cancel",
                            id=f"library-ingest-cancel-{row.job_id}",
                            classes=(
                                "library-canvas-action library-ingest-cancel "
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
        """Bubble a genuine option edit up as a message.

        Textual posts ``Changed`` when a ``Select`` mounts, and when an
        ``Input`` mounts with a non-empty ``value=``. Those announce the value
        this canvas just handed the widget, not a user edit, so they are
        dropped here: forwarding them made the screen recompose, which
        remounted the widgets, which posted again -- an unbounded recompose
        cycle that pinned the UI at 100% CPU for every pdf/audio/ebook
        pre-flight (task-673). Comparing against the last value we rendered
        *or* forwarded (rather than a "still mounting" flag) keeps this free
        of event-ordering assumptions, and still lets a user return a field
        to its original value: the previous edit updated the record.
        """
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
        key = (group, name)
        if key in self._reported_option_values and (
            self._reported_option_values[key] == event.value
        ):
            return
        self._reported_option_values[key] = event.value
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

    @on(Button.Pressed, "#opt-audio_video-install-parakeet-v2")
    def _request_parakeet_v2_install(self, event: Button.Pressed) -> None:
        """Request explicit install confirmation from the owning screen."""
        event.stop()
        self.post_message(self.ParakeetInstallRequested())
