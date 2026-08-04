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
from tldw_chatbook.Library.library_ingest_jobs import IngestJobState
from tldw_chatbook.Library.library_ingest_state import (
    validate_ingest_option_value,
    LibraryIngestCanvasState,
    build_intro_lines,
)


class LibraryIngestPreflightSummary(Vertical):
    """Render-from-state pre-flight summary block (task-2042).

    Its own widget so the screen can recompose JUST these lines when a
    pre-flight result lands: recomposing the whole canvas remounted the
    Start/Browse buttons between a mouse-down and its mouse-up, silently
    swallowing the first click after typing.
    """

    def __init__(self, state: LibraryIngestCanvasState, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.styles.width = "1fr"
        self.styles.height = "auto"

    def compose(self) -> ComposeResult:
        state = self.state
        if state.preflight_checking:
            yield Static(
                "Checking…",
                id="ingest-preflight-status",
                classes="library-ingest-quiet-line",
                markup=False,
            )
            return
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
        if state.unsupported_line:
            yield Static(
                state.unsupported_line,
                id="ingest-unsupported-summary",
                classes="library-ingest-quiet-line",
                markup=False,
            )
        if state.duplicate_line:
            yield Static(
                state.duplicate_line,
                id="ingest-duplicate-summary",
                classes="library-ingest-quiet-line",
                markup=False,
            )


class LibraryIngestQueuePanel(Vertical):
    """Render-from-state queue block: counts, rows, actions, clear, recent.

    Its own widget so registry job ticks recompose ONLY the queue (task-2042):
    the whole-canvas recompose they used to trigger remounted the form
    widgets (swallowing in-flight clicks) and snapped the canvas scroll.
    """

    def __init__(self, state: LibraryIngestCanvasState, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.styles.width = "1fr"
        self.styles.height = "auto"

    def compose(self) -> ComposeResult:
        state = self.state
        if state.queue_counts_line:
            yield Static(
                state.queue_counts_line,
                id="library-ingest-queue-counts",
                markup=False,
            )
        if not state.queue_rows and state.queue_empty_line:
            yield Static(
                state.queue_empty_line,
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
            stt_actions = _stt_recovery_actions(row.error_detail)
            has_actions = (
                row.can_open
                or row.can_open_on_server
                or row.can_retry
                or row.can_dismiss
                or row.can_cancel
                or bool(row.error_detail)
                or bool(stt_actions)
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
                # (task-2016) The row line above already carries the terminal
                # state ("✓ done · …"); repeating it here read as stuttering.
                # Active states keep the prefix -- their progress message is
                # stage detail, not an outcome.
                terminal = row.state in (
                    IngestJobState.DONE,
                    IngestJobState.FAILED,
                    IngestJobState.CANCELLED,
                )
                yield Static(
                    progress_line
                    if terminal
                    else f"{row.state.value} {progress_line}",
                    id=f"library-ingest-progress-{row.job_id}",
                    classes="library-ingest-progress",
                    markup=False,
                )
            if row.details_expanded and row.detail_lines:
                for line_index, detail_line in enumerate(row.detail_lines):
                    yield Static(
                        detail_line,
                        id=(
                            f"library-ingest-detail-{row.job_id}-{line_index}"
                        ),
                        classes="library-ingest-detail-line",
                        markup=False,
                    )
            # Row-action buttons are keyed by the job's registry-assigned
            # ``job_id`` -- these ARE click targets and the registry mutates
            # asynchronously between a render and a click; an index-keyed id
            # can silently point at a different job by the time it's pressed
            # (PR #591 review, F1). One Horizontal per row so a failed row's
            # actions sit on one line (L5, F1b).
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
                        # content actually is. The id prefix must not collide
                        # with ``library-ingest-open-`` (that handler strips
                        # the prefix to recover a job id).
                        yield Button(
                            "View on server",
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
                            "Hide details"
                            if row.details_expanded
                            else "Show details",
                            id=f"library-ingest-details-{row.job_id}",
                            classes=(
                                "library-canvas-action library-ingest-details "
                                "library-ingest-row-action"
                            ),
                            compact=True,
                        )
                    if "choose_another_gguf" in stt_actions:
                        yield Button(
                            "Choose another GGUF…",
                            id=f"library-ingest-choose-gguf-{row.job_id}",
                            classes=(
                                "library-canvas-action library-ingest-choose-gguf "
                                "library-ingest-row-action"
                            ),
                            compact=True,
                        )
                    if "retry_faster_whisper" in stt_actions:
                        yield Button(
                            "Retry with faster-whisper",
                            id=(
                                "library-ingest-retry-faster-whisper-"
                                f"{row.job_id}"
                            ),
                            classes=(
                                "library-canvas-action "
                                "library-ingest-retry-faster-whisper "
                                "library-ingest-row-action"
                            ),
                            compact=True,
                        )
                    if row.can_retry and not stt_actions:
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
                state.queue_clear_finished_label,
                id="library-ingest-clear-finished",
                classes="library-canvas-action",
                compact=True,
            )
        # (task-2100) Hidden when empty: after a clear it expanded to an
        # unlabeled empty shell (round-3 critique; deliberately flips the
        # earlier always-visible contract on that evidence).
        if state.recent_jobs:
            with Collapsible(
                title="Recent ingests",
                collapsed=True,
                id="library-ingest-recent",
            ):
                for job in state.recent_jobs:
                    yield Static(
                        f"{escape_markup(job.source_path)} — {job.state.value}",
                        classes="library-ingest-recent-item",
                        markup=False,
                    )

_STT_RECOVERY_ACTIONS = frozenset(
    {"choose_another_gguf", "retry_faster_whisper"}
)


def _stt_recovery_actions(error_detail: dict[str, Any] | None) -> frozenset[str]:
    """Return only the bounded STT recovery actions implemented here."""
    if not error_detail or error_detail.get("category") != "stt_failure":
        return frozenset()
    actions = error_detail.get("actions")
    if not isinstance(actions, list):
        return frozenset()
    return frozenset(
        action
        for action in actions
        if isinstance(action, str) and action in _STT_RECOVERY_ACTIONS
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


def ingest_scope_label(cap: TypeGroupCapabilities, has_files: bool) -> str:
    """Scope-line copy for one options panel.

    Shared by ``_compose_type_group`` and the screen's in-place dynamic
    update (task-2042 review): per-group file counts change without the
    group SET changing, so the label must be updatable without a panel
    recompose -- one source keeps the two paths from drifting.

    Args:
        cap: The group's capability schema (supplies the display label).
        has_files: Whether the current pre-flight staged files for it.

    Returns:
        The scope sentence for the panel.
    """
    return (
        f"Applies to all {cap.label} in this import."
        if has_files
        else f"Applies to {cap.label} if this import contains any."
    )


class StateGlyphCheckbox(Checkbox):
    """Checkbox whose glyph carries on/off without color (task-2043).

    Stock ``ToggleButton`` renders ``BUTTON_INNER = "X"`` for BOTH states --
    on/off is a color change only, invisible in monochrome. The renderer
    reads ``self.BUTTON_INNER``, so a per-instance shadow tracked in
    ``watch_value`` gives a glyph-level state: ``✓`` checked, blank not.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.BUTTON_INNER = "✓" if self.value else " "

    def watch_value(self) -> None:
        self.BUTTON_INNER = "✓" if self.value else " "
        super().watch_value()


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

    class TranscribeCppGGUFRequested(Message):
        """The user requested a local transcribe.cpp GGUF picker."""

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
        has_files: bool = True,
    ) -> Collapsible:
        """Build a collapsible options panel for one detected type group."""
        # (task-2016) The generic panel is always rendered so global options
        # stay reachable -- but claiming "Applies to all X in this import."
        # with zero such files staged was a false statement.
        scope_label = ingest_scope_label(cap, has_files)
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
                    StateGlyphCheckbox(
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
                # (task-2043) Selects missed task-2012's labeling pass: a
                # bare "pymupdf4llm" carries no meaning on its own.
                children.append(
                    Static(
                        field.label,
                        classes="type-group-field-label",
                        markup=False,
                    )
                )
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
                # A populated Input never shows its placeholder, so
                # placeholder-as-label left values like "1000" with no
                # visible meaning (task-2012). The label gets its own line.
                children.append(
                    Static(
                        field.label,
                        classes="type-group-field-label",
                        markup=False,
                    )
                )
                children.append(
                    Input(
                        value=str(value),
                        placeholder=field.label,
                        id=widget_id,
                        disabled=disabled,
                    )
                )
                # (task-2130) Inline validation message -- a text line, not a
                # color-only border. Display-managed so typing updates it in
                # place without recomposing the panel.
                error_message = validate_ingest_option_value(field, value)
                error_line = Static(
                    error_message,
                    id=f"{widget_id}-error",
                    classes="type-group-field-error",
                    markup=False,
                )
                error_line.display = bool(error_message)
                children.append(error_line)

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
            if provider_value == "transcribe-cpp":
                configured = self.state.transcribe_cpp_configured
                children.append(
                    Static(
                        "Local GGUF configured."
                        if configured
                        else "No local GGUF configured.",
                        id="opt-audio_video-transcribe-cpp-status",
                        classes="type-group-scope",
                        markup=False,
                    )
                )
                children.append(
                    Button(
                        "Choose another GGUF…" if configured else "Choose GGUF…",
                        id="opt-audio_video-choose-transcribe-cpp-gguf",
                        classes="library-canvas-action",
                        compact=True,
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
            # (task-2042) Always mounted, shown/hidden via ``display`` so a
            # path appearing/disappearing never changes the canvas's widget
            # STRUCTURE -- structural changes force a full recompose, and a
            # full recompose in the type-then-click window swallows the
            # click in flight.
            clear_button = Button(
                "Clear",
                id="library-ingest-clear-path",
                classes="library-canvas-action",
                compact=True,
            )
            clear_button.display = state.show_clear_path
            yield clear_button
        # (task-2016/2042) Intro lines: always mounted, ``display``-managed
        # (the screen's typing handler toggles them live; keeping them in
        # the tree keeps intro transitions non-structural).
        for index, line in enumerate(build_intro_lines()):
            intro = Static(
                line,
                id=f"library-ingest-intro-{index}",
                classes="library-ingest-quiet-line library-ingest-intro",
                markup=False,
            )
            intro.display = bool(state.intro_lines)
            yield intro
        # Pre-flight summary lines live in their own render-from-state child
        # so a result landing recomposes ONLY them (task-2042).
        yield LibraryIngestPreflightSummary(
            state, id="library-ingest-preflight-summary"
        )
        # (task-2016) Bulk expand/collapse over exactly one panel is
        # noise -- the generic panel is always appended, so a single
        # entry means there is nothing to expand "all" of. Panels render
        # regardless of ``preflight_checking`` (a re-analysis lasts well
        # under a second; hiding them made the checking flag structural,
        # forcing exactly the full recompose task-2042 removes).
        if len(state.type_groups) > 1:
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
        if state.type_groups:
            for group in state.type_groups:
                cap = get_capabilities(group)
                values = state.form.type_options.get(group, {})
                expanded = group in state.expanded_type_groups
                yield self._compose_type_group(
                    group,
                    cap,
                    values,
                    expanded,
                    has_files=bool(state.type_group_file_counts.get(group)),
                )
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
        if state.commit_summary_line:
            yield Static(
                state.commit_summary_line,
                id="library-ingest-commit-summary",
                classes="library-ingest-quiet-line",
                markup=False,
            )
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
        # Queue block lives in its own render-from-state child so registry
        # job ticks recompose ONLY it (task-2042).
        yield LibraryIngestQueuePanel(state, id="library-ingest-queue-panel")


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

    @on(Button.Pressed, "#opt-audio_video-choose-transcribe-cpp-gguf")
    def _request_transcribe_cpp_gguf(self, event: Button.Pressed) -> None:
        """Request the shared local-GGUF picker from the owning screen."""
        event.stop()
        self.post_message(self.TranscribeCppGGUFRequested())
