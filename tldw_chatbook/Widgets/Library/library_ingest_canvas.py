"""Library ingest canvas: local-file ingest form + job queue (render-from-state)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import PurePath
from typing import Any

from rich.markup import escape as escape_markup
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Input,
    Select,
    Static,
    TextArea,
)

from tldw_chatbook.Library.ingest_capabilities import (
    TypeGroupCapabilities,
    _is_installed,
    capabilities_for_backend,
    field_disabled_state,
    get_capabilities,
    select_option_label,
)
from tldw_chatbook.Library.library_ingest_jobs import IngestJobState
from tldw_chatbook.Workspaces.conversation_browser_state import (
    format_console_relative_age,
)
from tldw_chatbook.Library.library_ingest_state import (
    WEB_LOCAL_SINGLE_PAGE_NOTE,
    validate_ingest_option_value,
    LibraryIngestCanvasState,
    build_intro_lines,
    build_web_scope_note,
    format_ingest_progress_line,
    library_ingest_retry_label,
)
from tldw_chatbook.Widgets.Library.library_canvas_sync import (
    PostRecomposeCallback,
)


def _command_short_name(command: str) -> str:
    """A compact identifier for an install command's copy button.

    Prefers the pyproject extra (``pip install -e ".[audio]"`` -> ``audio``)
    because that is the part users recognise; anything else truncates.

    (live-verify round) The bracketed spelling this used to return
    (``.[audio]``) never reached the screen: a ``Button`` label is parsed
    as Textual content markup, which ate ``[audio]`` as a style tag and
    left every one of the six or seven stacked buttons rendering the
    identical string "Copy install command (.)" -- the disambiguation was
    invisible exactly when there was most to disambiguate. The extra's
    bare name says the same thing and survives the renderer.
    """
    match = re.search(r"\[([^\]]+)\]", command)
    if match:
        return match.group(1)
    return command if len(command) <= 24 else f"{command[:23]}…"


#: (task-14822) The fold that holds the per-warning detail. The title is a
#: ``CollapsibleTitle``, which is a tab stop -- the detail is keyboard
#: reachable, not mouse-only.
INGEST_TOOLING_FOLD_TITLE = "What's missing"

#: The fold's id. Named once: the summary keeps its expansion across
#: recompose off it, and the canvas's option-panel handler must NOT claim it.
INGEST_TOOLING_FOLD_ID = "ingest-preflight-tooling-detail"

#: (task-14824 AC#3) The path field's persistent label. Names both accepted
#: shapes, because the field takes either -- the placeholder said so and
#: then disappeared.
INGEST_PATH_LABEL_COPY = "File, folder or URL to import"

#: The combined-command copy button's id. Deliberately NOT prefixed
#: ``ingest-preflight-copy-command-`` (the per-extra buttons' prefix, which
#: the shared handler parses an index out of).
INGEST_COPY_ALL_COMMANDS_ID = "ingest-preflight-copy-all-commands"

#: (task 11, spec §9.3 / AC 39) The chunking-template picker's contract.
#: The widget id follows the ``opt-<group>-<name>`` convention so the generic
#: option-value bubble parses it with no extra handler; ``chunk_template`` is
#: the exact key ``_ingest_job_options`` reads as the picker slot of the §9.1
#: resolution order (Task 10).
INGEST_CHUNK_TEMPLATE_FIELD = "chunk_template"
INGEST_CHUNK_TEMPLATE_PICKER_ID = "opt-generic-chunk_template"
#: The default choice's VALUE -- the empty string, so an untouched form
#: submits a falsy picker choice and resolution falls through to the config
#: default / plain options (today's behavior exactly).
INGEST_CHUNK_TEMPLATE_NONE_VALUE = ""
#: The default choice's LABEL (spec §9.3's exact wording).
INGEST_CHUNK_TEMPLATE_NONE_LABEL = "None (manual settings)"
#: What the picker's label line says (it is not a capability-schema field,
#: so it carries no schema hint).
INGEST_CHUNK_TEMPLATE_LABEL = "Chunking template"


def install_command_button_label(command: str) -> str:
    """One label shape for a per-extra copy button, at any count.

    (task-14825 / task-14822 AC#4) The suffix used to be dropped whenever
    there happened to be exactly one command, so the same control had two
    label shapes -- and a user who learned the one-warning wording did not
    recognise the nine-warning one.
    """
    return f"Copy install command ({_command_short_name(command)})"


def combined_install_command(commands: Sequence[str]) -> str:
    """ONE command installing the union of the missing extras.

    (task-14822 AC#4) Nine stacked buttons said "you must install nine
    things"; the truth is one pip invocation. Every warning command this
    codebase produces is ``pip install -e ".[<extra>]"``
    (``OptionalFeatureInfo.source_install_command``), so the union folds
    into a single bracket list in first-seen order. Anything that does not
    match that shape is NOT rewritten -- the commands are chained instead,
    because silently reshaping a command a user is about to paste into a
    shell is worse than a long one.

    Args:
        commands: The distinct install commands, in first-appearance order.

    Returns:
        The combined command, or ``""`` when there are none. A single
        command is returned unchanged.
    """
    commands = [command for command in commands if command.strip()]
    if not commands:
        return ""
    if len(commands) == 1:
        return commands[0]
    extras: list[str] = []
    for command in commands:
        match = re.fullmatch(r'pip install -e "\.\[([^\]]+)\]"', command.strip())
        if match is None:
            return " && ".join(commands)
        for extra in match.group(1).split(","):
            extra = extra.strip()
            if extra and extra not in extras:
                extras.append(extra)
    return f'pip install -e ".[{",".join(extras)}]"'


def preflight_advisory_lines(state: Any) -> tuple[str, ...]:
    """Pre-flight notes that name no missing component.

    (xhigh review round, G2) Not every pre-flight warning is a packaging
    warning: the URL probe emits ``{"label": "Could not check the link",
    "hint": ...}`` with NO ``feature`` key when a site answers oddly. That
    note was counted into "N optional components aren't installed" -- a
    note described as a package -- and then hidden inside the collapsed
    fold, so the one thing the pre-flight actually had to say about the
    link was the one thing not on screen.

    The split is the state's to declare (a rendered line cannot be
    reverse-engineered back into its warning dict); this reads it and
    degrades to "no notes" when the state does not carry the field, which
    is exactly the pre-split behaviour.

    Args:
        state: The canvas render state.

    Returns:
        The advisory note lines, in pre-flight order.
    """
    return tuple(getattr(state, "advisory_lines", ()) or ())


def preflight_tooling_lines(state: Any) -> tuple[str, ...]:
    """The warning lines that DO name a missing component.

    ``advisory_lines`` is the authority on which lines are notes; whether
    the state also leaves them in ``warning_lines`` is not this widget's
    business, so they are filtered out here either way.

    Args:
        state: The canvas render state.

    Returns:
        The tooling-warning lines, in pre-flight order.
    """
    advisory = set(preflight_advisory_lines(state))
    return tuple(
        line
        for line in (getattr(state, "warning_lines", ()) or ())
        if line not in advisory
    )


def ingest_tooling_summary_line(state: Any) -> str:
    """The one line that replaces the tooling-warning wall.

    (task-14822) ``LibraryIngestPreflightSummary`` used to emit one
    ``Static`` per warning -- eleven of them, CSS-double-spaced, plus nine
    copy buttons -- which owned the entire first viewport and read as
    "this app is broken". This states the blast radius in staged FILES,
    which is what the user actually has at stake.

    Everything is read from task-14820's single ``IngestForecast``, never
    recomputed here -- two independently-derived counts on one screen is
    precisely the P1 defect that arc exists to fix. That includes the
    VERB: ``consent_affected`` sums files whose group is missing a
    REQUIRED feature (``will_fail_tooling`` -- doomed) with files merely
    missing an OPTIONAL one (``at_risk`` -- degraded), and this line used
    to call every one of them "optional tooling … may fail". Live, 21 PDFs
    without the pdf extra rendered "⚠ 21 of 21 files need optional tooling
    — those imports may fail." beside a commit line reading "0 will import
    · 21 will fail (need tooling)": the same contradiction, re-created
    inside the fold built to remove it.

    Without a forecast the line falls back to what IS known (how many
    components are missing) rather than inventing a file count.

    Args:
        state: The canvas render state.

    Returns:
        A single ``⚠``-prefixed sentence. The glyph, not the colour,
        carries the severity (monochrome rule).
    """
    warning_count = len(preflight_tooling_lines(state))
    forecast = getattr(state, "forecast", None)
    affected = int(getattr(forecast, "consent_affected", 0) or 0)
    if forecast is None or not affected:
        noun = "component" if warning_count == 1 else "components"
        verb = "isn't" if warning_count == 1 else "aren't"
        tail = (
            "some imports may fail."
            if forecast is None
            else "no staged file needs them."
        )
        return f"⚠ {warning_count} optional {noun} {verb} installed — {tail}"
    # ``staged_total`` is optional: the proportion is what turns "eleven
    # warnings" back into "3 of your 21 files", so it is used whenever the
    # forecast carries it.
    staged_total = int(getattr(forecast, "staged_total", 0) or 0)
    doomed = int(getattr(forecast, "will_fail_tooling", 0) or 0)
    degraded = int(getattr(forecast, "at_risk", 0) or 0)
    scope = f"{affected} of {staged_total} files" if staged_total else (
        f"{affected} file" if affected == 1 else f"{affected} files"
    )
    singular = affected == 1 and not staged_total
    verb = "needs" if singular else "need"
    if doomed and degraded:
        # Both fates in one selection: stating one verb for both is the
        # defect, so both are stated, in the commit line's own vocabulary.
        return (
            f"⚠ {scope} {verb} more tooling — "
            f"{doomed} will fail, {degraded} may fail."
        )
    if doomed:
        outcome = "that import will fail." if singular else "those imports will fail."
        return f"⚠ {scope} {verb} tooling that isn't installed — {outcome}"
    # Degraded only -- and the defensive case of a forecast that reports an
    # affected count without the split, where the softer claim is the only
    # one still supported by what it does say.
    outcome = "that import may fail." if singular else "those imports may fail."
    return f"⚠ {scope} {verb} optional tooling — {outcome}"


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
        # (xhigh review round, G3) The fold's expansion is state, not a
        # widget accident: the screen calls ``refresh(recompose=True)`` on
        # THIS widget on every registry tick, so a hard-coded
        # ``collapsed=True`` snapped the fold shut under a user mid-read
        # during an active import. Kept on the instance because the
        # instance survives that recompose; seeded from the render state so
        # that once the screen persists the flag (the option panels'
        # ``expanded_type_groups`` convention), it survives the FULL
        # recompose a structural change forces too.
        self.tooling_detail_expanded = bool(
            getattr(state, "tooling_detail_expanded", False)
        )
        self.styles.width = "1fr"
        self.styles.height = "auto"

    @on(Collapsible.Expanded)
    @on(Collapsible.Collapsed)
    def _remember_tooling_fold(
        self,
        event: Collapsible.Expanded | Collapsible.Collapsed,
    ) -> None:
        """Keep the fold's expansion across recompose, and report it.

        The message is the durable half: this widget is render-only, so the
        screen owns persistence exactly as it does for
        ``OptionPanelToggled``. The event is deliberately NOT stopped --
        the canvas's option-panel handler ignores this id.
        """
        if (event.collapsible.id or "") != INGEST_TOOLING_FOLD_ID:
            return
        expanded = isinstance(event, Collapsible.Expanded)
        self.tooling_detail_expanded = expanded
        self.post_message(
            LibraryIngestCanvas.ToolingDetailToggled(expanded=expanded)
        )

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
                # (task-3312 #2) Verbatim, not escape-then-parse: the
                # escape_markup/content-markup pairing leaks a literal
                # backslash for mixed bracket runs (see the queue-row
                # comment in ``LibraryIngestQueuePanel.compose``).
                yield Static(
                    error,
                    id=f"ingest-preflight-error-{index}",
                    classes="library-ingest-quiet-line",
                    markup=False,
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
        # (xhigh review round, G2) A pre-flight note that names no missing
        # component -- the URL probe's "the site answered 403 to our check"
        # -- is not tooling and must not be folded away behind "What's
        # missing": it is the only thing on screen saying the link could
        # not be confirmed.
        for index, note in enumerate(preflight_advisory_lines(state)):
            yield Static(
                note,
                id=f"ingest-preflight-note-{index}",
                classes="library-ingest-quiet-line",
                markup=False,
            )
        tooling_lines = preflight_tooling_lines(state)
        if tooling_lines:
            # (task-14822) ONE line at canvas level; the eleven warnings and
            # their per-extra copy buttons live inside the fold below. The
            # wall used to own the entire first viewport, drowning the two
            # lines that DO matter (unsupported/empty) at identical weight.
            yield Static(
                ingest_tooling_summary_line(state),
                id="ingest-preflight-tooling-summary",
                classes="library-ingest-tooling-summary",
                markup=False,
            )
            # The combined command sits OUTSIDE the fold: recovering from a
            # missing dependency must stay one press away, and one pip
            # invocation is the truth the nine stacked buttons obscured.
            commands = tuple(state.warning_commands)
            if combined_install_command(commands):
                yield Button(
                    "Copy install command",
                    id=INGEST_COPY_ALL_COMMANDS_ID,
                    classes=(
                        "library-canvas-action ingest-preflight-copy-command"
                    ),
                    compact=True,
                )
            with Collapsible(
                title=INGEST_TOOLING_FOLD_TITLE,
                collapsed=not self.tooling_detail_expanded,
                id=INGEST_TOOLING_FOLD_ID,
            ):
                for index, warning in enumerate(tooling_lines):
                    yield Static(
                        f"⚠ {warning}",
                        id=f"ingest-preflight-warning-{index}",
                        classes="library-ingest-quiet-line",
                        markup=False,
                    )
                # (task-3304, MI-17) The pip command must be recoverable AT
                # the warning -- the guardrail modal used to hold the only
                # copy button, and a command read off a wrapped prose line
                # is a transcription exercise. One compact button per
                # DISTINCT command (several features often share one
                # extra), each disambiguated by its extra name in a label
                # shape that no longer changes with the count.
                #
                # (xhigh review round, G5) ...but ONLY when there is more
                # than one: at exactly one command the combined button
                # above already carries that identical string, so the pair
                # rendered the same command twice under two different
                # labels -- the one-label-shape defect task-14822 fixed,
                # re-introduced one level down.
                if len(commands) > 1:
                    for index, command in enumerate(commands):
                        yield Button(
                            install_command_button_label(command),
                            id=f"ingest-preflight-copy-command-{index}",
                            classes=(
                                "library-canvas-action "
                                "ingest-preflight-copy-command"
                            ),
                            compact=True,
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
        # (task-14822 AC#3) These two are OUTCOMES of this import ("5
        # unsupported files will be skipped", "1 empty file will fail"),
        # not facts about the environment -- they used to share
        # ``library-ingest-quiet-line`` with the eleven tooling warnings
        # and drowned in them at identical weight.
        if state.unsupported_line:
            yield Static(
                state.unsupported_line,
                id="ingest-unsupported-summary",
                classes="library-ingest-outcome-line",
                markup=False,
            )
        if state.empty_line:
            yield Static(
                state.empty_line,
                id="ingest-empty-summary",
                classes="library-ingest-outcome-line",
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
        # (task-2221 owner ruling) The tally leads with the LATEST batch;
        # the lifetime line stays secondary below it.
        if state.latest_batch_line:
            yield Static(
                state.latest_batch_line,
                id="library-ingest-latest-batch",
                markup=False,
            )
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
        # (task-2221) Per-submission group headers: rendered before the
        # first row of each headed group. Rows keep their flat order and
        # identity semantics -- the header is an extra Static, not a
        # container, so the in-place update paths are untouched.
        headers_before: dict[str, str] = {}
        for group in state.queue_groups:
            if group.header_line and group.job_ids:
                headers_before[group.job_ids[0]] = group.header_line
        for index, row in enumerate(state.queue_rows):
            header_line = headers_before.get(row.job_id, "")
            if header_line:
                yield Static(
                    header_line,
                    classes="library-ingest-batch-header",
                    markup=False,
                )
            # A source filename or error can contain markup syntax (a
            # literal "[/bracket]" in the name, an error quoting config
            # keys) -- ``markup=False`` below renders it verbatim, which
            # both keeps a hostile filename from raising MarkupError at
            # mount time (the L3a lesson) AND never leaks an escape
            # backslash. The old ``escape_markup``-then-parse pairing did:
            # rich's escape skips a bracket run that never closes as a tag
            # while escaping the inner closed ones, and Textual's content
            # markup then leaves the first escape's backslash literal --
            # the live "\[web_security]" receipt (task-3312 #2).
            row_classes = "library-ingest-row"
            has_progress_line = row.state in (
                IngestJobState.PARSING,
                IngestJobState.WRITING,
            ) or bool(row.progress)
            if has_progress_line:
                row_classes += " library-ingest-row-with-progress"
            # (task-2230 a11y) Severity gets a colour IN ADDITION to the
            # glyph+word it already carries -- failed and done rows were
            # byte-identical in colour, so scanning a tall queue for the
            # one failure was a linear read.
            if row.state == IngestJobState.FAILED:
                row_classes += " library-ingest-row-failed"
            elif row.state == IngestJobState.SKIPPED:
                row_classes += " library-ingest-row-skipped"
            stt_actions = _stt_recovery_actions(row.error_detail)
            has_actions = (
                row.can_open
                or row.can_open_on_server
                or row.can_retry
                or row.can_dismiss
                or row.can_cancel
                or row.can_force_stop
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
                row.line,
                id=f"library-ingest-row-{index}",
                classes=row_classes,
                markup=False,
            )
            if has_progress_line and row.state is not None:
                progress = row.progress
                if progress is None and row.state is IngestJobState.WRITING:
                    progress = {"phase": "writing"}
                yield Static(
                    format_ingest_progress_line(progress, state=row.state),
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
                    if row.can_force_stop:
                        yield Button(
                            "Force stop",
                            id=f"library-ingest-force-stop-{row.job_id}",
                            classes=(
                                "library-canvas-action "
                                "library-ingest-force-stop "
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
                title="Recent imports",
                collapsed=True,
                id="library-ingest-recent",
            ):
                for job in state.recent_jobs:
                    dismissed_suffix = (
                        " (dismissed)"
                        if getattr(job, "dismissed", False)
                        else ""
                    )
                    # (task-2223) Basename + relative time first -- a list
                    # of ~130-char absolute paths was unscannable. The full
                    # path keeps a muted second line.
                    name = PurePath(str(job.source_path)).name
                    age = (
                        format_console_relative_age(
                            job.finished_at_wall,
                            now=datetime.now(timezone.utc),
                        )
                        if getattr(job, "finished_at_wall", "")
                        else ""
                    )
                    age_suffix = f" · {age}" if age else ""
                    # (task-3305, MI-14) ``markup=False`` already renders
                    # these literally -- escaping on top of it painted the
                    # escape backslashes into bracketed filenames.
                    yield Static(
                        f"{name} — "
                        f"{job.state.value}{dismissed_suffix}{age_suffix}",
                        classes="library-ingest-recent-item",
                        markup=False,
                    )
                    yield Static(
                        str(job.source_path),
                        classes="library-ingest-recent-path",
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
    told a first-time user nothing about what any of it does. (task-3305)
    Select values resolve through their display labels -- the raw token
    (``pymupdf4llm``) must not leak into the title either.
    """
    if field.type == "checkbox":
        return f"{field.label}: {'on' if value else 'off'}"
    if field.type == "select":
        return f"{field.label}: {select_option_label(field, value)}"
    if field.type == "textarea":
        return f"{field.label}: {'set' if str(value).strip() else 'empty'}"
    return f"{field.label}: {value}"


def _option_is_default(field: Any, value: Any) -> bool:
    """Whether ``value`` is (semantically) the field's schema default.

    Form echoes hold display text, so a number field's ``"1000"`` must
    compare equal to its schema default ``1000``.
    """
    if field.type == "checkbox":
        return bool(value) == bool(field.default)
    return str(value) == str(field.default)


#: (task-3305, MI-16) Collapsed titles cap at this many name:value pairs --
#: the audio panel's full dump ran ~140 characters with a dangling empty
#: value; a receipt nobody can scan is not a receipt.
_TITLE_MAX_PAIRS = 3


def _is_packaging_gate(field: Any, *, is_installed: Any) -> bool:
    """Whether ``field`` is inert because a PACKAGE is missing.

    (xhigh review round, G4) The distinction task-14824 was after and the
    title lost: a ``depends_on`` feature that is not installed is work the
    user must do OUTSIDE the app, and it is the reason a keyboard user
    cannot reach at all (Textual drops a ``disabled`` widget from the tab
    order). A closed ``enabled_when`` sibling gate is neither -- it is the
    form working as designed, recoverable right here on a control that IS
    focusable. Counting both made a fully working default Web panel lead
    its receipt with "2 options unavailable — single-page fetch selected".

    This is the same first branch :func:`field_disabled_state` evaluates,
    and it is checked in the same order: a field whose package is missing
    is packaging-gated whatever its sibling gate says.

    Args:
        field: The option field under evaluation.
        is_installed: Feature-availability probe.

    Returns:
        ``True`` when the field's packaging gate is the closed one.
    """
    depends_on = getattr(field, "depends_on", None)
    return depends_on is not None and not is_installed(depends_on)


def build_type_group_title(
    cap: TypeGroupCapabilities,
    values: dict[str, Any],
    *,
    is_installed: Any = None,
) -> str:
    """Collapsed-panel title: group label + the few most salient facts.

    (task-3305, MI-16) Shared by ``_compose_type_group`` and the screen's
    in-place receipt update so the two renders can never drift. Rules:
    empty values are skipped outright (never ``"…folder: ,"``);
    changed-from-default pairs outrank untouched defaults (a receipt is
    about what the user chose); at most :data:`_TITLE_MAX_PAIRS` pairs
    render, with a trailing ``…`` naming the omission.

    Three later findings ride the same one-line receipt, ahead of the
    pairs because each of them changes what the panel MEANS:

    - (task-14826 AC#2) An invalid value inside a COLLAPSED panel was
      invisible: ``-ingest-option-invalid`` is applied to the ``Input``,
      which lives in the collapsed body, so the gate said "fix the
      highlighted options" while nothing on screen was highlighted. The
      title leads with a ``⚠`` and names the field. Text, not a CSS class,
      deliberately: the screen's in-place update assigns ``Collapsible
      .title`` and nothing else, so a class-based marker would drift --
      and a mark that survives monochrome is the house rule anyway.
    - (task-14824 AC#2) A disabled control is removed from the tab order
      by Textual outright, so the ``— needs X installed`` reasons written
      for keyboard users were unreachable by keyboard. The title is a
      ``CollapsibleTitle``, which IS a tab stop, so a blocked group states
      its reason there.
    - (task-14825 #7) Disabled fields no longer contribute pairs at all:
      the title advertised ``Extract text (OCR): on`` while the control
      below it read ``— needs OCR backend installed``.

    Args:
        cap: The group's capability schema.
        values: Current per-group option values (missing keys fall back to
            schema defaults).
        is_installed: Feature-availability probe forwarded to
            :func:`field_disabled_state`. Defaults to this module's own
            ``_is_installed`` so the late lookup keeps working for tests
            that patch it.

    Returns:
        The full title string, e.g.
        ``"Audio & video — 13 options unavailable — needs faster-whisper
        installed"``.
    """
    probe = _is_installed if is_installed is None else is_installed
    changed: list[str] = []
    blocked_reasons: list[str] = []
    blocked_count = 0
    invalid_labels: list[str] = []
    for field in cap.fields:
        value = values.get(field.name, field.default)
        disabled, reason = field_disabled_state(
            field, cap, values, is_installed=probe
        )
        if disabled:
            # (task-14825 #7) No disabled field contributes a value pair --
            # advertising a setting the user cannot change is a promise the
            # panel does not keep. But only a PACKAGING gate counts as
            # "unavailable" (G4): a closed sibling gate is the form working.
            if _is_packaging_gate(field, is_installed=probe):
                blocked_count += 1
                if reason and reason not in blocked_reasons:
                    blocked_reasons.append(reason)
            continue
        if validate_ingest_option_value(field, value):
            invalid_labels.append(field.label)
            continue
        if value is None or str(value).strip() == "":
            continue
        if not _option_is_default(field, value):
            changed.append(_summarise_option(field, value))
    # Order: the blocker first (nothing in this panel can be committed
    # while it stands), then what the USER chose (the receipt's whole
    # point), then the blocked-options clause -- which is never dropped by
    # the cap, because being droppable is what made it unreachable -- then
    # Omit untouched defaults: collapsed titles are change receipts, not a
    # second copy of the schema. This keeps the default state scannable.
    shown: list[str] = []
    if invalid_labels:
        extra = (
            f" (+{len(invalid_labels) - 1} more)"
            if len(invalid_labels) > 1
            else ""
        )
        shown.append(f"⚠ {invalid_labels[0]} needs fixing{extra}")
    blocked_clause = ""
    if blocked_count:
        noun = "option" if blocked_count == 1 else "options"
        blocked_clause = f"{blocked_count} {noun} unavailable"
        if blocked_reasons:
            # Every reason collected above is a packaging one by
            # construction, so the first is the right one to carry -- the
            # preference used to be applied here over a mixed list, which
            # is what let within-form gates onto a healthy panel's title.
            blocked_clause += f" — {blocked_reasons[0]}"
    reserved = len(shown) + (1 if blocked_clause else 0)
    shown.extend(changed[: max(_TITLE_MAX_PAIRS - reserved, 0)])
    if blocked_clause:
        shown.append(blocked_clause)
    if len(changed) > len([pair for pair in shown if pair in changed]):
        shown.append("…")
    if not shown:
        return cap.label
    return f"{cap.label} — {', '.join(shown)}"


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
        The scope sentence for the panel. (task-3305, MI-16) Composed from
        the group's noun phrases, not its category label -- "Applies to all
        Plain text & HTML in this import." was not a sentence.
    """
    singular = cap.noun_singular or cap.label
    plural = cap.noun_plural or cap.label
    return (
        f"Applies to every {singular} in this import."
        if has_files
        else f"Applies to {plural} if this import contains any."
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


#: (task-3304, MI-08) The task-1623 fold-indicator convention: a reserved
#: bottom row saying more content exists, shown only while the canvas
#: actually overflows -- a mid-sentence clip must never be the only signal.
INGEST_FOLD_HINT_COPY = "▼ more — scroll for the rest"

class LibraryIngestCanvas(PostRecomposeCallback, VerticalScroll):
    """Render the Library ingest canvas: the local-file ingest form and its job queue.

    ``VerticalScroll`` root (the L3a clipping lesson -- a plain ``Vertical``
    canvas clips content past the fold); every child is stacked full-width,
    mirroring ``LibraryNotesCanvas``'s sync panel. Per-type option panels
    are rendered from ``ingest_capabilities.py`` schemas and post messages
    for all state changes so the screen can persist them.
    """

    # (task-3314) The two-press Start confirm rides the gate line; while
    # armed it carries the warning treatment. Theme tokens only (the same
    # rule the retired guardrail modal's CSS was pinned to); the "⚠" glyph
    # in the copy keeps the state legible in monochrome.
    DEFAULT_CSS = """
    LibraryIngestCanvas .-ingest-start-confirm {
        color: $warning;
        text-style: bold;
    }
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

    class ToolingDetailToggled(Message):
        """The pre-flight tooling fold was expanded or collapsed.

        (xhigh review round, G3) Posted by
        ``LibraryIngestPreflightSummary`` so the screen can persist the
        expansion the way it persists ``expanded_type_groups`` -- without
        it, the full recompose a structural change forces rebuilds the
        summary widget and the fold reverts to closed.
        """

        def __init__(self, *, expanded: bool) -> None:
            super().__init__()
            self.expanded = expanded

    class DirectoryBrowseRequested(Message):
        """A directory-backed text option requested a native picker."""

        def __init__(self, group: str, name: str) -> None:
            super().__init__()
            self.group = group
            self.name = name

    class ParakeetInstallRequested(Message):
        """The user requested the curated Parakeet v2 installer."""

    class ExternalPreparationCancelRequested(Message):
        """The user requested cancellation of external-model preparation."""

    class TranscribeCppGGUFRequested(Message):
        """The user requested a local transcribe.cpp GGUF picker."""

    def __init__(
        self,
        state: LibraryIngestCanvasState,
        *,
        external_busy: bool = False,
        external_status: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.external_busy = external_busy
        self.external_status = external_status
        self.styles.width = "1fr"
        self.styles.min_width = 40
        # Value each option widget was last rendered/reported with, keyed by
        # ``(group, field name)``. Seeded by ``_compose_type_group`` so that a
        # widget announcing the value we just gave it is recognised as mount
        # noise rather than a user edit -- see ``_handle_option_value_changed``.
        self._reported_option_values: dict[tuple[str, str], Any] = {}
        # (task 11, spec §9.3) Live chunking-template names, fetched off the
        # mount path via the scope service (see ``_fetch_chunk_templates``).
        # Lives on the CANVAS, not a child, because ``sync_state`` recomposes
        # the children -- a rebuilt ``Select`` re-reads this cache so the
        # populated list survives every re-render without a re-query.
        self._chunk_template_names: list[str] = []

    def sync_state(self, state: LibraryIngestCanvasState) -> None:
        """Rebuild only the mounted ingest canvas from a complete snapshot.

        Args:
            state: Complete ingest form and submission state to render.
        """
        self.state = state
        self._reported_option_values.clear()
        self.refresh(recompose=True)

    def _compose_type_group(
        self,
        group: str,
        cap: TypeGroupCapabilities,
        values: dict[str, Any],
        expanded: bool,
        has_files: bool = True,
    ) -> Collapsible:
        """Build a collapsible options panel for one detected type group."""
        # A control that does not apply to the effective backend must not
        # render: this schema filter is the mode-visibility source of truth,
        # ahead of both the field body and the collapsed summary title.
        visible_cap = capabilities_for_backend(cap, self.state.ingest_backend)
        # (task-2016) The generic panel is always rendered so global options
        # stay reachable -- but claiming "Applies to all X in this import."
        # with zero such files staged was a false statement.
        scope_label = ingest_scope_label(visible_cap, has_files)
        children: list[Any] = [Static(scope_label, classes="type-group-scope")]
        cap_fields_by_name = {f.name: f for f in visible_cap.fields}

        for field in visible_cap.fields:
            value = values.get(field.name, field.default)
            # Two independent reasons a field can be uneditable: its tooling
            # is not installed, or the sibling field that gates it is off.
            # (task-3304, MI-07) One shared computation returns BOTH the
            # disabled flag and the reason annotation, so the inert state
            # and its explanation can never disagree. ``_is_installed`` is
            # passed as this module's own global on purpose: tests patch
            # ``library_ingest_canvas._is_installed`` and the late lookup
            # keeps that seam working.
            disabled, disabled_note = field_disabled_state(
                field, visible_cap, values, is_installed=_is_installed
            )
            control_disabled = disabled or self.external_busy
            widget_id = f"opt-{group}-{field.name}"

            if field.type == "checkbox":
                self._reported_option_values[(group, field.name)] = bool(value)
                # (task-3303) A gated checkbox must carry its reason at the
                # control: the label absorbs the schema hint ("Enable OCR
                # (docling or docext engines only)"), so the inert state is
                # explained where the user is looking, not somewhere else.
                checkbox_label = (
                    f"{field.label} ({field.hint})"
                    if getattr(field, "hint", "")
                    else field.label
                )
                # (task-3304, MI-07) Disabled-state annotation: the WHY at
                # the control while the gate is closed. Empty for fields
                # whose static hint above already names the gate, so
                # labels never double-annotate.
                if disabled and disabled_note:
                    checkbox_label = f"{checkbox_label} — {disabled_note}"
                children.append(
                    StateGlyphCheckbox(
                        checkbox_label,
                        value=bool(value),
                        id=widget_id,
                        disabled=control_disabled,
                    )
                )
            elif field.type == "select":
                # (task-3305, MI-09) Human display labels; the VALUE side
                # (and everything persisted/submitted) stays the internal
                # token.
                select_options = [
                    (select_option_label(field, opt), opt)
                    for opt in field.options
                ]
                select_value = value if value in field.options else field.default
                if select_value not in field.options and field.options:
                    select_value = field.options[0]
                self._reported_option_values[(group, field.name)] = select_value
                # (task-2043) Selects missed task-2012's labeling pass: a
                # bare "pymupdf4llm" carries no meaning on its own.
                # (task-3304, MI-07) While schema-disabled, the label
                # carries the reason -- selects re-compose on every gate
                # flip (checkbox/select changes recompose the canvas), so
                # compose-time is the single point of truth.
                select_label = field.label
                if disabled and disabled_note:
                    select_label = f"{select_label} — {disabled_note}"
                children.append(
                    Static(
                        select_label,
                        classes="type-group-field-label",
                        markup=False,
                    )
                )
                children.append(
                    Select(
                        select_options,
                        value=select_value,
                        id=widget_id,
                        disabled=control_disabled,
                        allow_blank=False,
                    )
                )
                if group == "web" and field.name == "scrape_method":
                    # (task-3303 AC5) Local single-page honesty, right under
                    # the control that promises otherwise: the local article
                    # path fetches ONE page, so a multi-page method selected
                    # while targeting this machine must say so. Always
                    # mounted, display-managed (select changes recompose,
                    # but the stable structure keeps in-place updates safe).
                    scope_note = Static(
                        WEB_LOCAL_SINGLE_PAGE_NOTE,
                        id="web-local-scope-note",
                        classes="type-group-scope",
                        markup=False,
                    )
                    scope_note.display = bool(
                        build_web_scope_note(self.state.ingest_backend, values)
                    )
                    children.append(scope_note)
            else:
                self._reported_option_values[(group, field.name)] = str(value)
                # A populated Input never shows its placeholder, so
                # placeholder-as-label left values like "1000" with no
                # visible meaning (task-2012). The label gets its own line,
                # carrying the unit/range hint up front (task-2223).
                label_text = (
                    f"{field.label} ({field.hint})"
                    if getattr(field, "hint", "")
                    else field.label
                )
                # (task-3304, MI-07) Disabled-state reason at the control.
                # (live-verify round) APPENDED to ``label_text``, not
                # rebuilt from ``field.label``: rebuilding dropped the hint
                # -- so on a stock install the cookies field's "video URLs
                # only" and the trim fields' "HH:MM:SS or seconds" were
                # invisible exactly while the control was inert and the
                # user had the most to work out. The checkbox branch above
                # already appends; these two now agree.
                if disabled and disabled_note:
                    label_text = f"{label_text} — {disabled_note}"
                children.append(
                    Static(
                        label_text,
                        classes="type-group-field-label",
                        markup=False,
                    )
                )
                if field.type == "textarea":
                    input_widget = TextArea(
                        str(value),
                        # Prompts must keep newlines; a single-line Input
                        # would silently flatten the instructions the user
                        # supplied.
                        placeholder=field.placeholder or field.label,
                        id=widget_id,
                        disabled=control_disabled,
                    )
                    # TextArea defaults to ``height: 1fr``. Bound it inside
                    # an option panel so two prompts remain compact-viewport
                    # reachable instead of consuming all available height.
                    input_widget.styles.height = 4
                    input_widget.styles.min_width = 0
                else:
                    input_widget = Input(
                        value=str(value),
                        # (task-3305) Example content when the schema
                        # provides it; a placeholder repeating the label
                        # line directly above is stutter.
                        placeholder=field.placeholder or field.label,
                        id=widget_id,
                        disabled=control_disabled,
                    )
                if field.directory_picker:
                    input_widget.styles.width = "1fr"
                    input_widget.styles.min_width = 0
                    browse_button = Button(
                        "Browse…",
                        id=f"{widget_id}-browse",
                        classes=(
                            "library-canvas-action library-ingest-directory-browse"
                        ),
                        compact=True,
                        disabled=control_disabled,
                    )
                    browse_button.styles.width = "auto"
                    path_row = Horizontal(
                        input_widget,
                        browse_button,
                        classes="library-ingest-path-actions",
                    )
                    path_row.styles.width = "100%"
                    path_row.styles.height = 3
                    children.append(path_row)
                    if field.name == "transcription_model_dir":
                        children.append(
                            Static(
                                "This import and its retries only · does not "
                                "change Lab Models or your global source.",
                                id="library-external-scope-helper",
                                classes="type-group-scope",
                                markup=False,
                            )
                        )
                else:
                    children.append(input_widget)
                # (task-2130) Inline validation message -- a text line, not a
                # color-only border. Display-managed so typing updates it in
                # place without recomposing the panel.
                # A disabled field no longer gates Start (Qodo round) --
                # its error line hides with it, so message and gate agree.
                error_message = (
                    "" if disabled else validate_ingest_option_value(field, value)
                )
                if error_message:
                    # (task-2230 a11y) Persistent marker: the stock invalid
                    # border only paints while focused.
                    input_widget.add_class("-ingest-option-invalid")
                error_line = Static(
                    error_message,
                    id=f"{widget_id}-error",
                    classes="type-group-field-error",
                    markup=False,
                )
                error_line.display = bool(error_message)
                children.append(error_line)

        if (
            group == "generic"
            and str(self.state.ingest_backend).strip().lower() != "server"
        ):
            # (task 11, spec §9.3 / AC 39) The chunking-template picker.
            # HIDDEN in server mode via the same compose-time filter that is
            # this file's mode-visibility source of truth for schema fields
            # (``capabilities_for_backend`` above): a server-mode snapshot
            # never carries a template, and Task 10's ``build_server_ingest_
            # kwargs`` strip is the defensive half for stale snapshots.
            # Options come from the canvas-level cache, so a recompose
            # re-renders the populated list without re-querying the DB.
            # escape_markup: template names are user-authored free text and
            # ``Select`` parses its labels as markup (the bench_editor
            # precedent) -- an unescaped ``[red]`` in a name would be eaten
            # as a style tag.
            available = [
                INGEST_CHUNK_TEMPLATE_NONE_VALUE,
                *self._chunk_template_names,
            ]
            picker_value = values.get(
                INGEST_CHUNK_TEMPLATE_FIELD, INGEST_CHUNK_TEMPLATE_NONE_VALUE
            )
            if picker_value not in available:
                picker_value = INGEST_CHUNK_TEMPLATE_NONE_VALUE
            self._reported_option_values[
                ("generic", INGEST_CHUNK_TEMPLATE_FIELD)
            ] = picker_value
            chunk_on = bool(values.get("chunk", True))
            picker_label = INGEST_CHUNK_TEMPLATE_LABEL
            if not chunk_on:
                picker_label = f"{picker_label} — needs Chunk content on"
            children.append(
                Static(
                    picker_label,
                    classes="type-group-field-label",
                    markup=False,
                )
            )
            children.append(
                Select(
                    [
                        (INGEST_CHUNK_TEMPLATE_NONE_LABEL, INGEST_CHUNK_TEMPLATE_NONE_VALUE),
                        *[
                            (escape_markup(name), name)
                            for name in self._chunk_template_names
                        ],
                    ],
                    value=picker_value,
                    id=INGEST_CHUNK_TEMPLATE_PICKER_ID,
                    disabled=(not chunk_on) or self.external_busy,
                    allow_blank=False,
                )
            )

        if group == "audio_video":
            provider = cap_fields_by_name["transcription_provider"]
            provider_value = values.get(
                "transcription_provider", provider.default
            )
            install_gated = provider_value != "parakeet-onnx"
            install_label = "Install verified Parakeet v2 INT8 (630.6 MiB)…"
            if install_gated:
                # (task-3304, MI-07) Inert-actions rule: a disabled button
                # carries the WHY in its label, never dimming alone.
                install_label += " — needs the parakeet-onnx provider"
            children.append(
                Button(
                    install_label,
                    id="opt-audio_video-install-parakeet-v2",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=install_gated or self.external_busy,
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
                        disabled=self.external_busy,
                    )
                )

        children.append(
            Button(
                "Reset to defaults",
                id=f"opt-{group}-reset",
                classes="library-canvas-action library-ingest-option-reset",
                compact=True,
                disabled=self.external_busy,
            )
        )

        panel = Vertical(*children, classes="type-group-contents")
        # (task-3305, MI-16) Shared with the screen's in-place receipt
        # update: capped, empty-skipping, changed-values-first.
        title = build_type_group_title(visible_cap, values)
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
                disabled=self.external_busy,
            )
        if state.unavailable_line:
            yield Static(
                state.unavailable_line,
                id="library-ingest-unavailable-line",
                classes="library-ingest-quiet-line",
                markup=False,
            )
        # (task-14824 AC#3) The primary control's identity was
        # placeholder-only -- and a placeholder is gone the moment the field
        # holds a path, which is exactly when a long absolute path needs
        # saying what it IS. Same fix task-2012 applied to the option
        # fields, finally applied to the field above them.
        yield Static(
            INGEST_PATH_LABEL_COPY,
            id="library-ingest-path-label",
            classes="library-ingest-field-label",
            markup=False,
        )
        yield Input(
            value=state.form.path,
            placeholder="Path to a local file or a URL…",
            id="library-ingest-path",
            classes="library-ingest-field",
            disabled=self.external_busy,
        )
        with Horizontal(classes="library-ingest-path-actions"):
            yield Button(
                "Browse…",
                id="library-ingest-browse",
                classes="library-canvas-action",
                compact=True,
                disabled=self.external_busy,
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
                disabled=self.external_busy,
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
        # TASK-15702: placeholders disappear as soon as the user types and
        # therefore cannot carry field identity. Keep short, persistent
        # labels above the three optional metadata fields; placeholders are
        # now examples/default guidance only.
        with Horizontal(id="library-ingest-metadata-row"):
            with Vertical(classes="library-ingest-metadata-field"):
                yield Static(
                    "Title (optional)",
                    id="library-ingest-title-label",
                    classes="library-ingest-field-label",
                    markup=False,
                )
                yield Input(
                    value=state.form.title,
                    placeholder="Defaults to source name",
                    id="library-ingest-title",
                    classes="library-ingest-field",
                )
            with Vertical(classes="library-ingest-metadata-field"):
                yield Static(
                    "Author (optional)",
                    id="library-ingest-author-label",
                    classes="library-ingest-field-label",
                    markup=False,
                )
                yield Input(
                    value=state.form.author,
                    placeholder="e.g. Ada Lovelace",
                    id="library-ingest-author",
                    classes="library-ingest-field",
                )
            with Vertical(classes="library-ingest-metadata-field"):
                yield Static(
                    "Keywords (optional)",
                    id="library-ingest-keywords-label",
                    classes="library-ingest-field-label",
                    markup=False,
                )
                yield Input(
                    value=state.form.keywords,
                    placeholder="comma-separated",
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
        # (task-2140) Always mounted, display-managed: the conditional
        # compose reintroduced the round-3 empty-Recent bug class -- a
        # text-only pre-flight applies via the NON-structural in-place
        # path, which never mounts a conditionally-composed canvas-level
        # element (PDF selections rendered the line, plain text never),
        # and after Clear the stale line survived. The in-place updater
        # owns its content and visibility.
        # TASK-15702: the decision and its consequences are docked together
        # so a long preflight never separates the forecast from Start. Once
        # a submission clears the form, hide this blank gate and give the
        # activity receipt the viewport instead.
        with Vertical(id="library-ingest-commit-bar") as commit_bar:
            commit_bar.display = bool(state.form.path.strip() or not state.queue_rows)
            commit_summary = Static(
                state.commit_summary_line,
                id="library-ingest-commit-summary",
                classes="library-ingest-quiet-line",
                markup=False,
            )
            commit_summary.display = bool(state.commit_summary_line)
            yield commit_summary
            start_quiet_line = Static(
                state.start_quiet_line,
                id="library-ingest-start-quiet-line",
                classes="library-ingest-quiet-line",
                markup=False,
            )
            start_quiet_line.styles.height = 1
            start_quiet_line.set_class(
                state.start_confirm_armed, "-ingest-start-confirm"
            )
            yield start_quiet_line
            analysis_hint = Static(
                state.analysis_hint_line,
                id="library-ingest-analysis-hint",
                classes="library-ingest-quiet-line",
                markup=False,
            )
            analysis_hint.display = bool(state.analysis_hint_line)
            yield analysis_hint
            external_status = Static(
                self.external_status,
                id="library-external-prepare-status",
                classes="library-ingest-quiet-line",
                markup=False,
            )
            external_status.display = bool(self.external_status)
            yield external_status
            external_cancel = Button(
                "Cancel external preparation",
                id="library-external-prepare-cancel",
                classes="library-canvas-action",
                compact=True,
            )
            external_cancel.display = self.external_busy
            yield external_cancel
            yield Button(
                "Start import",
                id="library-ingest-start",
                classes="library-canvas-action",
                compact=True,
                disabled=not state.start_enabled or self.external_busy,
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
        # (task-3313) "Retry this batch": re-stages the last submission's
        # source + options + metadata into the form. Lives with the queue's
        # outcome area but deliberately OUTSIDE the recomposing queue panel
        # -- always mounted, display-managed by the screen's dynamic-region
        # updater (never conditionally composed, the four-incident lesson),
        # so it keeps object identity across job ticks.
        retry_last = Button(
            library_ingest_retry_label(state.retry_confirm_armed),
            id="library-ingest-retry-last",
            classes="library-canvas-action",
            compact=True,
        )
        retry_last.display = state.show_retry_last
        yield retry_last
        # (task-3304, MI-08) Fold indicator, task-1623 convention: docked
        # chrome (a docked child of a scroll container never scrolls with
        # the content), always mounted, display-managed by
        # ``sync_fold_hint`` -- never conditionally composed, per the
        # canvas's in-place-update discipline.
        fold_hint = Static(
            INGEST_FOLD_HINT_COPY,
            id="library-ingest-fold-hint",
            markup=False,
        )
        fold_hint.display = False
        yield fold_hint


    def on_mount(self) -> None:
        """Settle the fold indicator once first layout has real sizes."""
        self.call_after_refresh(self.sync_fold_hint)

    def on_show(self) -> None:
        """Populate DB-backed controls once the canvas is actually visible.

        (task 11, spec §9.3 / AC 39) The chunking-template picker is
        populated OFF the mount path: mount-time DB populate is the
        documented cause of "(0)" count bugs in the Notes rebuild, so the
        fetch is scheduled from the visibility event (never ``on_mount``)
        into a worker. Re-entering the Ingest canvas remounts it, so this
        also re-queries after the user creates/renames a template
        elsewhere; within one mount the populated list survives recomposes
        off the canvas-level cache.
        """
        self._request_chunk_template_refresh()

    def _request_chunk_template_refresh(self) -> None:
        """Schedule (once per visibility) the template-list fetch worker."""
        if str(self.state.ingest_backend).strip().lower() == "server":
            return
        try:
            self.run_worker(
                self._fetch_chunk_templates(),
                group="library-ingest-chunk-templates",
                exclusive=True,
            )
        except Exception:
            # A worker-scheduling failure must never break the canvas.
            return

    async def _fetch_chunk_templates(self) -> None:
        """Query the live chunking-template names via the scope service.

        Reaches for the app's ``rag_admin_scope_service`` (local mode) and
        degrades quietly -- a missing service, a policy denial, or a store
        error leaves the picker at its "None (manual settings)" default
        rather than breaking the ingest form. Applies the fetched names to
        the LIVE select in place (``set_options``); recomposes re-read the
        cache at compose time, so no structural update is needed.
        """
        service = getattr(self.app, "rag_admin_scope_service", None)
        list_templates = getattr(service, "list_templates", None)
        if not callable(list_templates):
            return
        try:
            records = await list_templates(mode="local")
        except Exception:
            return
        names: list[str] = []
        for record in records or []:
            name = str((record or {}).get("name") or "").strip()
            if name and name not in names:
                names.append(name)
        self._chunk_template_names = names
        try:
            picker = self.query_one(f"#{INGEST_CHUNK_TEMPLATE_PICKER_ID}", Select)
        except NoMatches:
            return  # server mode (or mid-recompose): the cache has it
        options = [
            (INGEST_CHUNK_TEMPLATE_NONE_LABEL, INGEST_CHUNK_TEMPLATE_NONE_VALUE),
            *[(escape_markup(name), name) for name in names],
        ]
        # Preserve the current choice; ``set_options`` resets the value only
        # when it disappears (a deleted template falls back to None here,
        # and §9.1's not-found ruling fires at submit for stale snapshots).
        if picker.value not in {value for _label, value in options}:
            self._reported_option_values[
                ("generic", INGEST_CHUNK_TEMPLATE_FIELD)
            ] = INGEST_CHUNK_TEMPLATE_NONE_VALUE
        picker.set_options(options)

    def on_resize(self, _event: Any) -> None:
        """A viewport change can (un)cover the fold -- re-derive the hint."""
        self.sync_fold_hint()

    def sync_fold_hint(self) -> None:
        """Show the fold indicator only while the canvas content overflows.

        (task-3304, MI-08) Mirrors Settings' task-1623 fold row: sizes are
        read from the laid-out container, so callers route through
        ``call_after_refresh`` when a recompose is in flight. Safe to call
        any time; a missing hint (mid-recompose) degrades silently.
        """
        try:
            hint = self.query_one("#library-ingest-fold-hint", Static)
        except NoMatches:
            return
        hint.display = (
            self.virtual_size.height > self.container_size.height
        )

    @on(Button.Pressed, ".ingest-preflight-copy-command")
    def _copy_preflight_install_command(self, event: Button.Pressed) -> None:
        """Copy one warning's install command from the summary (MI-17).

        Mirrors the guardrail modal's copy button (same seam, same
        notifications) so the modal is no longer the only place the
        command can be recovered from.
        """
        event.stop()
        button_id = event.button.id or ""
        if button_id == INGEST_COPY_ALL_COMMANDS_ID:
            # (task-14822) The union of the missing extras, so a nine-warning
            # selection is one paste rather than nine.
            command = combined_install_command(self.state.warning_commands)
            if not command:
                return
        else:
            prefix = "ingest-preflight-copy-command-"
            if not button_id.startswith(prefix):
                return
            try:
                index = int(button_id[len(prefix):])
                command = self.state.warning_commands[index]
            except (ValueError, IndexError):
                return
        copy_fn = getattr(self.app, "copy_to_clipboard", None)
        if callable(copy_fn):
            try:
                copy_fn(command)
                self.notify("Install command copied to clipboard")
            except Exception:
                self.notify("Failed to copy command", severity="error")
        else:
            self.notify("Clipboard not available", severity="warning")

    @on(Checkbox.Changed)
    @on(Select.Changed)
    @on(Input.Changed)
    @on(TextArea.Changed)
    def _handle_option_value_changed(
        self,
        event: Checkbox.Changed | Select.Changed | Input.Changed | TextArea.Changed,
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
            "text_area",
            getattr(
                event,
                "checkbox",
                getattr(event, "select", getattr(event, "input", None)),
            ),
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
        value = widget.text if isinstance(widget, TextArea) else event.value
        if key in self._reported_option_values and self._reported_option_values[key] == value:
            return
        self._reported_option_values[key] = value
        self.post_message(self.OptionValueChanged(group, name, value))

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

    @on(Button.Pressed, "#library-external-prepare-cancel")
    def _request_external_preparation_cancel(self, event: Button.Pressed) -> None:
        """Bubble explicit cancellation to the owning screen."""

        event.stop()
        self.post_message(self.ExternalPreparationCancelRequested())

    @on(Button.Pressed, ".library-ingest-directory-browse")
    def _request_directory_browse(self, event: Button.Pressed) -> None:
        """Bubble an adjacent directory picker request to the screen."""

        event.stop()
        button_id = event.button.id or ""
        suffix = "-browse"
        if not button_id.startswith("opt-") or not button_id.endswith(suffix):
            return
        group, separator, name = button_id[4 : -len(suffix)].partition("-")
        if separator and group and name:
            self.post_message(self.DirectoryBrowseRequested(group, name))

    @on(Button.Pressed, "#opt-audio_video-choose-transcribe-cpp-gguf")
    def _request_transcribe_cpp_gguf(self, event: Button.Pressed) -> None:
        """Request the shared local-GGUF picker from the owning screen."""
        event.stop()
        self.post_message(self.TranscribeCppGGUFRequested())
