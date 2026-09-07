# Logs_Window.py
# Description: Logs destination UI — diagnose-and-share loop.
#
# The user journey this screen serves: something misbehaves -> the user
# comes here -> filters to the relevant lines -> copies them -> shares
# them when asking for help. Layout rule: nothing ever occludes log
# content; the copy actions live in their own bottom bar.
#
# Imports
import asyncio
import re
from collections import Counter, deque
from typing import TYPE_CHECKING, Iterable, NamedTuple, Optional

#
# 3rd-Party Imports
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.timer import Timer
from textual.widgets import Button, Input, RichLog, Static
from rich.text import Text

if TYPE_CHECKING:
    from ..app import TldwCli

#
########################################################################################################################
#
# Constants & record type:

#: Bounded record buffer mirroring app._log_records (kept in sync by the
#: app's PersistentLogHandler via ``append_record``).
MAX_LOG_RECORDS = 10000

#: Debounce for the free-text filter `Input` -- mirrors the picker/filter
#: family's 0.2 s shape (`console_prompt_picker_modal.py`). Every render
#: pass rescans up to `MAX_LOG_RECORDS` buffered records, so it must not run
#: on every keystroke (task-15476).
FILTER_DEBOUNCE_SECONDS = 0.2

#: Debounce for PERSISTING the saved-filter state (task-21124) -- distinct
#: from `FILTER_DEBOUNCE_SECONDS`, which debounces re-RENDERING. A level-chip
#: click used to fire two sequential synchronous `save_setting_to_cli_config`
#: calls on the event loop -- two full config.toml read-rewrite-reload cycles
#: (four fsyncs) per click, each holding the global config write lock. Chip
#: clicks now mark the filter state dirty and (re)arm this timer; the actual
#: write is ONE batched atomic mutation dispatched off the loop, and
#: `on_unmount` force-flushes any pending state (mirrors the task-15470
#: dictation-settings debounce shape, including its value).
LOGS_FILTER_SAVE_DEBOUNCE_SECONDS = 0.6

#: Cap the RichLog rendered slice: a filter matching thousands of buffered
#: records must not clear+rewrite the widget with all of them on every
#: render pass. The status line discloses when the cap trims output
#: (task-15476, AC #2); the most RECENT matches are kept, mirroring the
#: buffer's own oldest-evicted-first policy.
MAX_RENDERED_LINES = 1000

#: Level chips are THRESHOLDS, ordered by severity, matching the journalctl
#: convention: each chip shows its level and above. "Info+" hides DEBUG/TRACE
#: chatter; "Warnings+" and "Errors" surface what users come for (UX: the old
#: partition semantics let "Info+" hide warnings/errors entirely).
_LEVEL_FILTERS: tuple[tuple[str, str, frozenset[str]], ...] = (
    ("all", "All", frozenset()),
    ("info", "Info+", frozenset({"INFO", "WARNING", "ERROR", "CRITICAL"})),
    ("warning", "Warnings+", frozenset({"WARNING", "ERROR", "CRITICAL"})),
    ("error", "Errors", frozenset({"ERROR", "CRITICAL"})),
)


class LogRecord(NamedTuple):
    """One structured log entry for filtering and re-rendering."""

    level: str
    name: str
    message: str


def _passes_filter(
    record: LogRecord, level_chip: str, text: str, pattern: "re.Pattern | None" = None
) -> bool:
    """True when a record matches the active level chip and text/regex filter."""
    for chip_id, _label, levels in _LEVEL_FILTERS:
        if chip_id == level_chip:
            if levels and record.level not in levels:
                return False
            break
    if not text:
        return True
    if pattern is not None:
        return bool(pattern.search(record.message))
    return text.lower() in record.message.lower()


def _styled_line(record: LogRecord) -> Text:
    """Style a log line by level; the level word stays in the text itself,
    so color is a redundant scanner cue, never the only carrier (UX-075).
    Bright variants keep ERROR/WARNING legible on dark themes."""
    if record.level in ("ERROR", "CRITICAL"):
        return Text(_display_message(record), style="bold bright_red")
    if record.level == "WARNING":
        return Text(_display_message(record), style="bright_yellow")
    return Text(_display_message(record))


def _display_message(record: LogRecord) -> str:
    """Compact display form: short time, tail module segments, level, message.

    The full prefixed line is preserved in the record (copy actions use it);
    on screen, the message is the part that must survive the right edge.
    """
    parts = record.message.split(" - ", 3)
    if len(parts) != 4:
        return record.message
    timestamp, name, level, message = parts
    time_part = timestamp[11:19] if len(timestamp) >= 19 else timestamp
    short_name = ".".join(name.split(".")[-2:])
    return f"{time_part} {short_name} {level} {message}"


class LogsWindow(Container):
    """Logs destination: filter bar, log view, status line, action bar."""

    BUNDLED_CSS = """
    LogsWindow {
        layout: vertical;
    }
    #app-log-display {
        height: 1fr;
    }
    #logs-empty-state {
        display: none;
    }
    """

    # htop/less-style single letters (ADR-031). Printable keys are consumed
    # first by the focused filter Input, so they are safe everywhere else.
    BINDINGS = [
        Binding("/", "focus_filter", "Filter", show=False),
        Binding("p", "toggle_pause", "Pause", show=False),
        Binding("1", "level('all')", "All", show=False),
        Binding("2", "level('info')", "Info+", show=False),
        Binding("3", "level('warning')", "Warnings+", show=False),
        Binding("4", "level('error')", "Errors", show=False),
        Binding("n", "next_error", "Next error", show=False),
        Binding("N", "prev_error", "Previous error", show=False),
        Binding("y", "copy_visible", "Copy visible", show=False),
    ]

    #: Footer hint context for the Logs screen (registered by LogsScreen).
    LOGS_SHORTCUTS: tuple[tuple[str, str], ...] = (
        ("/", "filter"),
        ("1-4", "level"),
        ("p", "pause"),
        ("n", "next error"),
        ("y", "copy"),
    )

    def __init__(self, app_instance: "TldwCli", **kwargs):
        """Create the Logs window.

        Args:
            app_instance: The running TldwCli app; source of the buffered
                log records this window mirrors.
            **kwargs: Forwarded to ``Container`` (id, classes, …).
        """
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self._records: deque[LogRecord] = deque(maxlen=MAX_LOG_RECORDS)
        # Per-level counts of the buffered records, maintained incrementally
        # on append/evict so chip/header refreshes never rescan the buffer.
        self._level_counts: Counter[str] = Counter()
        # "Info+" is the front door: the level word and the message stay
        # visible without the DEBUG firehose (users can still hit "All").
        self._level_chip = "info"
        self._paused = False
        self._pending_while_paused = 0
        self._rendered_count = 0
        self._loaded_from_app = False
        # task-15476: how many records the active filter actually matched
        # (>= _rendered_count once MAX_RENDERED_LINES trims the render),
        # the records actually written to the RichLog on the last render
        # pass (n/N error-jump indexes against this, not the full matched
        # set, since that's all that's really on screen to scroll to), and
        # a one-entry cache so re-rendering with the same filter text does
        # not recompile the same regex.
        self._visible_total = 0
        self._last_rendered: list[LogRecord] = []
        self._compiled_pattern_cache: tuple[str, "re.Pattern | None"] | None = None
        self._filter_debounce_timer: Timer | None = None
        # task-21124: debounce state for the saved-filter persist -- see
        # `LOGS_FILTER_SAVE_DEBOUNCE_SECONDS`. `_persisted_filter_state` is
        # the last state known to be on disk (seeded by `load_from_app`);
        # comparing against it instead of a bare dirty flag means neither the
        # mount-time restore nor a click-and-click-back sequence produces a
        # write.
        self._filter_save_timer: Timer | None = None
        self._filter_persist_worker = None
        self._persisted_filter_state: dict[str, str] | None = None
        # Mirror of the filter Input's value, kept current by
        # `_on_filter_text_changed` and the `load_from_app` restore. The
        # persist snapshot reads THIS, never the DOM: during teardown the
        # Input may already be unmounted, and a DOM query degrading to ""
        # there made the unmount flush clobber the user's saved filter with
        # an empty string (caught by test_saved_filter_roundtrip while
        # building task-21124).
        self._filter_text = ""

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Horizontal(id="logs-filter-bar"):
            for chip_id, label, _levels in _LEVEL_FILTERS:
                yield Button(
                    label,
                    id=f"logs-filter-{chip_id}",
                    classes="logs-filter-chip"
                    + (" is-active" if chip_id == "info" else ""),
                )
            yield Input(placeholder="Filter logs (regex ok)…", id="logs-filter-text")
            yield Button("Pause", id="logs-pause")
        # TASK-19555: this text used to say, flatly, "copy the logs and share
        # them when asking for help" -- an invitation to put an unfiltered
        # session transcript on the clipboard. Credentials and the account
        # name are now stripped at the sink, but file names, note titles and
        # search terms are still in there, so the invitation says what it is
        # inviting and points at the action the user can actually read first.
        yield Static(
            "No log entries yet.\n"
            "Something not working? Reproduce the problem, filter to the "
            "lines that matter, then use Copy visible logs — you share "
            "exactly what you can see.\n"
            "Recognised API-key formats and your account name are removed; "
            "file names, titles and search terms are not, so read before you "
            "share. Copy all (redacted) shares timings, loggers and error "
            "types only.",
            id="logs-empty-state",
        )
        yield RichLog(
            id="app-log-display",
            wrap=False,
            highlight=True,
            markup=False,  # Prevent log messages from being interpreted as markup
            auto_scroll=True,
            max_lines=MAX_LOG_RECORDS,
        )
        yield Static("", id="logs-status-line")
        with Horizontal(id="logs-action-bar"):
            yield Button(
                "Copy visible logs",
                id="copy-visible-logs-button",
                classes="logs-action-button",
                variant="primary",
            )
            yield Button(
                # The label must not promise more than the artifact carries
                # (TASK-19555): "Copy all" now yields the metadata-only form.
                "Copy all (redacted)",
                id="copy-logs-button",
                classes="logs-action-button",
            )

    # ------------------------------------------------------------------
    # Data intake
    # ------------------------------------------------------------------
    def load_from_app(self) -> None:
        """Seed the local record buffer from the app's persistent records."""
        if self._loaded_from_app:
            return
        self._loaded_from_app = True
        # Restore the last-used filter and level chip (UX-077 saved filters).
        try:
            from ..config import get_cli_setting

            saved_chip = get_cli_setting("logs", "last_level_chip", "all")
            if saved_chip in {chip_id for chip_id, _l, _lv in _LEVEL_FILTERS}:
                self._level_chip = saved_chip
            saved_text = get_cli_setting("logs", "last_filter", "")
            if saved_text:
                self._filter_text = saved_text
                self.query_one("#logs-filter-text", Input).value = saved_text
            # Baseline for change detection (task-21124): what we just
            # restored is, by definition, what is persisted -- so neither
            # the restore itself nor an unmount without edits writes.
            self._persisted_filter_state = self._filter_state_snapshot()
        except Exception:  # noqa: BLE001 - config read must never block logs
            pass
        app_records: Iterable[tuple] = getattr(
            self.app_instance, "_log_records", ()
        ) or ()
        for entry in app_records:
            level, name, message = entry
            self._records.append(LogRecord(level, name, message))
        # The deque may have evicted oldest-first while seeding; rebuild the
        # counts from the final buffer state (one pass, load-time only).
        self._level_counts = Counter(record.level for record in self._records)
        self._render_view()

    def _filter_state_snapshot(self) -> dict[str, str]:
        """Capture the persistable filter state on the event-loop thread.

        Prefers the live Input (an un-dispatched `Input.Changed` may not
        have reached the `_filter_text` mirror yet); falls back to the
        mirror when the Input is already unmounted at teardown, where a
        degrade-to-"" would clobber the user's saved filter (see
        `_filter_text`).
        """
        try:
            self._filter_text = self.query_one("#logs-filter-text", Input).value
        except Exception:  # noqa: BLE001 - teardown: mirror keeps last value
            pass
        return {
            "last_filter": self._filter_text,
            "last_level_chip": self._level_chip,
        }

    def _write_filter_state(self, snapshot: dict[str, str]) -> None:
        """Persist a pre-captured filter snapshot with ONE atomic write.

        task-21124: replaces two sequential `save_setting_to_cli_config`
        calls (two full config rewrites, four fsyncs) with one batched
        mutation. Safe to call from a worker thread: touches only the
        passed-in snapshot.
        """
        try:
            from ..config import save_settings_to_cli_config

            save_settings_to_cli_config({"logs": snapshot})
            self._persisted_filter_state = snapshot
        except Exception:  # noqa: BLE001 - config write must never break navigation
            pass

    def save_filter_state(self) -> None:
        """Persist the current filter text and level chip (UX-077), now.

        Synchronous, immediate form -- the debounced path
        (`_persist_filter_state`) is what UI event handlers use.
        """
        self._write_filter_state(self._filter_state_snapshot())

    def _persist_filter_state(self) -> None:
        """Schedule a debounced, batched, off-loop filter-state save.

        task-21124: the single gate chip clicks go through -- see
        `LOGS_FILTER_SAVE_DEBOUNCE_SECONDS`. A no-op when the current state
        already matches what is persisted (e.g. click away and back).
        """
        if self._filter_state_snapshot() == self._persisted_filter_state:
            return
        if self._filter_save_timer is not None:
            self._filter_save_timer.stop()
        self._filter_save_timer = self.set_timer(
            LOGS_FILTER_SAVE_DEBOUNCE_SECONDS,
            self._flush_filter_state_after_debounce,
        )

    def _flush_filter_state_after_debounce(self) -> None:
        """Debounce timer callback: hand the actual write to a worker."""
        self._filter_save_timer = None
        self._filter_persist_worker = self.run_worker(
            self._persist_filter_state_off_loop(),
            exclusive=True,
            group="logs-filter-persist",
        )

    async def _persist_filter_state_off_loop(self) -> None:
        """Write the filter state on a worker thread, off the event loop.

        Snapshots on the main thread before handing the write to
        `to_thread`, so a further chip click cannot race the worker's read
        (same shape as the task-15470 dictation persist).
        """
        snapshot = self._filter_state_snapshot()
        if snapshot == self._persisted_filter_state:
            return
        await asyncio.to_thread(self._write_filter_state, snapshot)

    async def on_unmount(self) -> None:
        """Flush any pending filter-state change when the screen is left.

        Also picks up filter-TEXT edits, which (as before task-21124) are
        persisted only at unmount -- but now only when the state actually
        changed, where the old code rewrote the config file on every exit
        from the Logs screen. If a debounced write is in flight, waits for
        it rather than dispatching a second writer against the same file.
        """
        if self._filter_save_timer is not None:
            self._filter_save_timer.stop()
            self._filter_save_timer = None
        # Capture before any await: after the wait the Input may be gone.
        snapshot = self._filter_state_snapshot()
        worker = self._filter_persist_worker
        if worker is not None and not worker.is_finished:
            try:
                await worker.wait()
            except Exception:  # noqa: BLE001 - flush must never break teardown
                pass
        if snapshot != self._persisted_filter_state:
            await asyncio.to_thread(self._write_filter_state, snapshot)

    def append_record(self, level: str, name: str, message: str) -> None:
        """Receive one live log record from the app's logging handler.

        Args:
            level: Log level name (e.g. "INFO", "ERROR").
            name: Name of the logger that emitted the record.
            message: Fully formatted log line, prefix included.
        """
        record = LogRecord(level, name, message)
        was_empty = not self._records
        if len(self._records) == self._records.maxlen:
            # The deque is full: appending evicts the oldest record, so its
            # level count must leave with it.
            self._level_counts[self._records[0].level] -= 1
        self._records.append(record)
        self._level_counts[record.level] += 1
        if was_empty:
            # Leaving the empty state: restore the log widget and re-render.
            self._render_view()
        # The header chip must track errors even while the view is paused.
        if record.level in ("ERROR", "CRITICAL"):
            self._update_header_chip()
        if self._paused:
            self._pending_while_paused += 1
            self._update_status_line()
            self._update_filter_chips()
            return
        if self._passes(record):
            self.query_one("#app-log-display", RichLog).write(_styled_line(record))
            self._rendered_count += 1
        self._update_status_line()
        self._update_filter_chips()

    # ------------------------------------------------------------------
    # Filtering & rendering
    # ------------------------------------------------------------------
    def _passes(self, record: LogRecord) -> bool:
        text = self.query_one("#logs-filter-text", Input).value
        return _passes_filter(
            record, self._level_chip, text, self._compile_pattern(text)
        )

    def _compile_pattern(self, text: str) -> "re.Pattern | None":
        """Compile the filter text as a regex; invalid input falls back to
        plain substring matching (None means: use substring).

        Cached on ``text`` (task-15476, AC #2): the level-chip buttons and
        the debounced text filter both re-render through this on every
        settle, and re-compiling the same pattern each time is pure waste.
        """
        cached = self._compiled_pattern_cache
        if cached is not None and cached[0] == text:
            return cached[1]
        if not text:
            pattern = None
        else:
            try:
                pattern = re.compile(text, re.IGNORECASE)
            except re.error:
                pattern = None
        self._compiled_pattern_cache = (text, pattern)
        return pattern

    def _visible_records(self) -> list[LogRecord]:
        text = self.query_one("#logs-filter-text", Input).value
        pattern = self._compile_pattern(text)
        return [
            record
            for record in self._records
            if _passes_filter(record, self._level_chip, text, pattern)
        ]

    def _render_view(self) -> None:
        """Re-render the log view from the record buffer.

        Caps the rendered slice to the most recent `MAX_RENDERED_LINES`
        filter matches (task-15476, AC #2): a filter matching thousands of
        the buffered records must not clear+rewrite the RichLog with all of
        them. `_update_status_line` discloses the truncation.
        """
        log_widget = self.query_one("#app-log-display", RichLog)
        empty_state = self.query_one("#logs-empty-state", Static)
        if not self._records:
            self._rendered_count = 0
            self._visible_total = 0
            self._last_rendered = []
            empty_state.display = "block"
            log_widget.display = False
        else:
            empty_state.display = "none"
            log_widget.display = True
            log_widget.clear()
            visible = self._visible_records()
            capped = (
                visible[-MAX_RENDERED_LINES:]
                if len(visible) > MAX_RENDERED_LINES
                else visible
            )
            for record in capped:
                log_widget.write(_styled_line(record))
            self._rendered_count = len(capped)
            self._visible_total = len(visible)
            self._last_rendered = capped
            log_widget.scroll_end()
        self._update_status_line()
        self._update_filter_chips()
        self._update_header_chip()

    def _update_filter_chips(self) -> None:
        """Refresh chip active states and per-level counts.

        Counts come from the incrementally maintained ``_level_counts`` —
        O(distinct levels) per refresh instead of rescanning up to
        MAX_LOG_RECORDS buffered records on every appended line.
        """
        counts = {"all": len(self._records)}
        for chip_id, _label, levels in _LEVEL_FILTERS:
            if levels:
                counts[chip_id] = sum(
                    self._level_counts.get(level, 0) for level in levels
                )
        for chip_id, label, _levels in _LEVEL_FILTERS:
            chip = self.query_one(f"#logs-filter-{chip_id}", Button)
            count = counts[chip_id]
            new_label = f"{label} ({count})" if count else label
            if str(chip.label) != new_label:
                chip.label = new_label
                # Textual 8.2.7 does not re-layout auto-width buttons on
                # label change; without this the grown label clips.
                chip.refresh(layout=True)
            chip.set_class(chip_id == self._level_chip, "is-active")

    def _update_status_line(self) -> None:
        """Honest accounting of what's shown, filtered, capped, and paused."""
        total = len(self._records)
        shown = self._rendered_count
        parts = [f"Showing {shown} of {total} lines"]
        if self._visible_total > shown:
            # task-15476 AC #2: the filter matched more than the rendered
            # cap -- say so, rather than silently showing a partial result.
            parts.append(
                f"(filter matched {self._visible_total}; "
                f"showing most recent {shown})"
            )
        if total >= MAX_LOG_RECORDS:
            parts.append(f"(buffer keeps last {MAX_LOG_RECORDS:,})")
        if self._paused:
            parts.append(f"— paused, {self._pending_while_paused} new")
        self.query_one("#logs-status-line", Static).update(" ".join(parts))

    def _update_header_chip(self) -> None:
        """Reflect buffer health in the screen's destination header chip."""
        try:
            from .Workbench.workbench_state import WorkbenchHeaderState
            from .Workbench.workbench_widgets import DestinationHeader

            header = self.screen.query_one(
                "#logs-destination-header", DestinationHeader
            )
        except Exception:  # noqa: BLE001 - header not mounted / no screen
            return
        errors = self._level_counts.get("ERROR", 0) + self._level_counts.get(
            "CRITICAL", 0
        )
        if errors:
            status, label = "error", f"{errors} error{'s' if errors != 1 else ''} in buffer"
        else:
            status, label = "ready", "Listening"
        header.sync_state(
            WorkbenchHeaderState(
                title="Logs",
                subtitle="Application logs and diagnostics.",
                status=status,
                status_label=label,
            )
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    @on(Button.Pressed, ".logs-filter-chip")
    def _on_filter_chip(self, event: Button.Pressed) -> None:
        chip_id = (event.button.id or "").replace("logs-filter-", "")
        if chip_id and chip_id != self._level_chip:
            self._level_chip = chip_id
            self._render_view()
            # task-21124: debounced, batched, off-loop -- never a
            # synchronous double config rewrite on the click.
            self._persist_filter_state()

    @on(Input.Changed, "#logs-filter-text")
    def _on_filter_text_changed(self, event: Input.Changed) -> None:
        """Debounced (task-15476): a render pass rescans up to
        `MAX_LOG_RECORDS` buffered records and clears+rewrites the RichLog,
        so it must not run on every keystroke."""
        self._filter_text = event.value
        if self._filter_debounce_timer is not None:
            self._filter_debounce_timer.stop()
        self._filter_debounce_timer = self.set_timer(
            FILTER_DEBOUNCE_SECONDS, self._apply_filter_text_debounced
        )

    def _apply_filter_text_debounced(self) -> None:
        self._filter_debounce_timer = None
        self._render_view()

    @on(Button.Pressed, "#logs-pause")
    def _on_pause_toggle(self, event: Button.Pressed) -> None:
        self._set_paused(not self._paused)

    def _set_paused(self, paused: bool) -> None:
        """Set the pause state and update the toggle button."""
        self._paused = paused
        button = self.query_one("#logs-pause", Button)
        new_label = "Resume" if paused else "Pause"
        if str(button.label) != new_label:
            button.label = new_label
            # Auto-width buttons do not re-layout on label change (Textual
            # 8.2.7); force it so "Resume" is not clipped to "Pause"'s width.
            button.refresh(layout=True)
        if not paused:
            self._pending_while_paused = 0
            self._render_view()
        else:
            self._update_status_line()

    @on(Button.Pressed, "#copy-visible-logs-button")
    def _on_copy_visible(self) -> None:
        """Copy the currently filtered/visible log lines to the clipboard."""
        records = self._visible_records()
        if not records:
            self.app.notify(
                "Nothing to copy — no log lines match the current filter.",
                title="Clipboard",
                severity="warning",
                timeout=4,
            )
            return
        self.app.copy_to_clipboard("\n".join(record.message for record in records))
        # TASK-19555: this is the deliberate, filtered action, so the payload
        # stays descriptive -- but the notification names the residual
        # exposure rather than leaving the user to discover it in a bug report.
        self.app.notify(
            f"Copied {len(records)} visible log lines. Recognised key formats "
            "and your account name were removed; file names and search terms "
            "were not.",
            title="Clipboard",
            severity="information",
            timeout=6,
        )

    @on(Button.Pressed, "#copy-logs-button")
    def _on_copy_all(self) -> None:
        """Copy the redacted session log to the clipboard.

        The app's ``PersistentLogHandler`` fills ``_log_buffer`` with the
        metadata-only form of each record (TASK-19555): this action exports
        thousands of lines the user has never read, so it carries timestamps,
        loggers, levels and exception types, and no message bodies. Sharing
        actual log text is the job of "Copy visible logs", where the user can
        see what they are sharing first.
        """
        buffer = getattr(self.app_instance, "_log_buffer", None)
        if not buffer:
            self.app.notify(
                "Log is empty, nothing to copy.",
                title="Clipboard",
                severity="warning",
                timeout=4,
            )
            return
        self.app.copy_to_clipboard("\n".join(buffer))
        self.app.notify(
            f"Copied {len(buffer)} redacted log entries — timings, loggers "
            "and error types only. Use Copy visible logs to share log text.",
            title="Clipboard",
            severity="information",
            timeout=6,
        )

    # ------------------------------------------------------------------
    # Keyboard actions (see BINDINGS)
    # ------------------------------------------------------------------
    def action_focus_filter(self) -> None:
        """Focus the filter input (/ key)."""
        self.query_one("#logs-filter-text", Input).focus()

    def action_toggle_pause(self) -> None:
        """Toggle pause/resume (p key)."""
        self._set_paused(not self._paused)

    def action_level(self, chip_id: str) -> None:
        """Switch the level filter chip (1-4 keys).

        Args:
            chip_id: One of the ``_LEVEL_FILTERS`` ids ("all", "info",
                "warning", "error").
        """
        if chip_id != self._level_chip:
            self._level_chip = chip_id
            self._render_view()

    def action_copy_visible(self) -> None:
        """Copy the visible lines (y key)."""
        self._on_copy_visible()

    def action_next_error(self) -> None:
        """Jump to the next error line (n key)."""
        self._jump_to_error(1)

    def action_prev_error(self) -> None:
        """Jump to the previous error line (N key)."""
        self._jump_to_error(-1)

    def _error_row_indices(self) -> list[int]:
        """Indices of error/critical records within the RENDERED view.

        Indexed against `_last_rendered` (what `_render_view` actually
        wrote to the RichLog), not the full filtered match set: when
        `MAX_RENDERED_LINES` trims the render, an index computed from the
        full match set could point past what the widget can `scroll_to`
        (task-15476).
        """
        return [
            index
            for index, record in enumerate(self._last_rendered)
            if record.level in ("ERROR", "CRITICAL")
        ]

    def _jump_to_error(self, direction: int) -> None:
        """Scroll to the next/previous error line (n / N keys)."""
        indices = self._error_row_indices()
        if not indices:
            self.app.notify("No errors in the current view.", severity="warning")
            return
        log_widget = self.query_one("#app-log-display", RichLog)
        current = int(log_widget.scroll_offset.y) if log_widget.scroll_offset else 0
        if direction > 0:
            target = next((i for i in indices if i > current), indices[0])
        else:
            target = next((i for i in reversed(indices) if i < current), indices[-1])
        log_widget.scroll_to(y=target, animate=False)


#
# End of Logs_Window.py
#######################################################################################################################
