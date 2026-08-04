# Logs_Window.py
# Description: Logs destination UI — diagnose-and-share loop.
#
# The user journey this screen serves: something misbehaves -> the user
# comes here -> filters to the relevant lines -> copies them -> shares
# them when asking for help. Layout rule: nothing ever occludes log
# content; the copy actions live in their own bottom bar.
#
# Imports
import re
from collections import deque
from typing import TYPE_CHECKING, Iterable, NamedTuple, Optional

#
# 3rd-Party Imports
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
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
        return Text(record.message, style="bold bright_red")
    if record.level == "WARNING":
        return Text(record.message, style="bright_yellow")
    return Text(record.message)


class LogsWindow(Container):
    """Logs destination: filter bar, log view, status line, action bar."""

    DEFAULT_CSS = """
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
        Binding("y", "copy_visible", "Copy visible", show=False),
    ]

    #: Footer hint context for the Logs screen (registered by LogsScreen).
    LOGS_SHORTCUTS: tuple[tuple[str, str], ...] = (
        ("/", "filter"),
        ("1-4", "level"),
        ("p", "pause"),
        ("y", "copy"),
    )

    def __init__(self, app_instance: "TldwCli", **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self._records: deque[LogRecord] = deque(maxlen=MAX_LOG_RECORDS)
        self._level_chip = "all"
        self._paused = False
        self._pending_while_paused = 0
        self._rendered_count = 0
        self._loaded_from_app = False

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Horizontal(id="logs-filter-bar"):
            for chip_id, label, _levels in _LEVEL_FILTERS:
                yield Button(
                    label,
                    id=f"logs-filter-{chip_id}",
                    classes="logs-filter-chip" + (" is-active" if chip_id == "all" else ""),
                )
            yield Input(placeholder="Filter logs (regex ok)…", id="logs-filter-text")
            yield Button("Pause", id="logs-pause")
        yield Static(
            "No log entries yet.\n"
            "Something not working? Reproduce the problem, then copy the logs "
            "and share them when asking for help.",
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
                "Copy all",
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
                self.query_one("#logs-filter-text", Input).value = saved_text
        except Exception:  # noqa: BLE001 - config read must never block logs
            pass
        app_records: Iterable[tuple] = getattr(
            self.app_instance, "_log_records", ()
        ) or ()
        for entry in app_records:
            level, name, message = entry
            self._records.append(LogRecord(level, name, message))
        self._render_view()

    def save_filter_state(self) -> None:
        """Persist the current filter text and level chip (UX-077)."""
        try:
            from ..config import save_setting_to_cli_config

            save_setting_to_cli_config(
                "logs", "last_filter", self.query_one("#logs-filter-text", Input).value
            )
            save_setting_to_cli_config("logs", "last_level_chip", self._level_chip)
        except Exception:  # noqa: BLE001 - config write must never break navigation
            pass

    def on_unmount(self) -> None:
        """Save the filter state when the screen is left."""
        self.save_filter_state()

    def append_record(self, level: str, name: str, message: str) -> None:
        """Receive one live log record from the app's logging handler."""
        record = LogRecord(level, name, message)
        was_empty = not self._records
        self._records.append(record)
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

    @staticmethod
    def _compile_pattern(text: str) -> "re.Pattern | None":
        """Compile the filter text as a regex; invalid input falls back to
        plain substring matching (None means: use substring)."""
        if not text:
            return None
        try:
            return re.compile(text, re.IGNORECASE)
        except re.error:
            return None

    def _visible_records(self) -> list[LogRecord]:
        text = self.query_one("#logs-filter-text", Input).value
        pattern = self._compile_pattern(text)
        return [
            record
            for record in self._records
            if _passes_filter(record, self._level_chip, text, pattern)
        ]

    def _render_view(self) -> None:
        """Re-render the log view from the record buffer."""
        log_widget = self.query_one("#app-log-display", RichLog)
        empty_state = self.query_one("#logs-empty-state", Static)
        if not self._records:
            self._rendered_count = 0
            empty_state.display = "block"
            log_widget.display = False
        else:
            empty_state.display = "none"
            log_widget.display = True
            log_widget.clear()
            visible = self._visible_records()
            for record in visible:
                log_widget.write(_styled_line(record))
            self._rendered_count = len(visible)
            log_widget.scroll_end()
        self._update_status_line()
        self._update_filter_chips()
        self._update_header_chip()

    def _update_filter_chips(self) -> None:
        """Refresh chip active states and per-level counts."""
        counts = {chip_id: 0 for chip_id, _label, _ in _LEVEL_FILTERS}
        counts["all"] = len(self._records)
        for record in self._records:
            for chip_id, _label, levels in _LEVEL_FILTERS:
                if levels and record.level in levels:
                    counts[chip_id] += 1
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
        """Honest accounting of what's shown, filtered, and paused."""
        total = len(self._records)
        shown = self._rendered_count
        parts = [f"Showing {shown} of {total} lines"]
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
        errors = sum(
            1 for record in self._records if record.level in ("ERROR", "CRITICAL")
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
            self.save_filter_state()

    @on(Input.Changed, "#logs-filter-text")
    def _on_filter_text_changed(self, event: Input.Changed) -> None:
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
        self.app.notify(
            f"Copied {len(records)} visible log lines to clipboard!",
            title="Clipboard",
            severity="information",
            timeout=4,
        )

    @on(Button.Pressed, "#copy-logs-button")
    def _on_copy_all(self) -> None:
        """Copy the full session log (unbounded buffer) to the clipboard."""
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
            f"Copied {len(buffer)} log entries to clipboard!",
            title="Clipboard",
            severity="information",
            timeout=4,
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
        """Switch the level filter chip (1-4 keys)."""
        if chip_id != self._level_chip:
            self._level_chip = chip_id
            self._render_view()

    def action_copy_visible(self) -> None:
        """Copy the visible lines (y key)."""
        self._on_copy_visible()


#
# End of Logs_Window.py
#######################################################################################################################
