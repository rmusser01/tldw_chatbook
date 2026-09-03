"""Projection-only Terminal workspace and keyboard-local viewport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.style import Style
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.Terminal.contracts import (
    MAX_COLUMNS,
    MAX_ROWS,
    MAX_SESSION_RECORDS,
    MIN_COLUMNS,
    MIN_ROWS,
    TerminalLifecycle,
)
from tldw_chatbook.Terminal.screen_model import (
    SafeTerminalLine,
    SafeTerminalStyle,
    TerminalScreenSnapshot,
)
from tldw_chatbook.Terminal.session_manager import (
    TerminalSessionView,
    TerminalViewState,
)
from tldw_chatbook.Widgets.Console.console_terminal_messages import (
    ConsoleTerminalActionRequested,
    ConsoleTerminalInputRequested,
    ConsoleTerminalResizeRequested,
    TerminalAction,
)

_GLOBAL_KEYS = frozenset({"ctrl+p", "ctrl+q", "f1", "f6"})
_RELEASE_KEY = "ctrl+right_square_bracket"
_WORKSPACE_GRID_COLUMNS = 6
_FIXED_KEY_BYTES = {
    "tab": b"\t",
    "enter": b"\r",
    "escape": b"\x1b",
    "backspace": b"\x7f",
    "up": b"\x1b[A",
    "down": b"\x1b[B",
    "right": b"\x1b[C",
    "left": b"\x1b[D",
    "home": b"\x1b[H",
    "end": b"\x1b[F",
    "insert": b"\x1b[2~",
    "pageup": b"\x1b[5~",
    "pagedown": b"\x1b[6~",
    "delete": b"\x1b[3~",
    "shift+tab": b"\x1b[Z",
    "backtab": b"\x1b[Z",
    "f2": b"\x1bOQ",
    "f3": b"\x1bOR",
    "f4": b"\x1bOS",
    "f5": b"\x1b[15~",
    "f7": b"\x1b[18~",
    "f8": b"\x1b[19~",
    "f9": b"\x1b[20~",
    "f10": b"\x1b[21~",
    "f11": b"\x1b[23~",
    "f12": b"\x1b[24~",
}


def terminal_key_bytes(key: str, character: str | None) -> bytes | None:
    """Encode one supported focused-viewport key without shadowing globals.

    Args:
        key: Textual key identifier for the input event.
        character: Textual character payload when the event carries text.

    Returns:
        Encoded terminal input, or ``None`` when the key must bubble locally.
    """
    if key in _GLOBAL_KEYS or key == _RELEASE_KEY:
        return None
    fixed = _FIXED_KEY_BYTES.get(key)
    if fixed is not None:
        return fixed
    if key.startswith("ctrl+") and len(key) == 6:
        letter = key[-1].lower()
        if "a" <= letter <= "z":
            return bytes((ord(letter) - ord("a") + 1,))
    if key.startswith("alt+"):
        value = character if character else key.removeprefix("alt+")
        return b"\x1b" + value.encode("utf-8") if value else None
    return character.encode("utf-8") if character else None


@dataclass(slots=True)
class _ViewportSessionState:
    snapshot: TerminalScreenSnapshot
    history_offset: int = 0
    new_output_count: int = 0
    frozen: Text | None = None


def _rich_style(style: SafeTerminalStyle) -> Style:
    """Convert a bounded safe style, dropping unsupported color names."""
    color = None if style.fg == "default" else style.fg
    background = None if style.bg == "default" else style.bg
    try:
        return Style(
            color=color,
            bgcolor=background,
            bold=style.bold,
            italic=style.italics,
            underline=style.underscore,
            strike=style.strikethrough,
            reverse=style.reverse,
            blink=style.blink,
        )
    except Exception:
        return Style(
            bold=style.bold,
            italic=style.italics,
            underline=style.underscore,
            strike=style.strikethrough,
            reverse=style.reverse,
            blink=style.blink,
        )


def _render_lines(lines: tuple[SafeTerminalLine, ...]) -> Text:
    rendered = Text()
    for line_index, line in enumerate(lines):
        if line_index:
            rendered.append("\n")
        for run in line.runs:
            rendered.append(run.text, style=_rich_style(run.style))
    return rendered


class TerminalViewport(Static):
    """Render immutable safe cells and own only local history/focus state."""

    can_focus = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(Text(), *args, **kwargs)
        self.input_focused = True
        self.history_offset = 0
        self.new_output_count = 0
        self.status_text = ""
        self._session_id: str | None = None
        self._snapshot: TerminalScreenSnapshot | None = None
        self._states: dict[str, _ViewportSessionState] = {}

    @property
    def page_size(self) -> int:
        """Return one terminal page in retained lines."""
        return max(1, len(self._snapshot.lines) if self._snapshot else 1)

    def project(self, *, session_id: str, snapshot: TerminalScreenSnapshot) -> None:
        """Project one selected immutable snapshot without parsing raw output."""
        if self._session_id is not None and self._snapshot is not None:
            self._states[self._session_id] = _ViewportSessionState(
                snapshot=self._snapshot,
                history_offset=self.history_offset,
                new_output_count=self.new_output_count,
                frozen=(
                    self.renderable.copy()
                    if isinstance(self.renderable, Text)
                    else None
                ),
            )

        previous_state = self._states.get(session_id)
        switching = session_id != self._session_id
        if switching:
            self._session_id = session_id
            self.input_focused = True
            self.status_text = ""
            self.history_offset = (
                previous_state.history_offset if previous_state is not None else 0
            )
            self.new_output_count = (
                previous_state.new_output_count if previous_state is not None else 0
            )

        previous = (
            previous_state.snapshot
            if switching and previous_state is not None
            else self._snapshot
        )
        added_scrollback, new_output = _snapshot_delta(previous, snapshot)
        if previous is not None and snapshot.generation != previous.generation:
            if self.history_offset > 0:
                self.history_offset = min(
                    len(snapshot.scrollback), self.history_offset + added_scrollback
                )
                self.new_output_count += new_output
                if not switching and isinstance(self.renderable, Text):
                    frozen = self.renderable.copy()
                else:
                    frozen = previous_state.frozen if previous_state else None
            elif switching:
                self.new_output_count += new_output
                frozen = None
            else:
                frozen = None
        else:
            frozen = previous_state.frozen if switching and previous_state else None

        self._snapshot = snapshot
        if self.history_offset > 0 and frozen is not None:
            self.update(frozen)
        else:
            self._render_current()
        self._remember()
        self._notify_workspace()

    def release_input(self) -> None:
        """Enter keyboard-local history navigation."""
        self.input_focused = False
        self.status_text = ""
        self._notify_workspace()

    def focus_input(self) -> None:
        """Return keyboard ownership to the terminal viewport."""
        self.input_focused = True
        self.status_text = ""
        self.focus()
        self._notify_workspace()

    def jump_live(self) -> None:
        """Return to the active screen without dropping retained state."""
        if self._alternate_history_noop():
            return
        self.history_offset = 0
        self.new_output_count = 0
        self.status_text = ""
        self._render_current()
        self._remember()
        self._notify_workspace()

    def scroll_up(self, lines: int = 1) -> None:
        """Move toward oldest normal-screen history."""
        snapshot = self._snapshot
        if snapshot is None:
            return
        if self._alternate_history_noop():
            return
        self.history_offset = min(
            len(snapshot.scrollback), self.history_offset + max(1, lines)
        )
        self._render_current()
        self._remember()
        self._notify_workspace()

    def scroll_down(self, lines: int = 1) -> None:
        """Move toward the active screen."""
        if self._alternate_history_noop():
            return
        self.history_offset = max(0, self.history_offset - max(1, lines))
        if self.history_offset == 0:
            self.new_output_count = 0
        self._render_current()
        self._remember()
        self._notify_workspace()

    def on_key(self, event: Any) -> None:
        """Route focused input or released-view navigation without stealing globals.

        Args:
            event: Textual key event to forward, handle locally, or leave bubbling.
        """
        key = event.key
        if key in _GLOBAL_KEYS:
            return
        if self.input_focused:
            if key == _RELEASE_KEY:
                self.release_input()
                _consume(event)
                return
            data = terminal_key_bytes(key, getattr(event, "character", None))
            if data is not None:
                self.post_message(ConsoleTerminalInputRequested(data))
                _consume(event)
            return

        if key == "tab":
            return
        if key == "enter":
            self.focus_input()
            _consume(event)
            return
        if key == "up":
            self.scroll_up()
        elif key == "down":
            self.scroll_down()
        elif key == "pageup":
            self.scroll_up(self.page_size)
        elif key == "pagedown":
            self.scroll_down(self.page_size)
        elif key == "home":
            if self._snapshot is not None:
                self.scroll_up(len(self._snapshot.scrollback))
        elif key == "end":
            self.jump_live()
        else:
            return
        _consume(event)

    def on_mouse_scroll_up(self, event: Any) -> None:
        self.release_input()
        self.scroll_up()
        event.stop()

    def on_mouse_scroll_down(self, event: Any) -> None:
        self.release_input()
        self.scroll_down()
        event.stop()

    def on_resize(self) -> None:
        """Report descendant-only layout changes using actual painted dimensions."""
        request = ConsoleTerminalResizeRequested(
            self.size.width,
            self.size.height,
            min_columns=MIN_COLUMNS,
            max_columns=MAX_COLUMNS,
            min_rows=MIN_ROWS,
            max_rows=MAX_ROWS,
        )
        workspace = self._workspace()
        if workspace is not None:
            workspace.update_painted_allocation(
                request.painted_width,
                request.painted_height,
            )
        self.post_message(request)

    def _render_current(self) -> None:
        snapshot = self._snapshot
        if snapshot is None:
            self.update(Text())
            return
        if self.history_offset == 0 or snapshot.in_alternate:
            lines = snapshot.lines
        else:
            retained = (*snapshot.scrollback, *snapshot.lines)
            end = max(0, len(retained) - self.history_offset)
            start = max(0, end - self.page_size)
            lines = tuple(retained[start:end])
        self.update(_render_lines(lines))

    def _alternate_history_noop(self) -> bool:
        snapshot = self._snapshot
        if snapshot is None or not snapshot.in_alternate:
            return False
        self.status_text = "Alternate screen has no local scrollback."
        self._notify_workspace()
        return True

    def _remember(self) -> None:
        if self._session_id is None or self._snapshot is None:
            return
        self._states[self._session_id] = _ViewportSessionState(
            snapshot=self._snapshot,
            history_offset=self.history_offset,
            new_output_count=self.new_output_count,
            frozen=(
                self.renderable.copy() if isinstance(self.renderable, Text) else None
            ),
        )

    def _notify_workspace(self) -> None:
        workspace = self._workspace()
        if workspace is not None:
            workspace._sync_viewport_status()

    def _workspace(self) -> ConsoleTerminalWorkspace | None:
        parent = self.parent
        while parent is not None:
            if isinstance(parent, ConsoleTerminalWorkspace):
                return parent
            parent = parent.parent
        return None


def _snapshot_delta(
    previous: TerminalScreenSnapshot | None,
    current: TerminalScreenSnapshot,
) -> tuple[int, int]:
    if previous is None or current.generation == previous.generation:
        return 0, 0
    added_scrollback = max(0, len(current.scrollback) - len(previous.scrollback))
    return added_scrollback, added_scrollback + len(current.dirty_lines)


def _consume(event: Any) -> None:
    event.stop()
    event.prevent_default()


class ConsoleTerminalWorkspace(Static):
    """Render manager-owned immutable state without owning terminal resources."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._projection = (False, False, TerminalViewState())
        self._selected_session_id: str | None = None
        self._session_ids: tuple[str, ...] = ()
        self._controller_status = ""
        self._painted_allocation: tuple[int, int] | None = None

    def compose(self) -> ComposeResult:
        yield Static("", id="console-terminal-danger", markup=False)
        yield Static("Terminal is locked.", id="console-terminal-access", markup=False)
        yield Button("Open Privacy & Security", id="console-terminal-open-settings")
        yield Button("Arm Terminal", id="console-terminal-arm")
        for index in range(MAX_SESSION_RECORDS):
            yield Button("", id=f"console-terminal-session-{index}")
        yield Button("New", id="console-terminal-new")
        yield Static("", id="console-terminal-metadata", markup=False)
        yield TerminalViewport(id="console-terminal-viewport")
        yield Button("Rename", id="console-terminal-rename")
        yield Button("Focus", id="console-terminal-focus")
        yield Button("Close", id="console-terminal-close")
        yield Button("Retry cleanup", id="console-terminal-retry")
        yield Button("Jump live", id="console-terminal-jump-live")
        yield Button("Return to transcript", id="console-terminal-return")
        yield Static("", id="console-terminal-status", markup=False)

    def on_mount(self) -> None:
        self._apply_projection()

    def project(
        self,
        *,
        permitted: bool,
        armed: bool,
        view_state: TerminalViewState,
    ) -> None:
        """Render one immutable manager view without recomposing the viewport."""
        self._projection = (permitted is True, armed is True, view_state)
        if self.is_mounted:
            self._apply_projection()

    def terminal_size(self) -> tuple[int, int]:
        """Return the visible viewport allocation within terminal bounds."""
        columns, rows, _clamped = self._terminal_allocation()
        return columns, rows

    def _terminal_allocation(self) -> tuple[int, int, bool]:
        if self._painted_allocation is not None:
            width, height = self._painted_allocation
        elif self.is_mounted:
            viewport = self.query_one("#console-terminal-viewport", TerminalViewport)
            width, height = viewport.size.width, viewport.size.height
        else:
            width, height = 80, 24
        columns = min(MAX_COLUMNS, max(MIN_COLUMNS, width or 80))
        rows = min(MAX_ROWS, max(MIN_ROWS, height or 24))
        return columns, rows, width > MAX_COLUMNS or height > MAX_ROWS

    def update_painted_allocation(self, width: int, height: int) -> None:
        """Refresh allocation-only metadata without changing manager state."""
        self._painted_allocation = (width, height)
        if not self.is_mounted:
            return
        selected = _selected_session(self._projection[2])
        if (
            selected is not None
            and selected.projection.session_id != self._selected_session_id
        ):
            selected = None
        self._sync_selected_metadata(selected)

    def jump_live(self) -> None:
        if self.is_mounted:
            self.query_one("#console-terminal-viewport", TerminalViewport).jump_live()

    def focus_terminal(self) -> None:
        if self.is_mounted:
            self.query_one("#console-terminal-viewport", TerminalViewport).focus_input()

    def set_status(self, message: str) -> None:
        """Show content-free controller feedback."""
        self._controller_status = message
        if self.is_mounted:
            self._sync_viewport_status()

    def _apply_projection(self) -> None:
        permitted, armed, view_state = self._projection
        danger = self.query_one("#console-terminal-danger", Static)
        access = self.query_one("#console-terminal-access", Static)
        settings = self.query_one("#console-terminal-open-settings", Button)
        arm = self.query_one("#console-terminal-arm", Button)
        danger.display = armed
        danger.update("HOST TERMINAL - FULL USER ACCESS" if armed else "")
        settings.display = not permitted
        arm.display = permitted and not armed
        if not permitted:
            access.update(
                "Terminal is locked. Unlock host access in Settings > Privacy & Security."
            )
        elif not armed:
            access.update(
                "Terminal is unlocked but not armed for this Chatbook launch."
            )
        else:
            access.update("Terminal content is user-only and is not sent to a model.")

        sessions = view_state.sessions
        visible_sessions = (
            sessions
            if armed
            else tuple(
                session
                for session in sessions
                if session.projection.lifecycle is TerminalLifecycle.CLEANUP_UNPROVEN
            )
        )
        self._session_ids = tuple(
            session.projection.session_id
            for session in visible_sessions[:MAX_SESSION_RECORDS]
        )
        for index in range(MAX_SESSION_RECORDS):
            button = self.query_one(f"#console-terminal-session-{index}", Button)
            if index < len(visible_sessions):
                session = visible_sessions[index]
                button.label = session.projection.name
                button.display = True
                button.variant = (
                    "primary"
                    if session.projection.session_id == view_state.selected_session_id
                    else "default"
                )
            else:
                button.label = ""
                button.display = False

        new = self.query_one("#console-terminal-new", Button)
        new.display = armed
        new.disabled = len(sessions) >= MAX_SESSION_RECORDS
        selected = _selected_session(view_state)
        if selected not in visible_sessions:
            selected = None
        self._selected_session_id = (
            selected.projection.session_id if selected is not None else None
        )
        self._project_selected(selected, armed=armed)
        self._fill_control_rows()

    def _fill_control_rows(self) -> None:
        """Keep full-span content aligned after each partial control row."""
        session_controls = [
            self.query_one(f"#console-terminal-session-{index}", Button)
            for index in range(MAX_SESSION_RECORDS)
        ]
        session_controls.append(self.query_one("#console-terminal-new", Button))
        bottom_controls = [
            self.query_one(f"#console-terminal-{name}", Button)
            for name in ("rename", "focus", "close", "retry", "jump-live", "return")
        ]
        for controls in (session_controls, bottom_controls):
            for control in controls:
                control.styles.column_span = 1
            visible = [control for control in controls if control.display]
            if visible:
                visible[-1].styles.column_span = (
                    _WORKSPACE_GRID_COLUMNS - len(visible) + 1
                )

    def _project_selected(
        self, selected: TerminalSessionView | None, *, armed: bool
    ) -> None:
        viewport = self.query_one("#console-terminal-viewport", TerminalViewport)
        rename = self.query_one("#console-terminal-rename", Button)
        focus = self.query_one("#console-terminal-focus", Button)
        close = self.query_one("#console-terminal-close", Button)
        retry = self.query_one("#console-terminal-retry", Button)
        jump = self.query_one("#console-terminal-jump-live", Button)
        has_selected = armed and selected is not None
        has_cleanup_receipt = (
            selected is not None
            and selected.projection.lifecycle is TerminalLifecycle.CLEANUP_UNPROVEN
        )
        for action in (rename, focus, jump):
            action.display = has_selected
        if selected is None:
            close.display = False
            retry.display = False
            self._sync_selected_metadata(None)
            viewport.update(Text())
            self._sync_viewport_status()
            return

        self._sync_selected_metadata(selected)
        retry.display = has_cleanup_receipt
        close.display = has_selected and not has_cleanup_receipt
        viewport.project(
            session_id=selected.projection.session_id,
            snapshot=selected.screen,
        )
        self._sync_viewport_status()

    def _sync_selected_metadata(
        self,
        selected: TerminalSessionView | None,
    ) -> None:
        metadata = self.query_one("#console-terminal-metadata", Static)
        if selected is None:
            metadata.update("No terminal session selected.")
            return
        lifecycle = selected.projection.lifecycle
        metadata_text = (
            f"{selected.projection.name} · {lifecycle.value} · {selected.shell} · "
            f"{selected.start_directory} · {selected.columns}×{selected.rows}"
        )
        columns, rows, clamped = self._terminal_allocation()
        if clamped:
            metadata_text += f" · viewport capped at {columns}×{rows}"
        metadata.update(metadata_text)

    def _sync_viewport_status(self) -> None:
        if not self.is_mounted:
            return
        viewport = self.query_one("#console-terminal-viewport", TerminalViewport)
        if self._controller_status:
            status = self._controller_status
        elif viewport.status_text:
            status = viewport.status_text
        elif viewport.input_focused:
            status = "Ctrl+] Release input"
        else:
            status = (
                "↑/↓ scroll · PgUp/PgDn page · Home oldest · End live · Enter focus"
            )
        if viewport.new_output_count:
            status += f" · {viewport.new_output_count} new output"
        self.query_one("#console-terminal-status", Static).update(status)

    @on(Button.Pressed)
    def _request_action(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        action_by_id: dict[str, TerminalAction] = {
            "console-terminal-return": "return",
            "console-terminal-open-settings": "open-settings",
            "console-terminal-arm": "arm",
            "console-terminal-new": "new",
            "console-terminal-rename": "rename",
            "console-terminal-focus": "focus",
            "console-terminal-close": "close",
            "console-terminal-retry": "retry",
            "console-terminal-jump-live": "jump-live",
        }
        if button_id.startswith("console-terminal-session-"):
            try:
                index = int(button_id.rsplit("-", 1)[1])
                session_id = self._session_ids[index]
            except (IndexError, ValueError):
                return
            event.stop()
            self.post_message(ConsoleTerminalActionRequested("select", session_id))
            return
        action = action_by_id.get(button_id)
        if action is None:
            return
        event.stop()
        self.post_message(
            ConsoleTerminalActionRequested(action, self._selected_session_id)
        )


def _selected_session(view_state: TerminalViewState) -> TerminalSessionView | None:
    return next(
        (
            session
            for session in view_state.sessions
            if session.projection.session_id == view_state.selected_session_id
        ),
        None,
    )
