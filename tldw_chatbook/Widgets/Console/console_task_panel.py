"""Pinned, read-only view of the agent's session task list (PRD Feature B).

The ``todo_*`` tools have kept a per-session ``SessionTodoStore`` since
TASK-13216, but the only UI was a transcript marker that scrolls away.
This panel sits above the transcript, mirrors the store on every change,
and stays put while the conversation moves.

Rendering reuses the transcript marker's glyphs and label sanitiser so
the two views can never disagree about what a task is called.
"""

from __future__ import annotations

from rich.markup import escape as _escape_markup
from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static

from tldw_chatbook.Chat.console_agent_bridge import _sanitize_task_marker_label

_GLYPHS = {"completed": "[x]", "in_progress": "[~]", "pending": "[ ]"}
_STATUS_CLASS = {
    "completed": "console-task-panel-done",
    "in_progress": "console-task-panel-active",
    "pending": "console-task-panel-pending",
}


def render_task_lines(
    tasks: list[dict[str, object]], *, collapsed: bool = False
) -> tuple[str, list[tuple[str, str]]]:
    """Derive the panel header and body rows from a task snapshot.

    Args:
        tasks: Task records as ``SessionTodoStore.list_after(None)`` returns
            them (``content``, ``status``, optional ``activeForm``).
        collapsed: Whether the body is hidden, which flips the header chevron.

    Returns:
        ``(header, rows)`` where ``header`` is the one-line summary
        (``Tasks · 3 of 7 done · Writing the migration``) and each row is a
        ``(status, text)`` pair -- ``text`` already carries the glyph.
    """
    done = sum(1 for task in tasks if task.get("status") == "completed")
    active = next(
        (task for task in tasks if task.get("status") == "in_progress"), None
    )
    parts = [f"Tasks · {done} of {len(tasks)} done"]
    if active is not None:
        parts.append(
            _sanitize_task_marker_label(
                str(active.get("activeForm") or active.get("content") or "")
            )
        )
    chevron = "▸" if collapsed else "▾"
    header = f"{chevron} " + " · ".join(parts)
    rows: list[tuple[str, str]] = []
    for task in tasks:
        status = str(task.get("status") or "pending")
        label = task.get("activeForm") if status == "in_progress" else None
        label = _sanitize_task_marker_label(str(label or task.get("content") or ""))
        rows.append((status, f"{_GLYPHS.get(status, '[ ]')} {label}"))
    return header, rows


class ConsoleTaskPanel(Vertical):
    """Collapsible task list pinned above the Console transcript.

    Hidden while the active session has no tasks (AC-B1). Collapse state is
    remembered per session for the widget's lifetime (AC-B6).
    """

    BUNDLED_CSS = """
    ConsoleTaskPanel {
        height: auto;
        max-height: 12;
        width: 1fr;
        border-top: solid $secondary;
        border-bottom: solid $secondary;
        padding: 0 1;
    }
    ConsoleTaskPanel > #console-task-panel-header {
        height: 1;
        text-style: bold;
    }
    ConsoleTaskPanel > #console-task-panel-header:hover {
        background: $boost;
    }
    ConsoleTaskPanel > #console-task-panel-body {
        height: auto;
        max-height: 10;
        overflow-y: auto;
        scrollbar-gutter: stable;
    }
    ConsoleTaskPanel .console-task-panel-row {
        height: auto;
        width: 1fr;
    }
    ConsoleTaskPanel .console-task-panel-done {
        color: $text-muted;
    }
    ConsoleTaskPanel .console-task-panel-active {
        color: $accent;
        text-style: bold;
    }
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.display = False
        self._session_id: str | None = None
        self._tasks: list[dict[str, object]] = []
        self._collapsed_by_session: dict[str, bool] = {}

    def compose(self) -> ComposeResult:
        yield Static("", id="console-task-panel-header")
        yield VerticalScroll(id="console-task-panel-body")

    @property
    def collapsed(self) -> bool:
        """Whether the current session's body is hidden."""
        return self._collapsed_by_session.get(self._session_id or "", False)

    def set_tasks(
        self, session_id: str | None, tasks: list[dict[str, object]]
    ) -> None:
        """Replace the rendered list with ``tasks`` for ``session_id``.

        Args:
            session_id: The session the snapshot belongs to; the panel
                renders whatever it is last told about and does not itself
                know which session is active.
            tasks: The session's full task list; empty hides the panel.
        """
        self._session_id = session_id
        self._tasks = list(tasks)
        self.display = bool(tasks)
        if not tasks:
            return
        self._repaint()

    def toggle_collapsed(self) -> None:
        """Flip the body's visibility for the current session."""
        key = self._session_id or ""
        self._collapsed_by_session[key] = not self._collapsed_by_session.get(key, False)
        self._repaint()

    def on_click(self, event: events.Click) -> None:
        if event.widget is not None and event.widget.id == "console-task-panel-header":
            event.stop()
            self.toggle_collapsed()

    def _repaint(self) -> None:
        if not self.is_mounted:
            return
        header, rows = render_task_lines(self._tasks, collapsed=self.collapsed)
        self.query_one("#console-task-panel-header", Static).update(
            _escape_markup(header)
        )
        body = self.query_one("#console-task-panel-body", VerticalScroll)
        body.display = not self.collapsed
        body.remove_children()
        body.mount_all(
            Static(
                _escape_markup(text),
                classes=f"console-task-panel-row {_STATUS_CLASS.get(status, '')}".strip(),
            )
            for status, text in rows
        )

    def on_mount(self) -> None:
        if self._tasks:
            self._repaint()
