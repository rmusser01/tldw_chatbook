"""Conflicts tab for the Schedules workbench."""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, DataTable, Static

logger = logger.bind(module="ConflictsTab")


class _SyncEngineProtocol(Protocol):
    """Minimal interface required of a sync conflict resolver."""

    def resolve_conflict(self, conflict_id: str, resolution: str) -> bool:
        """Resolve a conflict and report success."""
        ...


class ConflictsTab(Vertical):
    """DataTable of unresolved sync conflicts with per-row actions."""

    DEFAULT_CSS = """
    ConflictsTab {
        height: 1fr;
    }
    #scheduling-conflicts-table {
        height: 1fr;
    }
    """

    def __init__(self, sync_engine: _SyncEngineProtocol | None, **kwargs) -> None:
        """Initialize the conflicts tab.

        Args:
            sync_engine: Engine providing ``resolve_conflict(conflict_id, resolution)``.
            **kwargs: Passed to the parent widget.
        """
        super().__init__(**kwargs)
        self.sync_engine = sync_engine

    def compose(self) -> ComposeResult:
        """Build the tab layout."""
        yield Static("Unresolved conflicts")
        table = DataTable(id="scheduling-conflicts-table")
        table.add_columns("Title", "Conflict Type", "Server updated", "Local updated")
        yield table
        with Horizontal(id="scheduling-conflict-actions"):
            yield Button("Use server", id="scheduling-use-server")
            yield Button("Use local", id="scheduling-use-local")

    def populate(self, conflicts: list[dict[str, Any]]) -> None:
        """Populate the table with unresolved conflicts.

        Args:
            conflicts: List of conflict dictionaries.
        """
        table = self.query_one("#scheduling-conflicts-table", DataTable)
        table.clear()
        for conflict in conflicts:
            server_state = conflict.get("server_state") or {}
            local_state = conflict.get("local_state") or {}
            local_row = local_state.get("record") or local_state or {}
            conflict_type = "server-deletion" if not server_state else "server-update"
            server_updated = server_state.get("updated_at", "—")
            local_updated = local_row.get("updated_at", "—")
            table.add_row(
                local_row.get("title", "Untitled"),
                conflict_type,
                server_updated,
                local_updated,
                key=conflict["id"],
            )

    def on_mount(self) -> None:
        """Configure the table cursor."""
        table = self.query_one("#scheduling-conflicts-table", DataTable)
        table.cursor_type = "row"

    @on(Button.Pressed, "#scheduling-use-server")
    def _on_use_server(self) -> None:
        """Resolve the selected conflict using the server version."""
        self._resolve_selected("server")

    @on(Button.Pressed, "#scheduling-use-local")
    def _on_use_local(self) -> None:
        """Resolve the selected conflict using the local version."""
        self._resolve_selected("local")

    def _resolve_selected(self, resolution: str) -> None:
        """Resolve the conflict at the current cursor row.

        Args:
            resolution: Either ``"server"`` or ``"local"``.
        """
        table = self.query_one("#scheduling-conflicts-table", DataTable)
        if table.cursor_row is None:
            return
        row = table.ordered_rows[table.cursor_row]
        conflict_id = row.key.value
        if self.sync_engine is None:
            return
        try:
            result = self.sync_engine.resolve_conflict(conflict_id, resolution)
        except Exception:
            logger.exception("Failed to resolve conflict %s", conflict_id)
            return
        if not result:
            return
        table.remove_row(conflict_id)
        self.post_message(self.ConflictResolved(conflict_id, resolution))

    class ConflictResolved(Message):
        """Posted when the user resolves a conflict."""

        def __init__(self, conflict_id: str, resolution: str) -> None:
            """Initialize the message.

            Args:
                conflict_id: Identifier of the resolved conflict.
                resolution: Resolution chosen by the user.
            """
            super().__init__()
            self.conflict_id = conflict_id
            self.resolution = resolution
