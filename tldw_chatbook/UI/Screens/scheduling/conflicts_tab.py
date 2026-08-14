"""Conflicts tab for the Schedules workbench."""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, DataTable, Static

from ....Widgets.confirmation_dialog import ConfirmationDialog

logger = logger.bind(module="ConflictsTab")


class _SyncEngineProtocol(Protocol):
    """Minimal interface required of a sync conflict resolver."""

    def resolve_conflict(self, conflict_id: str, resolution: str) -> bool:
        """Resolve a conflict and report success."""
        ...


def _conflict_type_label(conflict: dict[str, Any]) -> str:
    """Return a plain-language label for a sync conflict type."""
    server_state = conflict.get("server_state") or {}
    return "Changed on server" if server_state else "Deleted on server"


class ConflictsTab(Vertical):
    """DataTable of unresolved sync conflicts with per-row actions."""

    BUNDLED_CSS = """
    ConflictsTab {
        height: 1fr;
    }
    #scheduling-conflicts-table {
        height: 1fr;
    }
    #scheduling-conflicts-empty {
        color: $text-muted;
        padding: 2 1;
        display: none;
    }
    #scheduling-conflict-actions {
        height: auto;
    }
    #scheduling-conflict-detail {
        height: auto;
        max-height: 9;
        padding: 0 1;
        color: $text;
        border-top: solid $surface-lighten-2;
    }
    .conflict-detail-muted {
        color: $text-muted;
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
        self._conflicts_by_id: dict[str, dict[str, Any]] = {}

    def compose(self) -> ComposeResult:
        """Build the tab layout."""
        yield Static("Unresolved conflicts")
        table = DataTable(id="scheduling-conflicts-table")
        table.add_columns("Title", "Conflict", "Server updated", "Local updated")
        yield table
        yield Static(
            "Select a conflict to compare both versions.",
            id="scheduling-conflict-detail",
            classes="conflict-detail-muted",
        )
        yield Static(
            "No unresolved conflicts.",
            id="scheduling-conflicts-empty",
        )
        with Horizontal(id="scheduling-conflict-actions"):
            yield Button(
                "Use server",
                id="scheduling-use-server",
                tooltip="Resolve the selected conflict with the server version.",
                disabled=True,
            )
            yield Button(
                "Use local",
                id="scheduling-use-local",
                tooltip="Resolve the selected conflict with the local version.",
                disabled=True,
            )

    def populate(self, conflicts: list[dict[str, Any]]) -> None:
        """Populate the table with unresolved conflicts.

        Args:
            conflicts: List of conflict dictionaries.
        """
        table = self.query_one("#scheduling-conflicts-table", DataTable)
        table.clear()
        self._conflicts_by_id = {}
        for conflict in conflicts:
            server_state = conflict.get("server_state") or {}
            local_state = conflict.get("local_state") or {}
            local_row = local_state.get("record") or local_state or {}
            server_updated = server_state.get("updated_at", "—")
            local_updated = local_row.get("updated_at", "—")
            self._conflicts_by_id[conflict["id"]] = conflict
            table.add_row(
                local_row.get("title", "Untitled"),
                _conflict_type_label(conflict),
                server_updated,
                local_updated,
                key=conflict["id"],
            )
        has_rows = bool(conflicts)
        table.display = has_rows
        detail = self.query_one("#scheduling-conflict-detail", Static)
        detail.display = has_rows
        if has_rows:
            detail.update("Select a conflict to compare both versions.")
        empty_state = self.query_one("#scheduling-conflicts-empty", Static)
        empty_state.display = "none" if has_rows else "block"
        self._set_actions_enabled(has_rows)

    def _set_actions_enabled(self, enabled: bool) -> None:
        """Show the resolution row only when there is something to resolve."""
        actions = self.query_one("#scheduling-conflict-actions", Horizontal)
        actions.display = enabled
        for button_id in ("#scheduling-use-server", "#scheduling-use-local"):
            button = self.query_one(button_id, Button)
            button.disabled = not enabled
            button.tooltip = (
                "Select a conflict above, then choose which version to keep."
                if enabled
                else "No conflicts to resolve."
            )

    def on_mount(self) -> None:
        """Configure the table cursor."""
        table = self.query_one("#scheduling-conflicts-table", DataTable)
        table.cursor_type = "row"
        self._set_actions_enabled(False)

    @on(DataTable.RowHighlighted, "#scheduling-conflicts-table")
    def _on_conflict_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Show both versions of the highlighted conflict for comparison."""
        conflict_id, conflict = self._selected_conflict()
        if conflict_id is None or conflict is None:
            return
        self._show_detail(conflict)

    @staticmethod
    def _version_summary(state: dict[str, Any], *, missing_text: str) -> str:
        """One-line summary of one side's version of a conflicted record."""
        if not state:
            return missing_text
        record = state.get("record") or state
        title = record.get("title") or "Untitled"
        updated = state.get("updated_at") or record.get("updated_at") or "—"
        details: list[str] = []
        schedule = (
            record.get("schedule_kind")
            or record.get("cron")
            or record.get("run_at")
        )
        if schedule:
            details.append(str(schedule))
        body = (record.get("body") or "").strip().replace("\n", " ")
        if body:
            details.append(body[:40] + ("…" if len(body) > 40 else ""))
        suffix = f" · {' · '.join(details)}" if details else ""
        return f"'{title}' · {updated}{suffix}"

    def _show_detail(self, conflict: dict[str, Any]) -> None:
        """Render both versions so the keep-server/local choice is informed."""
        server_state = conflict.get("server_state") or {}
        local_state = conflict.get("local_state") or {}
        detail = self.query_one("#scheduling-conflict-detail", Static)
        detail.update(
            "Server: "
            + self._version_summary(server_state, missing_text="(deleted on server)")
            + "\nLocal:  "
            + self._version_summary(local_state, missing_text="(no local copy)")
        )

    @on(Button.Pressed, "#scheduling-use-server")
    def _on_use_server(self) -> None:
        """Ask for confirmation, then resolve using the server version."""
        self._confirm_resolution("server")

    @on(Button.Pressed, "#scheduling-use-local")
    def _on_use_local(self) -> None:
        """Ask for confirmation, then resolve using the local version."""
        self._confirm_resolution("local")

    def _selected_conflict(self) -> tuple[str, dict[str, Any]] | tuple[None, None]:
        """Return the (id, conflict) pair at the current cursor row."""
        table = self.query_one("#scheduling-conflicts-table", DataTable)
        if table.cursor_row is None or not table.ordered_rows:
            return None, None
        row = table.ordered_rows[table.cursor_row]
        conflict_id = row.key.value
        return conflict_id, self._conflicts_by_id.get(conflict_id)

    def _confirm_resolution(self, resolution: str) -> None:
        """Confirm the destructive resolution before applying it."""
        conflict_id, conflict = self._selected_conflict()
        if conflict_id is None or conflict is None:
            self.app.notify("Select a conflict first.", severity="warning")
            return
        title = (conflict.get("local_state") or {}).get("record", {}).get(
            "title"
        ) or (conflict.get("local_state") or {}).get("title") or "Untitled"
        if resolution == "server":
            message = (
                f"Keep the server version of '{title}'?\n\n"
                "Your local changes will be overwritten."
            )
            confirm_label = "Keep server version"
        else:
            message = (
                f"Keep the local version of '{title}'?\n\n"
                "The server version will be overwritten."
            )
            confirm_label = "Keep local version"

        async def _on_confirm() -> None:
            self._resolve(conflict_id, resolution)

        self.app.push_screen(
            ConfirmationDialog(
                title="Resolve sync conflict",
                message=message,
                confirm_label=confirm_label,
                cancel_label="Cancel",
                confirm_callback=_on_confirm,
            )
        )

    def _resolve(self, conflict_id: str, resolution: str) -> None:
        """Resolve a conflict by id after confirmation.

        Args:
            conflict_id: Identifier of the conflict to resolve.
            resolution: Either ``"server"`` or ``"local"``.
        """
        if self.sync_engine is None:
            return
        try:
            result = self.sync_engine.resolve_conflict(conflict_id, resolution)
        except Exception:
            logger.exception("Failed to resolve conflict %s", conflict_id)
            return
        if not result:
            return
        table = self.query_one("#scheduling-conflicts-table", DataTable)
        table.remove_row(conflict_id)
        self._conflicts_by_id.pop(conflict_id, None)
        if not self._conflicts_by_id:
            self.query_one("#scheduling-conflicts-empty", Static).display = "block"
            self.query_one("#scheduling-conflicts-table", DataTable).display = False
            self.query_one("#scheduling-conflict-detail", Static).display = False
            self._set_actions_enabled(False)
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
