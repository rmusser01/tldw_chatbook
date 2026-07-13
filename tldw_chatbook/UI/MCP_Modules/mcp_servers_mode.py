"""Servers-mode canvas: readiness overview table and per-server detail."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, DataTable, Static

from tldw_chatbook.MCP.readiness import (
    ReadinessSnapshot,
    ReadinessState,
    aggregate_summary,
)
from tldw_chatbook.MCP.redaction import redact_args

_TABLE_COLUMNS = ("Name", "Transport", "Status", "Tools", "Auth", "Scope")


class MCPServersMode(Vertical):
    """Canvas for the Servers mode. Read-only in Phase 1."""

    DEFAULT_CSS = """
    MCPServersMode {
        width: 1fr;
        height: 100%;
        min-height: 0;
    }
    #mcp-servers-table {
        height: 1fr;
        min-height: 4;
    }
    #mcp-detail-scroll {
        height: 1fr;
        min-height: 0;
    }
    """

    class ServerRowSelected(Message, namespace="mcp_servers_mode"):
        def __init__(self, server_key: str) -> None:
            super().__init__()
            self.server_key = server_key

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshots: list[ReadinessSnapshot] = []
        self._detail_snapshot: ReadinessSnapshot | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="mcp-servers-overview"):
            yield Static("", id="mcp-overview-summary", classes="ds-status-badge", markup=False)
            table = DataTable(id="mcp-servers-table")
            table.cursor_type = "row"
            yield table
            yield Vertical(id="mcp-overview-callouts")
        with Vertical(id="mcp-servers-detail"):
            yield Static("", id="mcp-detail-title", classes="destination-section", markup=False)
            with VerticalScroll(id="mcp-detail-scroll"):
                yield Static("", id="mcp-detail-body", classes="ds-field-row", markup=False)
                yield Button(
                    "Copy client config",
                    id="mcp-detail-copy-snippet",
                    classes="console-action-secondary",
                    compact=True,
                    tooltip="Copy this built-in server's client config snippet to the clipboard.",
                )

    def on_mount(self) -> None:
        table = self.query_one("#mcp-servers-table", DataTable)
        table.add_columns(*_TABLE_COLUMNS)
        self._show_overview_container(True)

    def _show_overview_container(self, show_overview: bool) -> None:
        self.query_one("#mcp-servers-overview").display = show_overview
        self.query_one("#mcp-servers-detail").display = not show_overview

    def update_overview(self, snapshots: list[ReadinessSnapshot]) -> None:
        self._snapshots = list(snapshots)
        summary = self.query_one("#mcp-overview-summary", Static)
        summary.update(aggregate_summary(self._snapshots))
        table = self.query_one("#mcp-servers-table", DataTable)
        table.clear()
        for snap in self._snapshots:
            table.add_row(
                snap.label,
                snap.transport,
                snap.badge_text(),
                "—" if snap.tool_count is None else str(snap.tool_count),
                snap.auth_display,
                snap.scope_display,
                key=snap.server_key,
            )
        callouts = self.query_one("#mcp-overview-callouts", Vertical)
        callouts.remove_children()
        for snap in self._snapshots:
            if snap.state in (ReadinessState.READY, ReadinessState.CHECKING):
                continue
            callouts.mount(
                Static(
                    f"{snap.label}: {snap.message}",
                    classes="ds-recovery-callout",
                    markup=False,
                )
            )
        if self._detail_snapshot is None:
            self._show_overview_container(True)

    def show_detail(self, snapshot: ReadinessSnapshot | None) -> None:
        self._detail_snapshot = snapshot
        if snapshot is None:
            self._show_overview_container(True)
            return
        self._show_overview_container(False)
        self.query_one("#mcp-detail-title", Static).update(
            f"{snapshot.badge_text()}  {snapshot.label}"
        )
        self.query_one("#mcp-detail-body", Static).update(self._detail_text(snapshot))
        self.query_one("#mcp-detail-copy-snippet", Button).display = (
            snapshot.source == "builtin"
        )

    def _detail_text(self, snapshot: ReadinessSnapshot) -> str:
        detail = snapshot.detail or {}
        lines: list[str] = [snapshot.message, ""]
        if snapshot.source == "local":
            args = redact_args([str(a) for a in detail.get("args") or []])
            lines.append(f"Command · {detail.get('command') or '—'} {' '.join(args)}".rstrip())
            placeholders = detail.get("env_placeholders") or {}
            missing = set(detail.get("missing_env") or [])
            for env_key, raw in placeholders.items():
                marker = "missing" if str(raw).strip("${}") in missing else "set"
                lines.append(f"Env · {env_key} ({marker})")
            discovery = detail.get("discovery_snapshot") or {}
            for kind in ("tools", "resources", "prompts"):
                items = discovery.get(kind) or []
                names = ", ".join(str(item.get("name") or item.get("uri") or "?") for item in items[:8])
                suffix = f": {names}" if names else ""
                lines.append(f"{kind.title()} · {len(items)}{suffix}")
        elif snapshot.source == "server":
            lines.append(f"Base URL · {detail.get('base_url') or '—'}")
            lines.append(f"Auth · {snapshot.auth_display}")
            lines.append("External server records: see Advanced ▸ External Servers.")
        else:  # builtin
            lines.append("Runs over stdio when an MCP client launches it:")
            lines.append("  python3 -m tldw_chatbook.MCP")
            for flag in ("expose_tools", "expose_resources", "expose_prompts"):
                lines.append(f"{flag} · {detail.get(flag, True)}")
        return "\n".join(lines)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        if event.row_key is not None and event.row_key.value is not None:
            self.post_message(self.ServerRowSelected(str(event.row_key.value)))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "mcp-detail-copy-snippet":
            return
        event.stop()
        snippet = ""
        if self._detail_snapshot is not None:
            snippet = str((self._detail_snapshot.detail or {}).get("client_snippet") or "")
        if snippet:
            self.app.copy_to_clipboard(snippet)
            self.app.notify("Client config copied to clipboard.")
