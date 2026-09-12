"""Read-only, paged sub-agent selection for the existing Console drill-in."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import ClassVar

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Static

from ...Chat.cost_display import format_token_count
from ..modal_dismissal import SafeModalDismissMixin

PAGE_SIZE = 50


class ConsoleAgentHistoryModal(SafeModalDismissMixin, ModalScreen[str | None]):
    """Pick one saved child without loading its steps or changing run state."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("escape", "request_safe_cancel", "Close")
    ]
    SAFE_MODAL_CONTENT = "#agent-history-dialog"

    def __init__(self, *, load_page: Callable[..., list[dict]]) -> None:
        super().__init__()
        self._load_page = load_page
        self._cursors: list[tuple[str, str] | None] = [None]
        self._next_cursor: tuple[str, str] | None = None
        self._rows: dict[str, dict] = {}
        self._generation = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="agent-history-dialog"):
            yield Static("Sub-agent run history", classes="console-modal-header")
            yield Static("Loading runs…", id="agent-history-state", markup=False)
            yield DataTable(
                id="agent-history-table", cursor_type="row", zebra_stripes=True
            )
            with Horizontal(id="agent-history-actions"):
                yield Button("Previous", id="agent-history-previous", disabled=True)
                yield Button("Next", id="agent-history-next", disabled=True)
                yield Button("Refresh", id="agent-history-refresh")
                yield Button("Close", id="agent-history-close")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        for name, width in (
            ("Status", 12),
            ("Task", 48),
            ("Budget tok", 11),
            ("Started", 19),
        ):
            table.add_column(name, width=width)
        self._request_page()

    def _request_page(self) -> None:
        self._generation += 1
        self._rows = {}
        table = self.query_one(DataTable)
        table.clear()
        table.disabled = True
        self.query_one("#agent-history-state", Static).update("Loading runs…")
        self.query_one("#agent-history-previous", Button).disabled = True
        self.query_one("#agent-history-next", Button).disabled = True
        self.run_worker(
            self._read_page(self._generation, self._cursors[-1]),
            exclusive=True,
            group="agent-history-page",
        )

    async def _read_page(self, generation: int, cursor: tuple[str, str] | None) -> None:
        try:
            records = await asyncio.to_thread(
                self._load_page, before=cursor, limit=PAGE_SIZE + 1
            )
        except Exception:  # noqa: BLE001 -- loader boundary must leave recovery available
            if self.is_mounted and generation == self._generation:
                self.query_one("#agent-history-state", Static).update(
                    "Could not load runs. Refresh to retry."
                )
            return
        if not self.is_mounted or generation != self._generation:
            return
        page = records[:PAGE_SIZE]
        self._next_cursor = (
            (page[-1]["created_at"], page[-1]["id"])
            if len(records) > PAGE_SIZE
            else None
        )
        table = self.query_one(DataTable)
        for record in page:
            run_id = record["id"]
            self._rows[run_id] = record
            budget = record.get("budget_tokens")
            table.add_row(
                Text(str(record["status"])),
                Text(str(record.get("task") or "sub-agent").replace("\n", " ")[:200]),
                Text(
                    format_token_count(budget) if budget is not None else "Unavailable"
                ),
                Text(str(record["created_at"])[:19].replace("T", " ")),
                key=run_id,
            )
        table.disabled = False
        self.query_one("#agent-history-state", Static).update(
            f"Page {len(self._cursors)} · {len(page)} shown · newest first · Enter to inspect"
            if page
            else "No saved sub-agent runs. Refresh to check again."
        )
        self.query_one("#agent-history-previous", Button).disabled = (
            len(self._cursors) == 1
        )
        self.query_one("#agent-history-next", Button).disabled = (
            self._next_cursor is None
        )
        table.focus()

    @on(Button.Pressed)
    async def _navigate(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "agent-history-close":
            event.stop()
            await self.request_safe_cancel(source="visible")
            return
        if button_id == "agent-history-next" and self._next_cursor is not None:
            self._cursors.append(self._next_cursor)
        elif button_id == "agent-history-previous" and len(self._cursors) > 1:
            self._cursors.pop()
        elif button_id == "agent-history-refresh":
            self._cursors = [None]
        else:
            return
        event.stop()
        self._request_page()

    @on(DataTable.RowSelected, "#agent-history-table")
    def _select(self, event: DataTable.RowSelected) -> None:
        event.stop()
        run_id = str(event.row_key.value)
        if run_id in self._rows:
            self.dismiss(run_id)
