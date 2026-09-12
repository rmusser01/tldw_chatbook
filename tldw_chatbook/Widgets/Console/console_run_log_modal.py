"""Read-only Console run-log viewer retaining one bounded page."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from functools import partial
from typing import ClassVar

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea

from tldw_chatbook.Agents.run_log_paging import (
    RunLogPage,
    RunLogPageCursor,
    format_record_page,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

MODAL_ID = "console-run-log-modal"
TEXT_AREA_ID = "console-run-log-text"
CLOSE_BUTTON_ID = "console-run-log-close"
RunLogPageLoader = Callable[[RunLogPageCursor | None], RunLogPage | None]
MAX_HISTORY = 256


class ConsoleRunLogModal(SafeModalDismissMixin, ModalScreen[None]):
    """View stored content a page at a time; Previous reloads a cursor."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("escape", "request_safe_cancel", "Close")
    ]
    SAFE_MODAL_CONTENT = "#console-run-log-modal"

    def __init__(
        self,
        *,
        run_id: str,
        first_page: RunLogPage,
        page_loader: RunLogPageLoader,
        target_is_current: Callable[[], bool] = lambda: True,
    ) -> None:
        super().__init__()
        self._run_id = run_id
        self._page = first_page
        self._page_loader = page_loader
        self._target_is_current = target_is_current
        self._history: deque[RunLogPageCursor] = deque(maxlen=MAX_HISTORY)
        self._page_number = 1
        self._loading = False
        self._generation = 0
        self._error = False

    def compose(self) -> ComposeResult:
        with Vertical(id=MODAL_ID):
            yield Static(
                f"Full run log — {self._run_id}",
                classes="console-modal-header",
                markup=False,
            )
            yield TextArea(
                format_record_page(self._page),
                id=TEXT_AREA_ID,
                read_only=True,
                soft_wrap=True,
            )
            yield Static("", id="console-run-log-status", markup=False)
            with Horizontal(id="console-run-log-actions"):
                yield Button("First", id="console-run-log-first")
                yield Button("Previous", id="console-run-log-previous")
                yield Button("Next", id="console-run-log-next")
                yield Button("Close", id=CLOSE_BUTTON_ID, variant="primary")

    def on_mount(self) -> None:
        self.query_one(TextArea).focus()
        self._refresh_controls()

    def on_unmount(self) -> None:
        self._generation += 1

    def _refresh_controls(self) -> None:
        self.query_one("#console-run-log-first", Button).disabled = (
            self._loading or self._page_number == 1
        )
        self.query_one("#console-run-log-previous", Button).disabled = (
            self._loading or not self._history
        )
        self.query_one("#console-run-log-next", Button).disabled = (
            self._loading or self._page.next_cursor is None
        )
        if self._loading:
            status = "Loading page…"
        elif self._error:
            status = "Log no longer available. Last page kept; close to return."
        elif not self._page.slices and self._page.next_cursor is not None:
            status = "No matching records in this scan. Continue with Next."
        else:
            status = f"Page {self._page_number}"
            status += " · More available" if self._page.next_cursor else " · End of log"
        if self._page.diagnostics and not self._loading and not self._error:
            status += " · Some records could not be read"
        self.query_one("#console-run-log-status", Static).update(status)

    @on(Button.Pressed)
    async def _button(self, event: Button.Pressed) -> None:
        action = event.button.id
        if action == CLOSE_BUTTON_ID:
            event.stop()
            await self.request_safe_cancel(source="visible")
            return
        if action not in {
            "console-run-log-first",
            "console-run-log-previous",
            "console-run-log-next",
        }:
            return
        event.stop()
        if self._loading or event.button.disabled or not self._target_is_current():
            return
        direction = action.rsplit("-", 1)[-1]
        cursor = (
            None
            if direction == "first"
            else self._history[-1]
            if direction == "previous"
            else self._page.next_cursor
        )
        self._loading = True
        self._generation += 1
        self._refresh_controls()
        self.run_worker(
            partial(self._load_page, cursor, direction, self._generation),
            thread=True,
            exclusive=True,
            group="run-log-page",
        )

    def _load_page(
        self, cursor: RunLogPageCursor | None, direction: str, generation: int
    ) -> None:
        try:
            page = self._page_loader(cursor)
        except Exception as error:  # noqa: BLE001 - loader failures preserve the last page.
            logger.error("Run-log page unavailable ({})", type(error).__name__)
            page = None
        try:
            self.app.call_from_thread(self._publish_page, page, direction, generation)
        except RuntimeError:
            # The app can have exited while the filesystem read was in flight.
            return

    def _publish_page(
        self, page: RunLogPage | None, direction: str, generation: int
    ) -> None:
        if (
            not self.is_mounted
            or self not in self.app.screen_stack
            or generation != self._generation
            or not self._target_is_current()
        ):
            return
        self._loading = False
        self._error = page is None
        if page is not None:
            if direction == "next":
                self._history.append(self._page.start_cursor)
                self._page_number += 1
            elif direction == "previous":
                self._history.pop()
                self._page_number -= 1
            else:
                self._history.clear()
                self._page_number = 1
            self._page = page
            self.query_one(TextArea).load_text(format_record_page(page))
        self._refresh_controls()

    async def action_dismiss_viewer(self) -> None:
        """Dismiss through the shared safe cancellation boundary."""
        await self.request_safe_cancel(source="visible")
