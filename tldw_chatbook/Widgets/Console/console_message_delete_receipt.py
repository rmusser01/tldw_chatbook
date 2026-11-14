"""Receipt for a confirmed Console message Delete, offering Undo.

TASK-33628.2: follows the Archive receipt pattern
(``WorkspaceArchiveReceiptModal``) -- a persistent receipt dismissed only by
deliberate interaction, with Undo as the focused default -- so a delete that
removed a whole run of later turns states its real count and can be taken
back exactly.

TASK-33628.5: a large delete, and its Undo, save off the event loop. Given
``delete``, the receipt opens in a working state ("Deleting N messages...")
and becomes the receipt once that coroutine has saved the delete; given
``undo``, Undo shows "Restoring N messages..." while it runs. A save cannot be
cancelled once it starts, so while one runs Escape, a backdrop click and
Done are refused and Ctrl+Q says it is still working
(``refuse_quit_while_working``).

Its styling is NOT on the boot path (ADR-097): the ``console-delete-receipt*``
rules live in ``css/features/_console.tcss`` and ``build_css.py`` splits them
into ``screen_modal_console_delete_receipt.tcss``, which this modal's
``CSS_PATH`` loads the first time a receipt opens.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import ClassVar

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

#: What Ctrl+Q says while a save runs, by what is being saved.
_STILL_SAVING = {
    "delete": "The delete is still being saved.",
    "undo": "Undo is still restoring the messages.",
}


class ConsoleMessageDeleteReceiptModal(SafeModalDismissMixin, ModalScreen[str | None]):
    """Counted delete receipt.

    Dismisses with ``"undo"`` (restored), ``None`` (Done: the delete is
    final) or ``"failed"`` (``delete`` raised; nothing was deleted).
    """

    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "css"
        / "screen_modal_console_delete_receipt.tcss"
    )
    SAFE_MODAL_CONTENT = "#console-delete-receipt-box"
    BINDINGS: ClassVar = [("escape", "request_safe_cancel", "Done")]
    AUTO_FOCUS = "#console-delete-receipt-undo"

    @classmethod
    def preload_sheet(cls, app: object) -> None:
        """Parse the lazy sheet before the first push, restyling no mounted node.

        ``push_screen`` would read it through ``App._load_screen_css``, which
        then restyles every node of every screen: 5.3 s live on the first
        Delete with a scrolled-back 420-row transcript, 9.3 s with 3,000 rows
        (TASK-33628.5.1). Every rule build_css.py splits into this sheet names
        only receipt tokens, so no node outside a receipt can match one (pinned
        by ``Tests/UI/test_console_long_chat_bounds.py``), and a receipt's own
        nodes take the rules as they mount. The push then finds the sheet read.

        Args:
            app: The running app whose stylesheet gets the sheet. Anything
                else, or a sheet that cannot be read here, leaves the load to
                ``push_screen`` exactly as before.
        """
        stylesheet = getattr(app, "stylesheet", None)
        if stylesheet is None or stylesheet.has_source(cls.CSS_PATH, ""):
            return
        try:
            stylesheet.read(cls.CSS_PATH)
            stylesheet.reparse()
        except Exception:  # noqa: BLE001 - push_screen reports it as it always did
            # A failed reparse keeps the old rules; drop the source so the
            # push reads the sheet, and fails, exactly as it did before.
            stylesheet.source.pop((cls.CSS_PATH, ""), None)

    def __init__(
        self,
        *,
        count: int,
        delete: Callable[[], Awaitable[int | None]] | None = None,
        undo: Callable[[], Awaitable[str]] | None = None,
    ) -> None:
        """Build the receipt.

        Args:
            count: How many messages the delete removes.
            delete: Saves the delete and returns how many messages it
                removed (the receipt then counts those); raises when it did
                not happen (having said why). ``None``: already saved.
            undo: Restores the messages and returns ``"restored"``,
                ``"retry"`` (nothing changed; offer Undo again) or
                ``"final"`` (Undo is impossible; the delete stands). ``None``:
                Undo just dismisses with ``"undo"``.
        """
        super().__init__()
        # The lazy sheet centers the receipt through this class: a bare type
        # selector carries no owner token, so the split would keep it on the
        # boot bundle, and a screen id could collide with a retried receipt
        # whose predecessor is still being removed.
        self.add_class("console-delete-receipt-modal")
        self._count = count
        self._delete = delete
        self._undo = undo
        #: ``"delete"``/``"undo"`` while that save runs, else ``None``.
        self._working: str | None = "delete" if delete is not None else None

    def compose(self) -> ComposeResult:
        with Vertical(id="console-delete-receipt-box"):
            if self._working is not None:
                yield self._progress(self._working)
            else:
                yield self._receipt()

    def on_mount(self) -> None:
        if self._delete is not None:
            self.run_worker(
                self._run_delete(), group="console-delete-receipt", exit_on_error=False
            )

    def _noun(self) -> str:
        return "message" if self._count == 1 else "messages"

    def _receipt(self) -> Vertical:
        later = self._count - 1
        scope = (
            "The selected message was removed from this conversation."
            if later <= 0
            else (
                f"The selected message and {later} later "
                f"{'message' if later == 1 else 'messages'} under it were "
                "removed from this conversation."
            )
        )
        restore = (
            "Undo puts it back exactly where it was"
            if later <= 0
            else "Undo puts them back exactly where they were"
        )
        return Vertical(
            Static(
                f"Deleted {self._count} {self._noun()}",
                classes="console-modal-header",
                markup=False,
            ),
            Static(f"{scope} {restore}; Done keeps the delete.", markup=False),
            Horizontal(
                Button("Undo", id="console-delete-receipt-undo", compact=True),
                Button("Done", id="console-delete-receipt-done", compact=True),
                id="console-delete-receipt-actions",
            ),
            id="console-delete-receipt",
        )

    def _progress(self, mode: str) -> Vertical:
        verb = "Restoring" if mode == "undo" else "Deleting"
        detail = (
            "Putting the messages back. This screen closes when they are."
            if mode == "undo"
            else "Saving the delete. Undo is offered as soon as it is saved."
        )
        return Vertical(
            Static(
                f"{verb} {self._count} {self._noun()}…",
                classes="console-modal-header",
                markup=False,
            ),
            Static(detail, id="console-delete-receipt-status", markup=False),
            id="console-delete-receipt-progress",
        )

    async def _show(self, panel: Vertical) -> bool:
        # A save can finish after the receipt has gone (the app quitting).
        boxes = self.query("#console-delete-receipt-box")
        if not boxes:
            return False
        box = boxes.first(Vertical)
        await box.remove_children()
        await box.mount(panel)
        return True

    async def _show_receipt(self) -> None:
        self._working = None
        if not await self._show(self._receipt()):
            return
        undo = self.query("#console-delete-receipt-undo")
        if undo:
            undo.first(Button).focus()

    async def _run_delete(self) -> None:
        assert self._delete is not None
        try:
            saved = await self._delete()
        except Exception:  # noqa: BLE001 - the flow already told the user why
            self._working = None
            self.dismiss_safe_once_when_on_top("failed")
            return
        if isinstance(saved, int):
            self._count = saved
        await self._show_receipt()

    async def _run_undo(self) -> None:
        assert self._undo is not None
        try:
            outcome = await self._undo()
        except Exception as exc:  # noqa: BLE001 - nothing more to tell; offer again
            logger.warning("Console delete Undo raised: {}", type(exc).__name__)
            outcome = "retry"
        if outcome == "retry":
            await self._show_receipt()
            return
        self._working = None
        self.dismiss_safe_once_when_on_top("undo" if outcome == "restored" else None)

    async def _perform_safe_cancel(self, *, source: str) -> None:
        """Done, unless a save is running: then say it can't be stopped."""
        if self._working is not None:
            # The progress panel can still be mounting (Undo just pressed).
            for status in self.query("#console-delete-receipt-status").results(Static):
                status.update(f"{_STILL_SAVING[self._working]}\nIt can't be cancelled.")
            return
        self.dismiss_safe_once(None)

    async def confirm_quit(self) -> bool:
        """Stay while the delete or its Undo is being saved (TASK-33628.5).

        Returns:
            refuse_quit_while_working's answer while saving; else True.
        """
        if not self._working:
            return True
        from tldw_chatbook.Widgets.quit_while_working import (
            refuse_quit_while_working,
        )

        return await refuse_quit_while_working(
            self, _STILL_SAVING.get(self._working, _STILL_SAVING["delete"])
        )

    @on(Button.Pressed)
    async def _choose(self, event: Button.Pressed) -> None:
        event.stop()
        if self._working is not None:
            return
        if event.button.id != "console-delete-receipt-undo":
            await self.request_safe_cancel(source="button")
        elif self._undo is None:
            self.dismiss("undo")
        else:
            self._working = "undo"
            await self._show(self._progress("undo"))
            self.run_worker(
                self._run_undo(), group="console-delete-receipt", exit_on_error=False
            )
