"""Receipt for a confirmed Console message Delete, offering Undo.

TASK-33628.2: follows the Archive receipt pattern
(``WorkspaceArchiveReceiptModal``) -- a persistent receipt dismissed only by
deliberate interaction, with Undo as the focused default -- so a delete that
removed a whole run of later turns states its real count and can be taken
back exactly.

Its styling is NOT on the boot path (ADR-097): the ``console-delete-receipt*``
rules live in ``css/features/_console.tcss`` and ``build_css.py`` splits them
into ``screen_modal_console_delete_receipt.tcss``, which this modal's
``CSS_PATH`` loads the first time a receipt opens.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin


class ConsoleMessageDeleteReceiptModal(SafeModalDismissMixin, ModalScreen[str | None]):
    """Counted delete receipt; dismisses with ``"undo"`` or ``None`` (Done)."""

    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "css"
        / "screen_modal_console_delete_receipt.tcss"
    )
    SAFE_MODAL_CONTENT = "#console-delete-receipt"
    BINDINGS: ClassVar = [("escape", "request_safe_cancel", "Done")]
    AUTO_FOCUS = "#console-delete-receipt-undo"

    def __init__(self, *, count: int) -> None:
        super().__init__()
        # The lazy sheet centers the receipt through this class: a bare type
        # selector carries no owner token, so the split would keep it on the
        # boot bundle, and a screen id could collide with a retried receipt
        # whose predecessor is still being removed.
        self.add_class("console-delete-receipt-modal")
        self._count = count

    def compose(self) -> ComposeResult:
        noun = "message" if self._count == 1 else "messages"
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
        with Vertical(id="console-delete-receipt"):
            yield Static(
                f"Deleted {self._count} {noun}",
                classes="console-modal-header",
                markup=False,
            )
            restore = (
                "Undo puts it back exactly where it was"
                if later <= 0
                else "Undo puts them back exactly where they were"
            )
            yield Static(f"{scope} {restore}; Done keeps the delete.", markup=False)
            with Horizontal(id="console-delete-receipt-actions"):
                yield Button("Undo", id="console-delete-receipt-undo", compact=True)
                yield Button("Done", id="console-delete-receipt-done", compact=True)

    @on(Button.Pressed)
    async def _choose(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "console-delete-receipt-undo":
            self.dismiss("undo")
        else:
            await self.request_safe_cancel(source="button")
