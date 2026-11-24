"""Show Copy progress and durable per-item receipts."""

import asyncio

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Select, Static

from ...Research_Workspace.copy_models import ResearchCopyStatus
from ...Research_Workspace.copy_receipts import CopyItemStage, ResearchCopyReceipt
from ...Widgets.modal_dismissal import SafeModalDismissMixin


def copy_status(value: ResearchCopyStatus) -> str:
    return value.value.replace("_", " ").capitalize()


class ResearchCopyProgress(ModalScreen[None]):
    BINDINGS = [Binding("escape", "stop", "Stop", show=False)]

    def __init__(self, cancel: asyncio.Event) -> None:
        super().__init__()
        self.cancel = cancel

    def compose(self) -> ComposeResult:
        with Vertical(
            id="research-copy-progress-dialog", classes="research-transfer-dialog"
        ):
            yield Static("Copy in progress", classes="research-pane-title")
            yield Static("", id="research-copy-progress-status", markup=False)
            yield Button(
                "Stop after current item", id="research-copy-stop", compact=True
            )

    @on(Button.Pressed, "#research-copy-stop")
    def action_stop(self) -> None:
        self.cancel.set()
        self.query_one(Button).disabled = True
        self.query_one("#research-copy-progress-status", Static).update(
            "Stopping after the current owner checkpoint."
        )

    def update_receipt(self, receipt: ResearchCopyReceipt) -> None:
        if self.is_mounted and not self.cancel.is_set():
            verified = sum(row.stage is CopyItemStage.VERIFIED for row in receipt.items)
            selected = sum(row.item.actionable for row in receipt.items)
            self.query_one("#research-copy-progress-status", Static).update(
                f"{verified} of {selected} items verified"
            )


class ResearchCopyReceiptView(SafeModalDismissMixin, ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "request_safe_cancel", "Close", show=False)]
    SAFE_MODAL_CONTENT = "#research-copy-receipt-dialog"

    def __init__(self, receipt=None, *, open_receipts=()) -> None:
        super().__init__()
        self.receipt = receipt
        self.open_receipts = tuple(open_receipts)

    def compose(self) -> ComposeResult:
        receipt = self.receipt
        with Vertical(
            id="research-copy-receipt-dialog",
            classes="research-transfer-dialog research-transfer-scroll-dialog",
        ):
            yield Static("Copy receipts", classes="research-pane-title")
            with VerticalScroll(classes="research-transfer-scroll"):
                if receipt:
                    yield Static(
                        copy_status(receipt.status),
                        id="research-copy-receipt-status",
                        markup=False,
                    )
                    target = (
                        receipt.manifest.target_name
                        or receipt.manifest.destination.workspace_id
                    )
                    yield Static(f"Destination · {target}", markup=False)
                    for row in receipt.items:
                        state = (
                            row.failure.value.replace("_", " ")
                            if row.failure
                            else row.stage.value.replace("_", " ")
                        )
                        yield Static(
                            f"{row.item.source.kind.value} {row.item.source.owner_id} · {state}",
                            markup=False,
                        )
                else:
                    yield Select(
                        [
                            (
                                f"{copy_status(row.status)} · {row.manifest.target_name or row.manifest.destination.workspace_id}",
                                row.manifest.transfer_id,
                            )
                            for row in self.open_receipts
                        ],
                        prompt="Recoverable Copy",
                        id="research-copy-receipt-select",
                    )
            if (
                self.open_receipts
                or receipt
                and receipt.status
                in {
                    ResearchCopyStatus.PARTIALLY_COMPLETED,
                    ResearchCopyStatus.FAILED_RETRYABLE,
                }
            ):
                yield Button("Resume Copy", id="research-copy-resume", compact=True)
            yield Button("Close", id="research-copy-receipt-close", compact=True)

    @on(Button.Pressed, "#research-copy-resume")
    def resume(self) -> None:
        selection = (
            self.receipt.manifest.transfer_id
            if self.receipt
            else self.query_one(Select).value
        )
        if isinstance(selection, str):
            self.dismiss(selection)

    @on(Button.Pressed, "#research-copy-receipt-close")
    def close(self) -> None:
        self.dismiss(None)
