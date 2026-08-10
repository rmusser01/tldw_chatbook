"""I/O-free Console choices for a generated video outside managed storage."""

from __future__ import annotations

from typing import Literal

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

CapacityReason = Literal["over_capacity", "store_failure"]
CapacityAction = Literal["keep", "save_external", "discard"]


def _format_bytes(size_bytes: int) -> str:
    """Return a compact binary-unit size for the modal copy."""
    mib = size_bytes / (1024 * 1024)
    return f"{mib:.1f} MiB"


class ConsoleVideoCapacityModal(ModalScreen[CapacityAction]):
    """Ask where to put a generated video that is not in the managed store."""

    DEFAULT_CSS = """
    ConsoleVideoCapacityModal {
        align: center middle;
    }

    #video-capacity-dialog {
        width: 76;
        max-width: 100%;
        height: auto;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #video-capacity-summary,
    #video-capacity-guidance {
        width: 100%;
        height: auto;
    }

    #video-capacity-summary {
        margin: 1 0 0 0;
    }

    #video-capacity-guidance {
        color: $text-muted;
        margin: 1 0;
    }

    #video-capacity-actions {
        width: 100%;
        height: 3;
        align-horizontal: center;
    }

    #video-capacity-actions Button {
        width: auto;
        min-width: 10;
        height: 3;
        margin: 0 1;
    }
    """

    BINDINGS = [("escape", "discard", "Discard")]

    def __init__(
        self,
        *,
        reason: CapacityReason,
        size_bytes: int,
        max_bytes: int,
    ) -> None:
        """Initialize the modal from safe, display-only capacity facts.

        Args:
            reason: Why the result remains outside managed storage.
            size_bytes: Size of the generated payload.
            max_bytes: Configured managed-store capacity.
        """
        if reason not in ("over_capacity", "store_failure"):
            raise ValueError("Unsupported generated-video storage reason.")
        super().__init__()
        self._reason = reason
        self._size_bytes = size_bytes
        self._max_bytes = max_bytes

    def compose(self) -> ComposeResult:
        """Compose the reason-specific copy and three terminal actions."""
        if self._reason == "over_capacity":
            guidance = (
                "This generated video exceeds the configured capacity. Keep it "
                "here by removing other videos, save it to disk, or discard it."
            )
            keep_label = "Keep here (remove other videos)"
            keep_variant = "default"
            save_variant = "primary"
        else:
            guidance = (
                "This generated video could not be stored here. Retry, save it to "
                "disk, or discard it."
            )
            keep_label = "Retry"
            keep_variant = "primary"
            save_variant = "default"

        with Vertical(id="video-capacity-dialog"):
            yield Static("Generated video", classes="console-modal-header", markup=False)
            yield Static(
                "Generated size: "
                f"{_format_bytes(self._size_bytes)} · Configured capacity: "
                f"{_format_bytes(self._max_bytes)}",
                id="video-capacity-summary",
                markup=False,
            )
            yield Static(
                guidance,
                id="video-capacity-guidance",
                markup=False,
            )
            with Horizontal(id="video-capacity-actions"):
                yield Button(
                    keep_label,
                    id="video-capacity-keep",
                    variant=keep_variant,
                )
                yield Button(
                    "Save to disk",
                    id="video-capacity-save",
                    variant=save_variant,
                )
                yield Button("Discard", id="video-capacity-discard")

    def on_mount(self) -> None:
        """Focus the safest reason-specific default action."""
        button_id = (
            "video-capacity-save"
            if self._reason == "over_capacity"
            else "video-capacity-keep"
        )
        self.query_one(f"#{button_id}", Button).focus()

    def action_discard(self) -> None:
        """Treat Escape as an explicit discard decision."""
        self.dismiss("discard")

    @on(Button.Pressed, "#video-capacity-keep")
    def _keep(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss("keep")

    @on(Button.Pressed, "#video-capacity-save")
    def _save_external(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss("save_external")

    @on(Button.Pressed, "#video-capacity-discard")
    def _discard(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss("discard")
