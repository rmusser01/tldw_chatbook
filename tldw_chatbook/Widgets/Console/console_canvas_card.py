"""Source-private transcript card for reopening a Canvas revision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static


async def open_canvas_with_textual(gateway: Any, scope: Any, app: Any) -> Any:
    """Open Canvas through Textual and preserve a manual recovery URL.

    ``CanvasGateway.open_shell`` absorbs platform-opener failures so a terminal
    stays responsive. This adapter keeps the one-time loopback URL visible and
    copyable through Textual's notification surface when that happens.
    """

    launch = await gateway.open_shell(scope, opener=app.open_url)
    if launch.opened is False:
        app.notify(
            f"Could not open a browser — {launch.browser_url}", severity="error"
        )
    return launch


@dataclass(frozen=True, slots=True)
class ConsoleCanvasCardPresentation:
    """Metadata-only projection of one Canvas revision."""

    canvas_id: str
    revision_id: str | None
    label: str
    digest: str
    reopenable: bool
    error_code: str | None


def canvas_card_signature(card: ConsoleCanvasCardPresentation) -> tuple:
    """Return the complete render identity for a Canvas transcript card."""

    return (
        "canvas-card",
        card.canvas_id,
        card.revision_id,
        card.label,
        card.digest,
        card.reopenable,
        card.error_code,
    )


class ConsoleCanvasCardOpenRequested(Message):
    """Request exact-revision or active-head Canvas navigation."""

    def __init__(
        self,
        *,
        canvas_id: str,
        revision_id: str | None,
        follow_latest: bool,
    ) -> None:
        super().__init__()
        self.canvas_id = canvas_id
        self.revision_id = revision_id
        self.follow_latest = follow_latest


class ConsoleCanvasCard(Vertical):
    """Focused Canvas card that never carries source bytes."""

    BUNDLED_CSS = """
    ConsoleCanvasCard {
        width: 100%;
        height: auto;
        border: round $surface-lighten-1;
        padding: 0 1;
        margin: 0 0 1 0;
    }

    ConsoleCanvasCard .console-canvas-card-kicker {
        color: $text-muted;
        text-style: bold;
        height: 1;
    }

    ConsoleCanvasCard .console-canvas-card-label {
        height: auto;
        min-height: 1;
    }

    ConsoleCanvasCard .console-canvas-card-error {
        color: $warning;
        height: auto;
        min-height: 1;
    }

    ConsoleCanvasCard .console-canvas-card-actions {
        height: 3;
        align-vertical: middle;
    }

    ConsoleCanvasCard .console-canvas-card-actions Button {
        min-width: 16;
        height: 3;
        margin-right: 1;
    }
    """

    def __init__(
        self,
        presentation: ConsoleCanvasCardPresentation,
        *,
        message_id: str,
        card_index: int,
    ) -> None:
        super().__init__(
            id=f"console-canvas-card-{message_id}-{card_index}",
            classes="console-transcript-canvas-card",
        )
        self.presentation = presentation
        self._id_suffix = f"{message_id}-{card_index}"

    def compose(self) -> ComposeResult:
        yield Static("Canvas artifact", classes="console-canvas-card-kicker")
        yield Static(self.presentation.label, classes="console-canvas-card-label")
        if self.presentation.error_code:
            yield Static(
                f"Revision unavailable · {self.presentation.error_code}",
                classes="console-canvas-card-error",
            )
        exact = Button(
            "Open revision",
            id=f"canvas-open-revision-{self._id_suffix}",
            disabled=not self.presentation.reopenable,
        )
        if not self.presentation.reopenable:
            exact.tooltip = "This exact Canvas revision is unavailable."
        yield Horizontal(
            exact,
            Button("Follow latest", id=f"canvas-follow-latest-{self._id_suffix}"),
            classes="console-canvas-card-actions",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == f"canvas-open-revision-{self._id_suffix}":
            if not self.presentation.reopenable:
                return
            follow_latest = False
            revision_id = self.presentation.revision_id
        elif event.button.id == f"canvas-follow-latest-{self._id_suffix}":
            follow_latest = True
            revision_id = None
        else:
            return
        event.stop()
        self.post_message(
            ConsoleCanvasCardOpenRequested(
                canvas_id=self.presentation.canvas_id,
                revision_id=revision_id,
                follow_latest=follow_latest,
            )
        )
