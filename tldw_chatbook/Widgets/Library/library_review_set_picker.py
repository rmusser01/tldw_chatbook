"""The Library review-set picker dialog (task-28243).

A dumb modal: it renders pre-computed ``(set_id, name, progress_label,
active)`` rows -- built by ``review_set_state.build_picker_rows`` in the
screen's worker -- and dismisses with one ``(action, set_id)`` decision:
``(PICKER_OPEN, id)`` to resume/switch, ``(PICKER_DISMISS, id)`` to
soft-delete, or ``None`` on cancel. All service and liveness work stays in
the screen.
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from tldw_chatbook.Library.library_shell_state import LIBRARY_CHOICE_ACTIVE_MARKER
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

PICKER_OPEN = "open"
"""Decision action: activate the chosen set and resume at its cursor."""

PICKER_DISMISS = "dismiss"
"""Decision action: soft-delete the chosen set."""

PickerDecision = tuple[str, str] | None
"""``(PICKER_OPEN, set_id)`` | ``(PICKER_DISMISS, set_id)`` | ``None``."""


class LibraryReviewSetPickerDialog(
    SafeModalDismissMixin, ModalScreen[PickerDecision]
):
    """List saved review sets with resume-or-dismiss actions per row."""

    BINDINGS = (Binding("escape", "request_safe_cancel", "Cancel", show=False),)
    SAFE_MODAL_CONTENT = "#library-review-set-picker-dialog"

    BUNDLED_CSS = """
    LibraryReviewSetPickerDialog { align: center middle; }
    LibraryReviewSetPickerDialog > Vertical {
        width: 72; max-height: 80%; height: auto; padding: 1 2;
        border: tall $accent; background: $surface;
    }
    #library-review-set-picker-rows { height: auto; max-height: 20; }
    .library-review-set-row { height: auto; }
    .library-review-set-open { width: 1fr; }
    #library-review-set-picker-actions { height: auto; align-horizontal: right; }
    """

    def __init__(self, rows: Sequence[tuple[str, str, str, bool]]) -> None:
        """Hold the pre-computed picker rows for compose.

        Args:
            rows: ``(set_id, name, progress_label, active)`` per saved set, in
                display order. Names derive from user input and are rendered
                as literal text (never markup).
        """
        super().__init__()
        self._rows = tuple(rows)

    def compose(self) -> ComposeResult:
        """Render the title, the scrollable set rows, and the Close action.

        The rows live in their own scrolling region so a long saved-set list
        cannot push lower rows or the Close action past the dialog's height
        cap (Qodo #2337).

        Returns:
            ComposeResult for the dialog.
        """
        with Vertical(id="library-review-set-picker-dialog"):
            yield Static(
                "Review sets",
                id="library-review-set-picker-title",
                classes="destination-section",
            )
            if not self._rows:
                yield Static(
                    "No saved review sets. Use “Review these” on the "
                    "media list to start one.",
                    id="library-review-set-picker-empty",
                )
            with VerticalScroll(id="library-review-set-picker-rows"):
                for index, (set_id, name, progress, active) in enumerate(
                    self._rows
                ):
                    marker = (
                        f"{LIBRARY_CHOICE_ACTIVE_MARKER} " if active else ""
                    )
                    with Horizontal(classes="library-review-set-row"):
                        open_button = Button(
                            Text(f"{marker}{name} — {progress}"),
                            id=f"library-review-set-open-{index}",
                            classes="library-review-set-open",
                            compact=True,
                        )
                        open_button.review_set_id = set_id
                        yield open_button
                        dismiss_button = Button(
                            "Dismiss",
                            id=f"library-review-set-dismiss-{index}",
                            classes="library-review-set-dismiss",
                            compact=True,
                        )
                        dismiss_button.review_set_id = set_id
                        yield dismiss_button
            with Horizontal(id="library-review-set-picker-actions"):
                yield Button("Close", id="library-review-set-picker-close")

    @on(Button.Pressed, ".library-review-set-open")
    def _open(self, event: Button.Pressed) -> None:
        """Dismiss with the open decision for the pressed row's set.

        Args:
            event: The row's open Button press.
        """
        event.stop()
        self.dismiss((PICKER_OPEN, str(event.button.review_set_id)))

    @on(Button.Pressed, ".library-review-set-dismiss")
    def _dismiss_set(self, event: Button.Pressed) -> None:
        """Dismiss with the soft-delete decision for the pressed row's set.

        Args:
            event: The row's Dismiss Button press.
        """
        event.stop()
        self.dismiss((PICKER_DISMISS, str(event.button.review_set_id)))

    @on(Button.Pressed, "#library-review-set-picker-close")
    async def _close(self, event: Button.Pressed) -> None:
        """Cancel the picker without a decision.

        Args:
            event: The Close Button press.
        """
        event.stop()
        await self.request_safe_cancel(source="visible")
