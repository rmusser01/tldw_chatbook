"""Explicit, session-only inspection of one captured progress inbox."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import ClassVar
from unicodedata import category

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, SelectionList, Static

from ...Agents.fleet_messages import MessageError, ProgressMessage
from ..modal_dismissal import SafeModalDismissMixin


def _literal(value: str) -> str:
    """Keep line breaks; expose control characters rather than executing them."""
    return "".join(
        char
        if char == "\n" or category(char) not in {"Cc", "Cf"}
        else char.encode("unicode_escape").decode("ascii")
        for char in value
    )


class ConsoleAgentProgressModal(SafeModalDismissMixin, ModalScreen[None]):
    """Inspect without collection; discard only selected displayed message IDs."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("escape", "request_safe_cancel", "Close")
    ]
    SAFE_MODAL_CONTENT = "#agent-progress-dialog"

    def __init__(
        self,
        *,
        conversation_id: str,
        load: Callable[[], tuple[ProgressMessage, ...]],
        discard: Callable[[Sequence[str]], int],
    ) -> None:
        super().__init__()
        self.conversation_id = conversation_id
        self._load = load
        self._discard = discard
        self._snapshot: tuple[ProgressMessage, ...] = ()
        self._poll_timer = None

    def compose(self) -> ComposeResult:
        with Vertical(id="agent-progress-dialog"):
            yield Static(
                "Queued progress (this session)", classes="console-modal-header"
            )
            yield Static(
                "Inspection does not collect reports or wake an idle supervisor. "
                "Restart loses this queue. Discard does not stop children or erase history copies. "
                "Queue full? Check progress counts in conversation navigation; other conversations may hold capacity.",
                id="agent-progress-scope",
                markup=False,
            )
            yield Static(
                "No queued progress in this view.",
                id="agent-progress-count",
                markup=False,
            )
            yield SelectionList[str](id="agent-progress-list")
            with VerticalScroll(id="agent-progress-detail"):
                yield Static("", id="agent-progress-body", markup=False)
            yield Static(
                "Arrows inspect · Space selects · Discard removes only checked reports.",
                id="agent-progress-status",
                markup=False,
            )
            with Horizontal(id="agent-progress-actions"):
                yield Button(
                    "Discard — none selected",
                    id="agent-progress-discard",
                    disabled=True,
                )
                yield Button("Close", id="agent-progress-close")

    def on_mount(self) -> None:
        self._refresh_snapshot()
        self._poll_timer = self.set_interval(0.5, self._refresh_snapshot)
        self.query_one(SelectionList).focus()

    def on_unmount(self) -> None:
        if self._poll_timer is not None:
            self._poll_timer.stop()

    def _refresh_snapshot(self) -> None:
        try:
            snapshot = self._load()
        except MessageError:
            snapshot = ()
            self.query_one("#agent-progress-status", Static).update(
                "This inbox is unavailable. Close and reopen progress for the current session."
            )
        listing = self.query_one(SelectionList)
        if snapshot != self._snapshot:
            selected = set(listing.selected)
            highlighted = (
                listing.get_option_at_index(listing.highlighted).value
                if listing.highlighted is not None and listing.option_count
                else None
            )
            listing.clear_options()
            listing.add_options(
                [
                    (
                        Text(
                            f"Report {index + 1} · {_literal(message.identity.agent)}"
                        ),
                        message.message_id,
                        message.message_id in selected,
                    )
                    for index, message in enumerate(snapshot)
                ]
            )
            self._snapshot = snapshot
            listing.highlighted = (
                next(
                    (
                        index
                        for index, message in enumerate(snapshot)
                        if message.message_id == highlighted
                    ),
                    0,
                )
                if snapshot
                else None
            )
        self.query_one("#agent-progress-count", Static).update(
            f"{len(snapshot)} queued · select reports to discard"
            if snapshot
            else "No queued progress in this view."
        )
        self._sync_selection()

    @on(SelectionList.SelectionHighlighted, "#agent-progress-list")
    def _sync_body(self) -> None:
        listing = self.query_one(SelectionList)
        highlighted = listing.highlighted
        message_id = (
            listing.get_option_at_index(highlighted).value
            if highlighted is not None and listing.option_count
            else None
        )
        message = next(
            (item for item in self._snapshot if item.message_id == message_id), None
        )
        selected = message_id in listing.selected
        text = (
            f"Inspecting report {highlighted + 1} · {'selected for discard' if selected else 'not selected'}\n"
            f"{_literal(message.identity.agent)} · run {_literal(message.identity.run_id)}"
            f" · child {_literal(message.identity.handle_id)}\n\n{_literal(message.body)}"
            if message
            else "Highlight a queued report to inspect its content."
        )
        self.query_one("#agent-progress-body", Static).update(Text(text))

    @on(SelectionList.SelectedChanged, "#agent-progress-list")
    def _sync_selection(self) -> None:
        count = len(self.query_one(SelectionList).selected)
        button = self.query_one("#agent-progress-discard", Button)
        button.disabled = not count
        button.label = (
            f"Discard selected ({count})" if count else "Discard — none selected"
        )
        self._sync_body()

    @on(Button.Pressed, "#agent-progress-discard")
    def _discard_selected(self, event: Button.Pressed) -> None:
        event.stop()
        displayed = {message.message_id for message in self._snapshot}
        selected = tuple(
            key for key in self.query_one(SelectionList).selected if key in displayed
        )
        try:
            count = self._discard(selected)
        except MessageError:
            self._refresh_snapshot()
            return
        self.query_one("#agent-progress-status", Static).update(
            f"Discarded {count} queued report{'s' if count != 1 else ''}. Existing history copies remain."
        )
        self._refresh_snapshot()

    @on(Button.Pressed, "#agent-progress-close")
    async def _close(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="visible")
