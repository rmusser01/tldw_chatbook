"""Explicit inspection of one captured progress inbox."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import ClassVar
from unicodedata import category

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, SelectionList, Static

from ...Agents.fleet_messages import MessageError, ProgressMessage
from ...Backup_Recovery.participants import run_finite_local_worker
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
        prepare: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.conversation_id = conversation_id
        self._load = load
        self._discard = discard
        self._prepare = prepare
        self._load_error: str | None = None
        self._snapshot: tuple[ProgressMessage, ...] = ()
        self._poll_timer = None
        self._discarding = False
        self._discard_generation = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="agent-progress-dialog"):
            yield Static("Queued progress", classes="console-modal-header")
            yield Static(
                "Inspection does not collect reports or wake an idle supervisor. "
                "Saved chats retain queued reports after restart. Temporary chats lose them on close or restart unless saved. "
                "Discard does not stop children or erase history copies. "
                "Queue full? Check progress counts in conversation navigation; other conversations may hold capacity.",
                id="agent-progress-scope",
                markup=False,
            )
            yield Static(
                "Loading saved progress…"
                if self._prepare
                else "No queued progress in this view.",
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
        self.query_one(SelectionList).focus()
        if self._prepare is not None:
            self.run_worker(
                self._prepare_inbox,
                thread=True,
                exclusive=True,
                group="agent-progress-initial-load",
            )
        else:
            self._start_polling()

    def _prepare_inbox(self) -> None:
        error = None
        try:
            run_finite_local_worker(self._prepare)
        except MessageError as refusal:
            error = refusal.code
        except Exception:  # noqa: BLE001 - an inspection load grants no authority
            error = "durable_unavailable"
        try:
            self.app.call_from_thread(self._prepared, error)
        except RuntimeError:
            return

    def _prepared(self, error: str | None) -> None:
        if not self.is_mounted or self not in self.app.screen_stack:
            return
        self._load_error = error
        if error is not None:
            self._refresh_snapshot()
        else:
            self._start_polling()

    def _start_polling(self) -> None:
        self._refresh_snapshot()
        self._poll_timer = self.set_interval(0.5, self._refresh_snapshot)

    def on_unmount(self) -> None:
        self._discard_generation += 1
        if self._poll_timer is not None:
            self._poll_timer.stop()

    def _refresh_snapshot(self) -> None:
        error = self._load_error
        try:
            if error is not None:
                raise MessageError(error)
            snapshot = self._load()
        except MessageError as refusal:
            snapshot = ()
            error = refusal.code
            status = self.query_one("#agent-progress-status", Static)
            message = (
                "Saved reports remain queued. Close another chat to free live capacity, then reopen progress."
                if error == "queue_full"
                else "This inbox is unavailable. Close and reopen progress for the current session."
            )
            if str(status.renderable) != message:
                status.update(message)
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
                        Text(f"Report {index + 1} · {_literal(message.identity.agent)}"),
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
        count = self.query_one("#agent-progress-count", Static)
        count_text = (
            f"{len(snapshot)} queued · select reports to discard"
            if snapshot
            else "Progress unavailable · live capacity is full"
            if error == "queue_full"
            else "Progress unavailable"
            if error is not None
            else "No queued progress in this view."
        )
        if str(count.renderable) != count_text:
            count.update(count_text)
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
        body = self.query_one("#agent-progress-body", Static)
        content = Text(text)
        if body.renderable != content:
            body.update(content)

    @on(SelectionList.SelectedChanged, "#agent-progress-list")
    def _sync_selection(self) -> None:
        count = len(self.query_one(SelectionList).selected)
        button = self.query_one("#agent-progress-discard", Button)
        button.disabled = self._discarding or not count
        button.label = (
            "Discarding…"
            if self._discarding
            else f"Discard selected ({count})"
            if count
            else "Discard — none selected"
        )
        self._sync_body()

    @on(Button.Pressed, "#agent-progress-discard")
    def _discard_selected(self, event: Button.Pressed) -> None:
        event.stop()
        if self._discarding:
            return
        displayed = {message.message_id for message in self._snapshot}
        selected = tuple(
            key for key in self.query_one(SelectionList).selected if key in displayed
        )
        if not selected:
            return
        self._discarding = True
        self._discard_generation += 1
        self._sync_selection()
        self.query_one("#agent-progress-status", Static).update(
            "Discarding selected reports…"
        )
        self.run_worker(
            partial(self._discard_reports, selected, self._discard_generation),
            thread=True,
            exclusive=True,
            group="agent-progress-discard",
        )

    def _discard_reports(self, selected: tuple[str, ...], generation: int) -> None:
        count = 0
        error = None
        try:
            count = run_finite_local_worker(self._discard, selected)
        except MessageError as refusal:
            error = refusal.code
        except Exception:  # noqa: BLE001 - an uncertain discard never retries
            error = "durable_unavailable"
        try:
            self.app.call_from_thread(self._discard_completed, count, error, generation)
        except RuntimeError:
            return

    def _discard_completed(
        self, count: int, error: str | None, generation: int
    ) -> None:
        if (
            not self.is_mounted
            or self not in self.app.screen_stack
            or generation != self._discard_generation
        ):
            return
        self._discarding = False
        self._refresh_snapshot()
        self.query_one("#agent-progress-status", Static).update(
            "Discard could not be confirmed. Close and reopen progress before retrying."
            if error is not None
            else f"Discarded {count} queued report{'s' if count != 1 else ''}. Existing history copies remain."
        )

    @on(Button.Pressed, "#agent-progress-close")
    async def _close(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="visible")
