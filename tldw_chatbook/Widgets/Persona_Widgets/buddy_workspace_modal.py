"""Native workspace activity inbox with explicit result acknowledgement."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, ClassVar

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, OptionList, Static
from textual.widgets.option_list import Option

from tldw_chatbook.Persona_Buddy.inbox import BuddyInboxEntry
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin


class BuddyWorkspaceModal(SafeModalDismissMixin, ModalScreen[None]):
    """Display live rows while keeping keyboard focus and receipt ownership stable."""

    SAFE_MODAL_CONTENT = "#buddy-inbox"
    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("escape", "request_safe_cancel", "Close")
    ]
    DEFAULT_CSS = """
    BuddyWorkspaceModal { align: center middle; }
    #buddy-inbox {
        width: 82; max-width: 96%; height: 85%; max-height: 42;
        border: round $accent; background: $panel; padding: 0 1;
    }
    #buddy-inbox-title { height: 2; text-style: bold; padding-top: 1; }
    #buddy-inbox-help { height: auto; color: $text-muted; }
    #buddy-inbox-list { height: 1fr; margin-top: 1; }
    #buddy-inbox-error { height: auto; color: $error; }
    #buddy-inbox-actions { height: 3; min-height: 3; align-horizontal: right; }
    #buddy-inbox-actions Button { width: auto; min-width: 8; margin-left: 1; }
    """

    def __init__(
        self,
        *,
        snapshot: Callable[[], Any],
        open_entry: Callable[[BuddyInboxEntry], Any],
        acknowledge: Callable[[BuddyInboxEntry], Any],
        speech: Any = None,
    ) -> None:
        super().__init__()
        self._snapshot = snapshot
        self._open_entry = open_entry
        self._acknowledge = acknowledge
        self._speech = speech
        self._entries: tuple[BuddyInboxEntry, ...] = ()
        self._refreshing = False
        self._loaded = False

    def compose(self) -> ComposeResult:
        with Vertical(id="buddy-inbox"):
            yield Static("Workspace Buddy", id="buddy-inbox-title", markup=False)
            yield Static(
                "Open a conversation to review or reply. Results stay unread until marked seen.",
                id="buddy-inbox-help",
            )
            yield OptionList(id="buddy-inbox-list")
            if self._speech is not None:
                from .buddy_speech_controls import BuddySpeechControls

                yield BuddySpeechControls(self._speech)
            yield Static("", id="buddy-inbox-error", markup=False)
            with Horizontal(id="buddy-inbox-actions"):
                yield Button(
                    "Open", id="buddy-inbox-open", variant="primary", disabled=True
                )
                yield Button("Mark seen", id="buddy-inbox-seen", disabled=True)
                yield Button("Close", id="buddy-inbox-close")

    async def on_mount(self) -> None:
        self.call_after_refresh(self.refresh_inbox)
        self.set_interval(1.0, self.refresh_inbox)

    def selected_entry(self) -> BuddyInboxEntry | None:
        listing = self.query_one("#buddy-inbox-list", OptionList)
        if listing.highlighted is None:
            return None
        key = listing.get_option_at_index(listing.highlighted).id
        return next((entry for entry in self._entries if entry.key == key), None)

    async def refresh_inbox(self) -> None:
        """Refresh the projection without acknowledging or moving focus."""
        if self._refreshing or not self.is_mounted:
            return
        self._refreshing = True
        try:
            title, entries = await self._snapshot()
            if not self.is_mounted:
                return
            self.query_one("#buddy-inbox-title", Static).update(f"Buddy · {title}")
            self.query_one("#buddy-inbox-error", Static).update("")
            if entries == self._entries and self._loaded:
                self._sync_actions()
                return
            self._loaded = True
            selected = self.selected_entry()
            self._entries = entries
            listing = self.query_one("#buddy-inbox-list", OptionList)
            listing.clear_options()
            if not entries:
                listing.add_option(
                    Option(
                        Text("No active work or unread results in this workspace."),
                        disabled=True,
                    )
                )
            for group, label in (
                ("needs_you", "Needs you"),
                ("running", "Running"),
                ("results", "Results"),
            ):
                rows = [entry for entry in entries if entry.group == group]
                listing.add_option(
                    Option(Text(f"{label} ({len(rows)})", style="bold"), disabled=True)
                )
                for entry in rows:
                    listing.add_option(
                        Option(Text(f"{entry.title}\n{entry.summary}"), id=entry.key)
                    )
            if selected and any(row.key == selected.key for row in entries):
                listing.highlighted = listing.get_option_index(selected.key)
            elif entries:
                listing.highlighted = listing.get_option_index(entries[0].key)
            self._sync_actions()
        except Exception as exc:  # noqa: BLE001 - storage failure cannot break a live modal
            if self.is_mounted:
                message = (
                    str(exc)
                    if isinstance(exc, ValueError)
                    else "Could not refresh the inbox. Retry by reopening it."
                )
                self.query_one("#buddy-inbox-error", Static).update(message)
                self.query_one("#buddy-inbox-open", Button).disabled = True
                self.query_one("#buddy-inbox-seen", Button).disabled = True
        finally:
            self._refreshing = False

    def _sync_actions(self) -> None:
        entry = self.selected_entry()
        self.query_one("#buddy-inbox-open", Button).disabled = entry is None
        self.query_one("#buddy-inbox-seen", Button).disabled = (
            entry is None or not entry.receipt_ids
        )

    @on(OptionList.OptionHighlighted, "#buddy-inbox-list")
    def _highlighted(self) -> None:
        self._sync_actions()

    @on(OptionList.OptionSelected, "#buddy-inbox-list")
    def _selected(self) -> None:
        self._request_open()

    def _request_open(self) -> None:
        entry = self.selected_entry()
        if entry is not None:
            self._open_entry(entry)

    @on(Button.Pressed)
    async def _button(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "buddy-inbox-close":
            await self.action_request_safe_cancel()
        elif event.button.id == "buddy-inbox-open":
            self._request_open()
        elif event.button.id == "buddy-inbox-seen":
            entry = self.selected_entry()
            if entry is not None and entry.receipt_ids:
                self.app.run_worker(
                    self._mark_seen(entry),
                    group="buddy-inbox-acknowledge",
                    exclusive=False,
                )

    async def _mark_seen(self, entry: BuddyInboxEntry) -> None:
        try:
            pending = self._acknowledge(entry)
            count = await pending if inspect.isawaitable(pending) else pending
            if count == 0:
                raise ValueError("Could not confirm the result was marked seen. Retry.")
        except Exception as exc:  # noqa: BLE001 - retain receipt on failed storage
            self.app.notify(
                str(exc)
                if isinstance(exc, ValueError)
                else "Could not mark the result seen. Retry.",
                severity="warning",
            )
        if self.is_mounted:
            await self.refresh_inbox()
