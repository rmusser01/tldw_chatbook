"""Small presentation-only naming and recovery modal for Console chat forks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import inspect
from typing import Literal

from pydantic import ValidationError as PydanticValidationError
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from tldw_chatbook.Utils.input_validation import (
    CONSOLE_FORK_TITLE_MAX_LENGTH,
    ConsoleForkTitleInput,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin


ConsoleForkModalState = Literal[
    "editing",
    "validating",
    "committing",
    "precommit_error",
    "stale_source",
    "created_not_opened",
]


@dataclass(frozen=True, slots=True)
class ConsoleForkDialogSummary:
    """Immutable user-visible facts captured when the fork dialog opens."""

    default_title: str
    boundary_label: str
    boundary_excerpt: str
    message_count: int
    response_variant: str | None
    destination: str
    temporary: bool
    includes_attachments: bool
    includes_citations: bool
    contains_video: bool


@dataclass(frozen=True, slots=True)
class ConsoleForkSubmitResult:
    """One normalized title submitted by the naming modal."""

    title: str


class _ConsoleForkStatus(Static):
    can_focus = True


class _ConsoleForkTitleInput(Input):
    """Guard queued opening keys and freeze accepted titles without losing focus."""

    opened_at: float | None = None
    locked: bool = False

    async def _on_key(self, event: events.Key) -> None:
        if self.opened_at is not None and event.time < self.opened_at:
            event.stop()
            event.prevent_default()
            return
        await super()._on_key(event)

    def replace(self, text: str, start: int, end: int) -> None:
        """Keep the accepted title immutable without surrendering focus."""

        if not self.locked:
            super().replace(text, start, end)


class ConsoleForkChatModal(SafeModalDismissMixin, ModalScreen[None]):
    """Render fork facts and state while a controller owns all side effects."""

    DEFAULT_CSS = """
    ConsoleForkChatModal {
        align: center middle;
        background: $background 70%;
    }

    #console-fork-chat-modal {
        width: 74;
        max-width: 96%;
        height: 28;
        max-height: 96%;
        border: round $primary;
        background: $panel;
        padding: 1 2;
        overflow-y: hidden;
    }

    #console-fork-chat-content {
        height: 1fr;
        min-height: 0;
        overflow-y: auto;
        overflow-x: hidden;
        scrollbar-gutter: stable;
    }

    #console-fork-chat-heading {
        text-style: bold;
        color: $text;
        height: 1;
    }

    .console-fork-chat-summary {
        height: auto;
        color: $text-muted;
    }

    #console-fork-chat-boundary {
        color: $text;
        margin-top: 1;
        max-height: 2;
        text-overflow: ellipsis;
    }

    #console-fork-chat-destination {
        color: $accent;
        margin-top: 1;
    }

    #console-fork-chat-warning {
        color: $warning;
    }

    #console-fork-chat-disclosure {
        width: auto;
        min-width: 20;
        height: 1;
        min-height: 1;
        padding: 0 1;
        margin-top: 1;
        border: none;
    }

    #console-fork-chat-exclusions {
        display: none;
        height: auto;
        color: $text-muted;
    }

    #console-fork-chat-name-label {
        margin-top: 1;
        height: 1;
        text-style: bold;
    }

    #console-fork-chat-title {
        width: 100%;
        height: 3;
        border: round $surface-lighten-1;
    }

    #console-fork-chat-title:focus {
        border: round $accent;
    }

    #console-fork-chat-status {
        width: 100%;
        height: auto;
        min-height: 1;
        color: $text-muted;
    }

    #console-fork-chat-status:focus {
        background: $accent 10%;
        color: $text;
        text-style: bold;
    }

    #console-fork-chat-actions {
        width: 100%;
        height: 3;
        align-horizontal: right;
    }

    #console-fork-chat-actions Button {
        width: auto;
        min-width: 10;
        height: 3;
        min-height: 3;
    }

    #console-fork-chat-open {
        display: none;
    }

    """

    SAFE_MODAL_CONTENT = "#console-fork-chat-modal"
    BINDINGS = [("escape", "request_safe_cancel", "Cancel")]

    def __init__(
        self,
        summary: ConsoleForkDialogSummary,
        *,
        on_submit: Callable[[ConsoleForkSubmitResult], object],
        on_cancel: Callable[[], object] | None = None,
        on_open: Callable[[], object] | None = None,
    ) -> None:
        super().__init__()
        self.summary = summary
        self.state: ConsoleForkModalState = "editing"
        self._on_submit = on_submit
        self._on_cancel = on_cancel
        self._on_open = on_open
        self._disclosure_open = False

    def compose(self) -> ComposeResult:
        summary = self.summary
        detail = f" · {summary.response_variant}" if summary.response_variant else ""
        with Vertical(id="console-fork-chat-modal"):
            with VerticalScroll(id="console-fork-chat-content", can_focus=False):
                yield Static("Fork chat", id="console-fork-chat-heading", markup=False)
                yield Static(
                    f"{summary.boundary_label}: “{summary.boundary_excerpt}”",
                    id="console-fork-chat-boundary",
                    classes="console-fork-chat-summary",
                    markup=False,
                )
                count_label = "message" if summary.message_count == 1 else "messages"
                yield Static(
                    f"{summary.message_count} {count_label}{detail}",
                    classes="console-fork-chat-summary",
                    markup=False,
                )
                yield Static(
                    f"Creates: {summary.destination}",
                    id="console-fork-chat-destination",
                    classes="console-fork-chat-summary",
                    markup=False,
                )
                if summary.temporary:
                    yield Static(
                        "Saving this fork will not save the original chat.",
                        classes="console-fork-chat-summary",
                        markup=False,
                    )
                    if summary.includes_attachments:
                        yield Static(
                            "Includes sent attachments.",
                            classes="console-fork-chat-summary",
                            markup=False,
                        )
                    yield Static(
                        "Citation markers remain in the message text; source inspector "
                        "details are not copied.",
                        classes="console-fork-chat-summary",
                        markup=False,
                    )
                elif summary.includes_attachments or summary.includes_citations:
                    yield Static(
                        self._included_copy(),
                        classes="console-fork-chat-summary",
                        markup=False,
                    )
                yield Static(
                    "Starts with new private working files; file and tool access will "
                    "be requested again.",
                    classes="console-fork-chat-summary",
                    markup=False,
                )
                if summary.contains_video:
                    yield Static(
                        "This video will appear as unavailable in the fork. Save a copy "
                        "first if you need the file.",
                        id="console-fork-chat-warning",
                        classes="console-fork-chat-summary",
                        markup=False,
                    )
                yield Button(
                    "What is not copied",
                    id="console-fork-chat-disclosure",
                    variant="default",
                )
                yield Static(
                    "Runs, tool history, drafts, staged files, temporary working files, "
                    "and prior permissions are not copied."
                    + (
                        " Inspectable citation provenance is also not copied."
                        if summary.temporary
                        else ""
                    ),
                    id="console-fork-chat-exclusions",
                    markup=False,
                )
                yield Static("Name", id="console-fork-chat-name-label", markup=False)
                yield _ConsoleForkTitleInput(
                    value=summary.default_title,
                    max_length=CONSOLE_FORK_TITLE_MAX_LENGTH,
                    id="console-fork-chat-title",
                )
                yield _ConsoleForkStatus(
                    "Ready to create an independent fork.",
                    id="console-fork-chat-status",
                    markup=False,
                )
            with Horizontal(id="console-fork-chat-actions"):
                yield Button("Cancel", id="console-fork-chat-cancel")
                yield Button(
                    "Open fork", id="console-fork-chat-open", variant="primary"
                )
                yield Button(
                    "Fork chat", id="console-fork-chat-confirm", variant="primary"
                )

    def _included_copy(self) -> str:
        if self.summary.includes_attachments and self.summary.includes_citations:
            return "Includes sent attachments and cited source details."
        if self.summary.includes_attachments:
            return "Includes sent attachments."
        return "Includes cited source details."

    def on_mount(self, event: events.Mount) -> None:  # type: ignore[override]
        super().on_mount()
        self._opened_at = event.time
        title = self.query_one("#console-fork-chat-title", _ConsoleForkTitleInput)
        title.opened_at = event.time
        title.focus()
        title.action_select_all()
        content = self.query_one("#console-fork-chat-content", VerticalScroll)
        self.call_after_refresh(
            content.scroll_end, animate=False, immediate=True, force=True
        )

    def _invoke(self, callback: Callable[..., object], *args: object) -> None:
        result = callback(*args)
        if inspect.isawaitable(result):
            self.run_worker(result, exclusive=False, exit_on_error=False)

    def _set_status(self, copy: str) -> None:
        self.query_one("#console-fork-chat-status", Static).update(copy)

    def _request_submit(self) -> None:
        if self.state not in {"editing", "precommit_error"}:
            return
        title_input = self.query_one("#console-fork-chat-title", Input)
        try:
            title = ConsoleForkTitleInput.model_validate(
                {"title": title_input.value}
            ).title
        except PydanticValidationError:
            self._set_status("Fork title cannot be blank.")
            title_input.focus()
            return
        title_input.value = title
        self.show_validating()
        self._invoke(self._on_submit, ConsoleForkSubmitResult(title=title))

    @on(Input.Submitted, "#console-fork-chat-title")
    def _submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._request_submit()

    @on(Button.Pressed, "#console-fork-chat-confirm")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self._request_submit()

    @on(Button.Pressed, "#console-fork-chat-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    @on(Button.Pressed, "#console-fork-chat-open")
    def _open(self, event: Button.Pressed) -> None:
        event.stop()
        if self.state != "created_not_opened" or self._on_open is None:
            return
        event.button.disabled = True
        self._set_status("Opening the existing fork…")
        self._invoke(self._on_open)

    @on(Button.Pressed, "#console-fork-chat-disclosure")
    def _toggle_disclosure(self, event: Button.Pressed) -> None:
        event.stop()
        self._disclosure_open = not self._disclosure_open
        self.query_one(
            "#console-fork-chat-exclusions", Static
        ).display = self._disclosure_open
        event.button.label = (
            "Hide what is not copied" if self._disclosure_open else "What is not copied"
        )

    async def _perform_safe_cancel(self, *, source: str) -> None:
        if source == "escape" and self._disclosure_open:
            disclosure = self.query_one("#console-fork-chat-disclosure", Button)
            self._disclosure_open = False
            self.query_one("#console-fork-chat-exclusions", Static).display = False
            disclosure.label = "What is not copied"
            disclosure.focus()
            return
        if self.state == "committing":
            self._set_status(
                "Fork creation is finishing and can no longer be cancelled."
            )
            self.query_one("#console-fork-chat-status", Static).focus()
            return
        if self._on_cancel is not None:

            async def cancel_once() -> None:
                result = self._on_cancel()
                if inspect.isawaitable(result):
                    await result

            await self.run_cancel_effect_once(cancel_once)
        panel = self.query_one(self.SAFE_MODAL_CONTENT)
        if panel.is_attached:
            await panel.remove()
        self.dismiss_safe_once(None)

    def show_validating(self) -> None:
        self.state = "validating"
        self._set_status("Checking fork…")
        title = self.query_one("#console-fork-chat-title", _ConsoleForkTitleInput)
        title.locked = True
        title.focus()
        self.query_one("#console-fork-chat-content", VerticalScroll).scroll_end(
            animate=False, immediate=True, force=True
        )
        self.query_one("#console-fork-chat-confirm", Button).disabled = True

    def show_committing(self) -> None:
        self.state = "committing"
        self._set_status("Forking…")
        self.query_one("#console-fork-chat-title", Input).disabled = True
        confirm = self.query_one("#console-fork-chat-confirm", Button)
        confirm.label = "Forking…"
        for button in self.query(Button):
            button.disabled = True
        self.query_one("#console-fork-chat-status", Static).focus()
        self.query_one("#console-fork-chat-content", VerticalScroll).scroll_end(
            animate=False, immediate=True, force=True
        )

    def show_precommit_error(self, error: str, *, retryable: bool = True) -> None:
        self.state = "precommit_error"
        self._set_status(error)
        title = self.query_one("#console-fork-chat-title", _ConsoleForkTitleInput)
        confirm = self.query_one("#console-fork-chat-confirm", Button)
        close = self.query_one("#console-fork-chat-cancel", Button)
        self.query_one("#console-fork-chat-disclosure", Button).disabled = False
        title.locked = not retryable
        title.disabled = not retryable
        close.disabled = False
        close.label = "Cancel" if retryable else "Close"
        confirm.disabled = not retryable
        confirm.display = retryable
        if retryable:
            confirm.label = "Retry"
            confirm.focus()
        else:
            close.focus()
        content = self.query_one("#console-fork-chat-content", VerticalScroll)
        self.call_after_refresh(
            content.scroll_end, animate=False, immediate=True, force=True
        )

    def show_stale_source(self) -> None:
        self.state = "stale_source"
        self._set_status("This chat changed. Close and choose Fork again.")
        self.query_one("#console-fork-chat-title", Input).disabled = True
        close = self.query_one("#console-fork-chat-cancel", Button)
        close.disabled = False
        close.label = "Close"
        self.query_one("#console-fork-chat-confirm", Button).display = False
        self.query_one("#console-fork-chat-open", Button).display = False
        close.focus()

    def show_created_not_opened(
        self, *, title: str, identity: str, detail: str
    ) -> None:
        self.state = "created_not_opened"
        self.query_one("#console-fork-chat-title", Input).value = title
        self.query_one("#console-fork-chat-title", Input).disabled = True
        self._set_status(f"{detail} Identity: {identity}.")
        close = self.query_one("#console-fork-chat-cancel", Button)
        close.disabled = False
        close.label = "Close"
        self.query_one("#console-fork-chat-confirm", Button).display = False
        open_button = self.query_one("#console-fork-chat-open", Button)
        open_button.display = True
        open_button.disabled = False
        open_button.focus()
        content = self.query_one("#console-fork-chat-content", VerticalScroll)
        self.call_after_refresh(
            content.scroll_end, animate=False, immediate=True, force=True
        )

    def close_after_success(self) -> None:
        """Dismiss this exact mounted modal after controller-owned activation."""
        self.dismiss_safe_once(None)
