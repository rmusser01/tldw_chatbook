"""Console transcript message edit modal."""

from __future__ import annotations

from dataclasses import dataclass

from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea

from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin


@dataclass(frozen=True)
class ConsoleEditResult:
    """Outcome of the edit modal: the (possibly unchanged) text, and whether
    the caller asked to fork a new branch (Console branching Phase B) rather
    than edit the message in place."""

    text: str
    resend: bool


class _EditMessageTextArea(TextArea):
    """TextArea that ignores keys typed before the modal appeared.

    TASK-360: the edit action dispatches through a Button.Pressed hop and an
    async modal push; keys pressed in that gap (e.g. a retry `e` because
    nothing visibly happened) used to land here as text and silently corrupt
    the draft. A key whose event time predates the modal's mount was aimed
    at whatever the user was looking at then — never at this textarea.
    """

    opened_at: float | None = None

    async def _on_key(self, event: events.Key) -> None:
        if self.opened_at is not None and event.time < self.opened_at:
            event.stop()
            event.prevent_default()
            return
        await super()._on_key(event)


class ConsoleEditMessageModal(
    SafeModalDismissMixin, ModalScreen[ConsoleEditResult | None]
):
    """Edit an existing Console transcript message without using the composer."""

    DEFAULT_CSS = """
    ConsoleEditMessageModal {
        align: center middle;
    }

    #console-edit-message-modal {
        width: 92;
        height: 28;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #console-edit-message-context {
        height: auto;
        margin: 1 0 1 0;
    }

    #console-edit-message-body {
        width: 100%;
        height: 1fr;
        min-height: 8;
    }

    #console-edit-message-error {
        height: auto;
        min-height: 1;
        color: red;
    }

    #console-edit-message-actions {
        height: 3;
        min-height: 3;
        margin: 1 0 0 0;
        align-horizontal: right;
    }

    #console-edit-message-cancel,
    #console-edit-message-save,
    #console-edit-message-resend {
        width: 10;
        min-width: 10;
        height: 3;
        min-height: 3;
    }

    #console-edit-message-resend {
        width: 18;
        min-width: 18;
    }
    """

    SAFE_MODAL_CONTENT = "#console-edit-message-modal"
    BINDINGS = [("escape", "request_safe_cancel", "Cancel")]

    def __init__(
        self,
        *,
        content: str,
        can_resend: bool = False,
        clears_generation_provenance: bool = False,
    ) -> None:
        super().__init__()
        self._content = content
        self._can_resend = can_resend
        self._clears_generation_provenance = clears_generation_provenance

    def compose(self) -> ComposeResult:
        with Vertical(id="console-edit-message-modal"):
            yield Static("Edit Message", classes="console-modal-header")
            if self._clears_generation_provenance:
                context_copy = (
                    "Editing this generated answer. Saving clears model thinking and "
                    "provider continuation for this answer. Cancel keeps both intact."
                )
            elif self._can_resend:
                context_copy = (
                    "Editing existing transcript message. Save keeps the edit in "
                    "place; Edit & resend creates a new response branch in this chat "
                    "and gets a fresh reply."
                )
            else:
                context_copy = "Editing existing transcript message. This will not create a new prompt."
            yield Static(
                context_copy,
                id="console-edit-message-context",
                markup=False,
            )
            yield _EditMessageTextArea(self._content, id="console-edit-message-body")
            yield Static("", id="console-edit-message-error", markup=False)
            with Horizontal(id="console-edit-message-actions"):
                yield Button("Cancel", id="console-edit-message-cancel")
                yield Button(
                    "Save",
                    id="console-edit-message-save",
                    variant="default" if self._can_resend else "primary",
                )
                if self._can_resend:
                    yield Button(
                        "Edit & resend",
                        id="console-edit-message-resend",
                        variant="primary",
                    )

    # Textual supplies the event while composing MRO message handlers, so this
    # event-shaped handler is not an OO override of the mixin hook.
    def on_mount(self, event: events.Mount) -> None:  # type: ignore[override]
        # Event time shares the clock domain of Key.time — the stale-key
        # guard compares against it (TASK-360).
        self._opened_at = event.time
        area = self.query_one("#console-edit-message-body", _EditMessageTextArea)
        area.opened_at = event.time
        area.focus()

    @on(Button.Pressed, "#console-edit-message-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    @on(Button.Pressed, "#console-edit-message-save")
    def _save(self, event: Button.Pressed) -> None:
        event.stop()
        edited_content = self.query_one("#console-edit-message-body", TextArea).text
        if not edited_content.strip():
            self.query_one("#console-edit-message-error", Static).update(
                "Message content cannot be blank."
            )
            return
        self.dismiss(ConsoleEditResult(text=edited_content, resend=False))

    @on(Button.Pressed, "#console-edit-message-resend")
    def _resend(self, event: Button.Pressed) -> None:
        event.stop()
        edited_content = self.query_one("#console-edit-message-body", TextArea).text
        if not edited_content.strip():
            self.query_one("#console-edit-message-error", Static).update(
                "Message content cannot be blank."
            )
            return
        self.dismiss(ConsoleEditResult(text=edited_content, resend=True))


@dataclass(frozen=True)
class ConsoleThinkingEditResult:
    """Outcome of the thinking-block edit modal: the block's edited text.

    ADR-090 amendment (TASK-32312): only the displayable block's text
    changes; the answer, provenance, and replay encoding stay intact.
    """

    text: str


class ConsoleEditThinkingModal(
    SafeModalDismissMixin, ModalScreen[ConsoleThinkingEditResult | None]
):
    """Edit one displayable thinking block's text without touching the answer."""

    DEFAULT_CSS = """
    ConsoleEditThinkingModal {
        align: center middle;
    }

    #console-edit-thinking-modal {
        width: 92;
        height: 28;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #console-edit-thinking-context {
        height: auto;
        margin: 1 0 1 0;
    }

    #console-edit-thinking-body {
        width: 100%;
        height: 1fr;
        min-height: 8;
    }

    #console-edit-thinking-error {
        height: auto;
        min-height: 1;
        color: red;
    }

    #console-edit-thinking-actions {
        height: 3;
        min-height: 3;
        margin: 1 0 0 0;
        align-horizontal: right;
    }

    #console-edit-thinking-cancel,
    #console-edit-thinking-save {
        width: 10;
        min-width: 10;
        height: 3;
        min-height: 3;
    }
    """

    SAFE_MODAL_CONTENT = "#console-edit-thinking-modal"
    BINDINGS = [("escape", "request_safe_cancel", "Cancel")]

    def __init__(self, *, text: str) -> None:
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        with Vertical(id="console-edit-thinking-modal"):
            yield Static("Edit Thinking", classes="console-modal-header")
            yield Static(
                "Editing this thinking block. The answer, its provenance, "
                "and replay encoding stay intact.",
                id="console-edit-thinking-context",
                markup=False,
            )
            yield _EditMessageTextArea(self._text, id="console-edit-thinking-body")
            yield Static("", id="console-edit-thinking-error", markup=False)
            with Horizontal(id="console-edit-thinking-actions"):
                yield Button("Cancel", id="console-edit-thinking-cancel")
                yield Button(
                    "Save", id="console-edit-thinking-save", variant="primary"
                )

    def on_mount(self, event: events.Mount) -> None:  # type: ignore[override]
        # Same stale-key clock domain guard as the message edit modal
        # (TASK-360): keys pressed before this modal appeared never reach
        # the textarea.
        self._opened_at = event.time
        area = self.query_one("#console-edit-thinking-body", _EditMessageTextArea)
        area.opened_at = event.time
        area.focus()

    @on(Button.Pressed, "#console-edit-thinking-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    @on(Button.Pressed, "#console-edit-thinking-save")
    def _save(self, event: Button.Pressed) -> None:
        event.stop()
        edited_text = self.query_one("#console-edit-thinking-body", TextArea).text
        if not edited_text.strip():
            self.query_one("#console-edit-thinking-error", Static).update(
                "Thinking text cannot be blank."
            )
            return
        self.dismiss(ConsoleThinkingEditResult(text=edited_text))
