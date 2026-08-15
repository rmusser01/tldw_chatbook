"""Ephemeral Console side-chat modal (selection menu phase 2).

Answers questions about selected transcript text without touching the chat
store: the reply lives only in this modal, streamed through
:class:`~tldw_chatbook.Chat.console_side_chat.ConsoleSideChatService` in a
non-exclusive ``console-side-chat`` worker (never the session run group).

Two entry modes:

- ``auto_send_prompt`` (More Details): the rendered prompt is sent
  automatically on mount.
- ``None`` (Ask in Side Chat): the quote is shown read-only and the user
  types a freeform prompt.

Escape / backdrop / Close cancel any in-flight stream and dismiss (safe
modal contract); Stop and Retry ride the same worker choreography as the
prompt-improvement flow.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from loguru import logger
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea

from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_side_chat import (
    ConsoleSideChatService,
    SideChatOutcome,
    cap_reply_buffer,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

_CONTENT = "#console-side-chat-modal"
_IDENTITY = "#console-side-chat-identity"
_QUOTE = "#console-side-chat-quote"
_PROMPT = "#console-side-chat-prompt"
_REPLY = "#console-side-chat-reply"
_STATUS = "#console-side-chat-status"
_SEND = "#console-side-chat-send"
_STOP = "#console-side-chat-stop"
_RETRY = "#console-side-chat-retry"
_CLOSE = "#console-side-chat-close"

# Bare widget ids (Textual identifiers must not include the ``#``).
_ID_MODAL = "console-side-chat-modal"
_ID_IDENTITY = "console-side-chat-identity"
_ID_QUOTE = "console-side-chat-quote"
_ID_PROMPT = "console-side-chat-prompt"
_ID_REPLY = "console-side-chat-reply"
_ID_STATUS = "console-side-chat-status"
_ID_SEND = "console-side-chat-send"
_ID_STOP = "console-side-chat-stop"
_ID_RETRY = "console-side-chat-retry"
_ID_CLOSE = "console-side-chat-close"

SIDE_CHAT_WORKER_GROUP = "console-side-chat"


class ConsoleSideChatModal(SafeModalDismissMixin, ModalScreen[None]):
    """Ephemeral side chat about a quoted transcript selection."""

    DEFAULT_CSS = """
    ConsoleSideChatModal {
        align: center middle;
    }

    #console-side-chat-modal {
        width: 92;
        height: 30;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #console-side-chat-identity {
        height: auto;
        color: gray;
    }

    #console-side-chat-quote {
        height: 5;
        border: round gray;
        padding: 0 1;
        margin: 1 0;
        color: $text-muted;
    }

    #console-side-chat-prompt {
        height: 3;
        margin: 0 0 1 0;
    }

    #console-side-chat-reply-scroll {
        height: 1fr;
        min-height: 3;
        margin: 0 0 1 0;
    }

    #console-side-chat-status {
        height: auto;
        min-height: 1;
        color: $text-muted;
    }

    #console-side-chat-actions {
        height: 3;
        min-height: 3;
        margin: 1 0 0 0;
        align-horizontal: right;
    }

    #console-side-chat-close,
    #console-side-chat-retry,
    #console-side-chat-send,
    #console-side-chat-stop {
        width: 10;
        min-width: 10;
        height: 3;
        min-height: 3;
    }
    """

    SAFE_MODAL_CONTENT = _CONTENT
    BINDINGS = (("escape", "request_safe_cancel", "Cancel"),)

    def __init__(
        self,
        *,
        service: ConsoleSideChatService,
        provider_selection: ConsoleProviderSelection | None,
        sidechat_model: str,
        quote: str,
        auto_send_prompt: str | None,
        callback: Callable[[SideChatOutcome], object] | None = None,
    ) -> None:
        super().__init__()
        self._service = service
        self._provider_selection = provider_selection
        self._sidechat_model = sidechat_model
        self._quote = quote
        self._auto_send_prompt = auto_send_prompt
        self._callback = callback
        self._sidechat_worker = None
        self._request_counter = 0
        self._active_request_id: int | None = None
        self._last_prompt: str | None = None
        self._reply_text = ""

    def compose(self) -> ComposeResult:
        with Vertical(id=_ID_MODAL):
            yield Static("Side Chat", classes="console-modal-header")
            yield Static(
                self._requested_identity(),
                id=_ID_IDENTITY,
                markup=False,
            )
            yield Static(self._quote, id=_ID_QUOTE, markup=False)
            yield TextArea("", id=_ID_PROMPT)
            with VerticalScroll(id="console-side-chat-reply-scroll"):
                yield Static("", id=_ID_REPLY, markup=False)
            yield Static("", id=_ID_STATUS, markup=False)
            with Horizontal(id="console-side-chat-actions"):
                yield Button("Send", id=_ID_SEND, variant="primary")
                yield Button("Stop", id=_ID_STOP)
                yield Button("Retry", id=_ID_RETRY)
                yield Button("Close", id=_ID_CLOSE)

    # Textual composes MRO message handlers, so this event-shaped handler
    # runs alongside (not instead of) the mixin's opener-focus capture.
    def on_mount(self, event: events.Mount) -> None:  # type: ignore[override]
        del event
        auto_mode = self._auto_send_prompt is not None
        try:
            self.query_one(_PROMPT, TextArea).display = not auto_mode
        except NoMatches:
            pass
        self._set_action_visible(_SEND, not auto_mode)
        self._set_action_visible(_STOP, False)
        self._set_action_visible(_RETRY, False)
        if auto_mode:
            self._start_sidechat(self._auto_send_prompt or "")
        else:
            self.query_one(_PROMPT, TextArea).focus()

    def on_unmount(self) -> None:
        # The mixin's own on_unmount (opener-ref release) also runs via the
        # composed MRO dispatch.
        self._active_request_id = None
        worker = self._sidechat_worker
        if worker is not None and not worker.is_finished:
            worker.cancel()

    # ------------------------------------------------------------------
    # Worker choreography (mirrors the prompt-improvement flow)
    # ------------------------------------------------------------------

    def _next_request_id(self) -> int:
        self._request_counter += 1
        return self._request_counter

    def _start_sidechat(self, prompt: str) -> None:
        self._last_prompt = prompt
        self._reply_text = ""
        self._update_reply("")
        request_id = self._next_request_id()
        # Deliberately weaker than the prompts-modal choreography: a
        # superseded worker is NOT cancelled here. Only Send/Retry can call
        # this while a stream is live, and _set_streaming hides both, so the
        # race needs a second press to land before the display update —
        # unreachable in practice. If it ever fires, the stale worker is
        # wasted background work only; its deltas/outcome are dropped by the
        # request-id fence below and in _finish_sidechat.
        self._active_request_id = request_id
        self._set_streaming(True)
        self._set_status("Streaming…")
        self._sidechat_worker = self.run_worker(
            self._run_sidechat(request_id, prompt),
            exclusive=False,
            group=SIDE_CHAT_WORKER_GROUP,
        )

    async def _run_sidechat(self, request_id: int, prompt: str) -> None:
        try:
            async for item in self._service.run(
                selection_quote=self._quote,
                prompt=prompt,
                provider_selection=self._provider_selection,
                sidechat_model=self._sidechat_model,
            ):
                if self._active_request_id != request_id:
                    return  # a newer request (or dismissal) owns the UI now
                if isinstance(item, SideChatOutcome):
                    self._finish_sidechat(request_id, item)
                    return
                if isinstance(item, str):
                    self._append_delta(item)
        except asyncio.CancelledError:
            # The service yields its cancelled outcome before re-raising;
            # the outcome was handled above, so let cancellation finish.
            raise
        except Exception:  # noqa: BLE001 - worker guard must surface any failure inline
            logger.exception("Console side-chat worker failed unexpectedly")
            if self._active_request_id == request_id:
                self._active_request_id = None
                self._set_streaming(False)
                self._set_status(
                    "The side chat could not run. Check the provider and retry."
                )
                self._show_retry_actions()

    def _finish_sidechat(self, request_id: int, outcome: SideChatOutcome) -> None:
        if self._active_request_id != request_id:
            return
        self._active_request_id = None
        self._set_streaming(False)
        # Resolution arrives with the outcome; until then the header keeps
        # the requested identity.
        if outcome.provider or outcome.model:
            self._update_identity(
                self._identity_text(outcome.provider, outcome.model)
            )
        self._update_reply(cap_reply_buffer(outcome.text))
        if outcome.status == "complete":
            self._set_status("Complete.")
            self._set_action_visible(_SEND, self._auto_send_prompt is None)
        elif outcome.status == "cancelled":
            self._set_status("Cancelled.")
            self._show_retry_actions()
        else:
            error = outcome.error or "the provider reported an error"
            self._set_status(f"Provider error: {error}")
            self._show_retry_actions()
        if self._callback is not None:
            try:
                self._callback(outcome)
            except Exception:  # noqa: BLE001 - a failing callback must not break the modal
                logger.exception("Console side-chat callback failed")

    def _stop_streaming(self) -> None:
        worker = self._sidechat_worker
        if self._active_request_id is None or worker is None:
            return
        self._set_status("Cancelling…")
        # Keep _active_request_id so the cancelled outcome the service yields
        # on the way out still lands in the reply area.
        worker.cancel()

    # ------------------------------------------------------------------
    # Dismissal contract
    # ------------------------------------------------------------------

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        await self.run_cancel_effect_once(self._cancel_sidechat_worker)
        self.dismiss_safe_once(None)

    async def _cancel_sidechat_worker(self) -> None:
        self._active_request_id = None  # dismissing: in-flight outcomes stale
        worker = self._sidechat_worker
        if worker is not None and not worker.is_finished:
            worker.cancel()

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    @on(Button.Pressed, _SEND)
    def _send(self, event: Button.Pressed) -> None:
        event.stop()
        # Send always submits the CURRENT TextArea content (Ask mode only).
        prompt = self.query_one(_PROMPT, TextArea).text
        if not prompt.strip():
            self._set_status("Type a question first.")
            return
        self._start_sidechat(prompt)

    @on(Button.Pressed, _STOP)
    def _stop(self, event: Button.Pressed) -> None:
        event.stop()
        self._stop_streaming()

    @on(Button.Pressed, _RETRY)
    def _retry(self, event: Button.Pressed) -> None:
        event.stop()
        # Retry re-sends the last prompt UNCHANGED — it deliberately does
        # not read the TextArea; edited text goes through Send instead.
        prompt = self._last_prompt or self._auto_send_prompt
        if prompt is None:
            prompt = self.query_one(_PROMPT, TextArea).text
        if not prompt.strip():
            self._set_status("Type a question first.")
            return
        self._start_sidechat(prompt)

    @on(Button.Pressed, _CLOSE)
    async def _close(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _set_status(self, message: str) -> None:
        try:
            self.query_one(_STATUS, Static).update(message)
        except NoMatches:
            return

    def _set_action_visible(self, selector: str, visible: bool) -> None:
        try:
            self.query_one(selector, Button).display = visible
        except NoMatches:
            return

    def _show_retry_actions(self) -> None:
        """Actions after a cancelled/errored stream.

        Retry always re-sends the last prompt unchanged (``_last_prompt``).
        In Ask mode the prompt TextArea is still visible and editable, so
        Send must come back alongside Retry — otherwise an edited prompt
        could only be silently re-sent unchanged. Send always reads the
        TextArea. In auto mode there is no prompt area, so Retry alone.
        """
        self._set_action_visible(_RETRY, True)
        self._set_action_visible(_SEND, self._auto_send_prompt is None)

    def _set_streaming(self, active: bool) -> None:
        self._set_action_visible(_STOP, active)
        if active:
            self._set_action_visible(_SEND, False)
            self._set_action_visible(_RETRY, False)

    def _append_delta(self, delta: str) -> None:
        # Cap the in-memory accumulator too, not just the rendered copy: a
        # multi-gigabyte stream would otherwise grow ``_reply_text``
        # without bound between renders (T4 review).
        self._reply_text = cap_reply_buffer(self._reply_text + delta)
        self._update_reply(cap_reply_buffer(self._reply_text))

    def _update_reply(self, text: str) -> None:
        try:
            self.query_one(_REPLY, Static).update(text)
        except NoMatches:
            return

    def _update_identity(self, text: str) -> None:
        try:
            self.query_one(_IDENTITY, Static).update(text)
        except NoMatches:
            return

    # ------------------------------------------------------------------
    # Requested provider·model identity
    # ------------------------------------------------------------------

    def _requested_identity(self) -> str:
        configured = (self._sidechat_model or "").strip()
        if configured and "/" in configured:
            return configured
        provider = str(getattr(self._provider_selection, "provider", "") or "")
        model = configured or str(
            getattr(self._provider_selection, "explicit_model", "") or ""
        )
        return self._identity_text(provider, model)

    @staticmethod
    def _identity_text(provider: str, model: str) -> str:
        if provider and model:
            return f"{provider}·{model}"
        return provider or model or "session model"
