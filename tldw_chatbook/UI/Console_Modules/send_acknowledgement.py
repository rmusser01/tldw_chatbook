"""Acknowledge an Enter send on screen before its admission work (TASK-33620.5).

Measured on dev 7d155170dc (live 160x45, Anthropic haiku): after Enter nothing
changed for 0.46-0.95 s; then the composer cleared under an idle "Ready"
header, and the user row, tab dot and Run chip appeared 1.7-3.1 s after Enter.
The send's synchronous admission (``_build_console_turn_execution_context``,
0.25-1.0 s on the UI pump) ran before anything was painted, and the row and
status surfaces reached the screen only through the 0.2 s whole-screen sync
tick. A user who retyped into that gap sent a duplicate they never saw.

This module owns a view-only acknowledgement per Enter: a USER row marked
"Sending…" plus the run-active facts the header, tab marker, Run chip and
Send control derive from. It is pushed to those widgets on the screen pump,
then the unchanged send is handed to the app pump at once. No wait is added
before the send: the frame reaches the terminal on the screen's next refresh,
which lands while the send runs its own awaited steps ahead of admission
(mounted harness: frame on screen 36-46 ms after Enter, admission at 80-110
ms). Nothing here writes the store, the runtime or the durable turn; the row
is released when the store's own echo lands, when the dispatch admits no
turn, or when the runtime's custody of the admitted turn ends (a refusal
before the echo).

Imported on the first Enter only, so it adds nothing to the ADR-097 boot
census; boot-time readers go through ``getattr(screen, ACK_ATTRIBUTE)``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
import contextlib
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import partial
from typing import Any
from uuid import uuid4

from loguru import logger
from textual.css.query import NoMatches

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleRunMarker,
)
from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    blocked_turn_reason,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import (
    classify_console_raw_draft,
)

#: Screen attribute holding the lazily created acknowledgement.
ACK_ATTRIBUTE = "_console_send_ack"
#: Run chip / hidden mode-bar copy while a send is acknowledged.
SENDING_RUN_COPY = "Sending…"
#: The acknowledged Enter whose send is running in this task. A worker the
#: send starts (a hook review's continuation) copies it with the context, so
#: an admission binds to the Enter that dispatched it, never a newer one.
_DISPATCHING: ContextVar[object | None] = ContextVar(
    "console_send_ack_dispatching", default=None
)


@dataclass
class _PendingSend:
    token: object
    session_id: str
    row: ConsoleChatMessage
    baseline_ids: frozenset[str]
    admitted: bool = False


class ConsoleSendAcknowledgement:
    """Each session's Enter send acknowledged ahead of its store echo."""

    def __init__(self, on_release: Callable[[], None]) -> None:
        self._pending: dict[str, _PendingSend] = {}
        self._on_release = on_release

    def begin(
        self, session_id: str, text: str, current_ids: Iterable[str]
    ) -> object | None:
        """Acknowledge a send; ``None`` while this session has one pending."""
        if session_id in self._pending:
            return None
        token = object()
        row = ConsoleChatMessage(
            role=ConsoleMessageRole.USER,
            content=text,
            id=f"console-send-ack-{uuid4().hex}",
            status="pending",
        )
        self._pending[session_id] = _PendingSend(
            token, session_id, row, frozenset(current_ids)
        )
        return token

    def active_for(self, session_id: str | None) -> bool:
        return session_id in self._pending

    def run_copy(self, session_id: str | None) -> str:
        return SENDING_RUN_COPY if self.active_for(session_id) else ""

    def overlay_run_markers(
        self, markers: dict[str, ConsoleRunMarker] | None
    ) -> dict[str, ConsoleRunMarker] | None:
        """Mark each acknowledged session's tab running until its run starts."""
        if not self._pending or markers is None:
            return markers
        overlaid = dict(markers)
        for session_id in self._pending:
            if overlaid.get(session_id) is not ConsoleRunMarker.NEEDS_APPROVAL:
                overlaid[session_id] = ConsoleRunMarker.RUNNING
        return overlaid

    def project(
        self, session_id: str | None, messages: Iterable[ConsoleChatMessage]
    ) -> list[ConsoleChatMessage]:
        """Append the pending row until the store's own echo replaces it."""
        rows = list(messages)
        pending = self._pending.get(session_id)
        if pending is None:
            return rows
        if any(
            row.role is ConsoleMessageRole.USER and row.id not in pending.baseline_ids
            for row in rows
        ):
            # The echo is in this very projection: no resync is owed.
            del self._pending[pending.session_id]
            return rows
        return [*rows, pending.row]

    @contextlib.contextmanager
    def dispatching(self, token: object | None) -> Iterator[None]:
        """Bind runtime admissions made inside this block to ``token``."""
        reset = _DISPATCHING.set(token)
        try:
            yield
        finally:
            _DISPATCHING.reset(reset)

    def custody_callback(self, session_id: str) -> Callable[[bool], None] | None:
        """Runtime terminal callback that releases the dispatching send's row."""
        pending = self._dispatching_send(session_id)
        if pending is None:
            return None
        return partial(self._custody_ended, pending.token)

    def mark_admitted(self, session_id: str) -> None:
        if (pending := self._dispatching_send(session_id)) is not None:
            pending.admitted = True

    def dispatch_finished(self, token: object | None) -> None:
        """Release a row whose dispatch admitted no runtime turn."""
        pending = self._by_token(token)
        if pending is not None and not pending.admitted:
            self.release(token)

    def release(self, token: object) -> None:
        pending = self._by_token(token)
        if pending is None:
            return
        del self._pending[pending.session_id]
        self._on_release()

    def _dispatching_send(self, session_id: str) -> _PendingSend | None:
        """This session's pending send, only while its own dispatch runs here."""
        pending = self._pending.get(session_id)
        if pending is None or pending.token is not _DISPATCHING.get():
            return None
        return pending

    def _by_token(self, token: object | None) -> _PendingSend | None:
        return next(
            (item for item in self._pending.values() if item.token is token), None
        )

    def _custody_ended(self, token: object, _accepted: bool) -> None:
        self.release(token)


def acknowledgement_for(screen: Any) -> ConsoleSendAcknowledgement:
    """Return the screen's acknowledgement, creating it on first use."""
    ack = getattr(screen, ACK_ATTRIBUTE, None)
    if ack is None:
        ack = ConsoleSendAcknowledgement(partial(_request_resync, screen))
        setattr(screen, ACK_ATTRIBUTE, ack)
    return ack


def _torn_down(screen: Any) -> bool:
    from tldw_chatbook.UI.Screens.chat_screen import _console_screen_is_torn_down

    return _console_screen_is_torn_down(screen)


def _request_resync(screen: Any) -> None:
    """Repaint every surface the released row touched, through one sync."""
    if not _torn_down(screen):
        screen.call_later(_resync, screen)


def _resync(screen: Any) -> None:
    if _torn_down(screen):
        return
    screen._last_native_transcript_refresh_key = None
    if screen._console_sync_in_progress:
        # The running sync re-arms itself for this request when it settles.
        screen._console_sync_requested = True
        return
    screen.run_worker(
        screen._sync_native_console_chat_ui(), exclusive=True, group="console-sync"
    )


def _acknowledged_text(screen: Any, stash: Any, session_id: str) -> str | None:
    """Return the text an idle session's chat draft is sent as, else ``None``.

    Slash commands, a typed ``! `` local command and empty drafts take other
    paths (an escaped ``\\! `` draft is chat, sent without its backslash; a
    pasted ``! `` is chat too). A live run, a question card or a queue turns
    Enter into Queue or an answer, each of which has its own visible surface;
    behind a Blocked turn (TASK-33621.2) the send is refused, never sent.
    """
    if stash is None:
        return None
    classified = classify_console_raw_draft(stash)
    text = classified.text
    if classified.kind == "raw" or not text.strip() or text.lstrip().startswith("/"):
        return None
    controller = getattr(screen, "_console_chat_controller", None)
    if controller is None:
        return None
    activity = controller.activity_for(session_id)
    if (
        activity.accepted_live_turn
        or activity.occupies_slot
        or activity.queued_count
        or controller.run_state_for(session_id).is_stop_allowed
        or blocked_turn_reason(controller)
    ):
        return None
    return text


def schedule_acknowledged_send(screen: Any, pending_send: Any) -> None:
    """Paint the acknowledgement, then run the Enter's send on the app pump.

    Args:
        screen: The Console ``ChatScreen`` that captured the Enter.
        pending_send: The keypress capture: session id, draft stash and the
            token the visible-action send consumes.
    """
    session_id = pending_send.session_id
    send = partial(
        screen._send_console_message_from_visible_action,
        pending_send_token=pending_send.token,
    )
    ack = acknowledgement_for(screen)
    text = _acknowledged_text(screen, pending_send.stash, session_id)
    token = (
        ack.begin(session_id, text, _session_message_ids(screen, session_id))
        if text is not None
        else None
    )

    async def observed_send() -> bool:
        try:
            with ack.dispatching(token):
                return await send()
        finally:
            ack.dispatch_finished(token)

    def dispatch() -> None:
        screen.app.call_later(observed_send)

    if token is None or not screen.call_later(_paint_then, screen, dispatch):
        dispatch()


async def _paint_then(screen: Any, dispatch: Callable[[], None]) -> None:
    """Push the acknowledgement on this pump, then hand the send off at once.

    Every acknowledged fact is on its widget before the send starts. The
    frame itself goes out on the screen's next refresh, during the send's
    own awaited steps before its synchronous admission; the mounted test
    reads the frame on screen at the instant admission starts. Waiting here
    for that refresh (TASK-33620.5's first cut: frame-length timer hops)
    delayed every reply by ~60 ms and no test could tell it apart from not
    waiting.
    """
    try:
        await paint_acknowledgement(screen)
    except Exception as exc:  # noqa: BLE001 -- the send must never depend on its paint
        # Type only: an exception's text can carry the draft or session ids.
        logger.warning(
            "Console send acknowledgement paint failed (exception_type={})",
            type(exc).__name__,
        )
    finally:
        dispatch()


async def paint_acknowledgement(screen: Any) -> None:
    """Push the acknowledged state straight to each surface it changes.

    Built from the screen's whole derivations (the full transcript sync, the
    Workbench state, the composer's readiness gating) this paint measured
    66-286 ms live; these targeted pushes measured 28-29 ms. They carry exactly
    what the next whole sync derives from the same acknowledgement, so that
    sync confirms them.
    """
    from tldw_chatbook.Chat.console_display_state import (
        QUEUE_REASON_PREPARING,
        SEND_LABEL_SENDING,
    )

    session_id = screen._console_chat_store.active_session_id
    widget = screen.query_one("#console-native-transcript")
    ack = acknowledgement_for(screen)
    rows = screen._change_review_projection.project(
        screen._message._native_console_messages()
    )
    widget.set_messages(ack.project(session_id, rows), session_id=session_id)
    screen._last_native_transcript_refresh_key = None
    # Every lookup happens before the first await: the screen can be torn
    # down while one is pending (textual worker contract).
    composer = screen._console_composer_or_none()
    if composer is not None:
        composer.show_send_acknowledged(SEND_LABEL_SENDING, QUEUE_REASON_PREPARING)
    with contextlib.suppress(NoMatches):
        screen.query_one("#console-status-chips").sync_run_chip(True, SENDING_RUN_COPY)
    with contextlib.suppress(NoMatches):
        header = screen.query_one("#console-workbench-header")
        if header.state is not None:
            # What `build_console_workbench_state` derives while run-active.
            header.sync_state(replace(header.state, status="running", status_label=""))
    await widget.refresh_messages()
    await screen._sync_console_native_session_tabs()


def _session_message_ids(screen: Any, session_id: str) -> tuple[str, ...]:
    store = getattr(screen, "_console_chat_store", None)
    if store is None:
        return ()
    try:
        return tuple(message.id for message in store.messages_for_session(session_id))
    except KeyError:
        return ()
