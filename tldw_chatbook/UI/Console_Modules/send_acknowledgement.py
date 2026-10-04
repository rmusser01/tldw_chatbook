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
Send control derive from. It is painted on the screen pump, and the unchanged
send is handed to the app pump only after a refresh has laid that frame out
(live: on screen 89-97 ms after Enter, hand-off at +58-68 ms). Nothing here
writes the store, the runtime or the durable turn; the row is released when
the store's own echo lands, when the dispatch admits no turn, or when the
runtime's custody of the admitted turn ends (a refusal before the echo).

Imported on the first Enter only, so it adds nothing to the ADR-097 boot
census; boot-time readers go through ``getattr(screen, ACK_ATTRIBUTE)``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
import contextlib
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

#: Screen attribute holding the lazily created acknowledgement.
ACK_ATTRIBUTE = "_console_send_ack"
#: Run chip / hidden mode-bar copy while a send is acknowledged.
SENDING_RUN_COPY = "Sending…"
#: Frame-length waits the dispatch may spend on the acknowledgement's layout.
_PAINT_HOPS = 6
_PAINT_HOP_SECONDS = 1 / 60


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

    def pending_row_id(self, session_id: str | None) -> str | None:
        pending = self._pending.get(session_id)
        return pending.row.id if pending is not None else None

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

    def custody_callback(self, session_id: str) -> Callable[[bool], None] | None:
        """Runtime terminal callback that releases this session's row."""
        pending = self._pending.get(session_id)
        if pending is None:
            return None
        return partial(self._custody_ended, pending.token)

    def mark_admitted(self, session_id: str) -> None:
        if (pending := self._pending.get(session_id)) is not None:
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
    return bool(getattr(screen, "_closing", False) or getattr(screen, "_closed", False))


def _request_resync(screen: Any) -> None:
    """Repaint every surface the released row touched, through one sync."""
    if not _torn_down(screen):
        screen.call_later(_resync, screen)


def _resync(screen: Any) -> None:
    if _torn_down(screen):
        return
    screen._last_native_transcript_refresh_key = None
    if screen._console_sync_in_progress:
        screen._console_sync_requested = True
        return
    screen.run_worker(
        screen._sync_native_console_chat_ui(), exclusive=True, group="console-sync"
    )


def _acknowledgeable(screen: Any, stash: Any, session_id: str) -> bool:
    """Only an idle session's plain text draft is certain to become a turn.

    Slash and raw commands and empty drafts take other paths; a live run, a
    question card or a queue turns Enter into Queue or an answer, each of
    which already has its own visible surface.
    """
    text = getattr(stash, "text", "") or ""
    if not text.strip() or text.lstrip().startswith(("/", "!")):
        return False
    controller = getattr(screen, "_console_chat_controller", None)
    if controller is None:
        return False
    activity = controller.activity_for(session_id)
    return not (
        activity.accepted_live_turn
        or activity.occupies_slot
        or activity.queued_count
        or controller.run_state_for(session_id).is_stop_allowed
    )


def schedule_acknowledged_send(
    screen: Any,
    stash: Any,
    session_id: str,
    send: Callable[[], Awaitable[bool]],
) -> None:
    """Paint the acknowledgement, then run ``send`` on the app pump after it.

    Args:
        screen: The Console ``ChatScreen`` that captured the Enter.
        stash: The draft captured at the keypress.
        session_id: The session the draft belongs to.
        send: The unchanged visible-action send, bound to its Enter token.
    """
    ack = acknowledgement_for(screen)
    token = (
        ack.begin(session_id, stash.text, _session_message_ids(screen, session_id))
        if _acknowledgeable(screen, stash, session_id)
        else None
    )

    async def observed_send() -> bool:
        try:
            return await send()
        finally:
            ack.dispatch_finished(token)

    def dispatch() -> None:
        screen.app.call_later(observed_send)

    if token is None or not screen.call_later(_paint_then, screen, dispatch):
        dispatch()


async def _paint_then(screen: Any, dispatch: Callable[[], None]) -> None:
    """Paint on this pump; the dispatch runs after the frame reaches the screen."""
    try:
        await paint_acknowledgement(screen)
        # Let the row's and chip's own pumps post their Layout requests ahead
        # of the hand-off, so the next refresh normally lays both out.
        for _ in range(3):
            await asyncio.sleep(0)
    except Exception as exc:  # noqa: BLE001 -- the send must never depend on its paint
        # Type only: an exception's text can carry the draft or session ids.
        logger.warning(
            "Console send acknowledgement paint failed (exception_type={})",
            type(exc).__name__,
        )
    finally:
        if not screen.call_after_refresh(_dispatch_once_laid_out, screen, dispatch, 0):
            dispatch()


def _dispatch_once_laid_out(
    screen: Any, dispatch: Callable[[], None], hops: int
) -> None:
    """Dispatch once a refresh has laid the row and the Run chip out.

    The mounted row and the shown chip post their own Layout requests from
    their own pumps after the paint. Dispatching after the first refresh left
    the old arrangement on screen for the whole blocking admission (mounted
    harness: header, tab dot and Send updated; row and chip missing). A
    ``call_after_refresh`` hop cannot wait for them: with nothing dirty yet
    the screen runs it at once, and six such hops took 0.3 ms without the
    loop ever yielding. Each further hop is one frame-length timer instead,
    bounded by ``_PAINT_HOPS``.
    """
    if hops < _PAINT_HOPS and not _acknowledgement_laid_out(screen):
        try:
            screen.set_timer(
                _PAINT_HOP_SECONDS,
                partial(_dispatch_once_laid_out, screen, dispatch, hops + 1),
            )
            return
        except Exception:  # noqa: BLE001 -- a closing screen still sends
            pass
    dispatch()


def _acknowledgement_laid_out(screen: Any) -> bool:
    ack = getattr(screen, ACK_ATTRIBUTE, None)
    session_id = screen._console_chat_store.active_session_id
    row_id = ack.pending_row_id(session_id) if ack is not None else None
    if row_id is None:
        return True
    try:
        if not screen.query_one(f"#console-message-{row_id}").region.area:
            return False
        chips = screen.query_one("#console-status-chips")
        chip = chips.query_one("#console-run-chip")
    except NoMatches:
        return False
    return bool(chip.region.area) or bool(getattr(chips, "collapsed", False))


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
