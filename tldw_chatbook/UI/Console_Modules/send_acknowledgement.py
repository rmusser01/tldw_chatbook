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
and the unchanged send is handed to the app pump once a refresh has laid
that frame out (live, 80x24 / 160x45 / 235x52, first, warm and new-tab
sends: the whole acknowledgement on screen 31-78 ms after Enter). The cost
is a later start for the send: 34-53 ms after Enter in the mounted harness
against 1-5 ms on dev, and admission 10-20 ms later at the median. Nothing
here writes the store, the runtime or the durable turn; the row is released
when the store's own echo lands (the first USER row the session gains after
its turn is handed to the runtime: only that turn writes the echo), when the
dispatch admits no turn, or when the runtime's custody of the admitted turn
ends (a refusal before the echo). The paint is for the Enter's own tab: if
another tab is shown by the time it runs, nothing is painted (the send is
refused for its changed tab).

TASK-33620.15: the Send button and the Workbench's send now start here too
(``request_visible_send``), and the send runs as its own task: awaited from
the app pump's callback, as it was, every await of it held key delivery.
A send request made while one runs is replayed when it settles, as the busy
app pump used to replay it, and only in the chat it was made in.

Imported on the first send only, so it adds nothing to the ADR-097 boot
census; boot-time readers go through ``getattr(screen, ACK_ATTRIBUTE)``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Iterator
import contextlib
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
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
#: Screen attribute holding the visible sends in flight (TASK-33620.15).
FLIGHT_ATTRIBUTE = "_console_send_flight"
#: Run chip / hidden mode-bar copy while a send is acknowledged.
SENDING_RUN_COPY = "Sending…"
#: Frame-length waits the hand-off may spend on the acknowledgement's layout.
_PAINT_HOPS = 6
_PAINT_HOP_SECONDS = 1 / 60
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
    #: The session's message ids when its turn was handed to the runtime;
    #: ``None`` until then. Only that turn writes the echo, so the echo is
    #: the first USER row outside this set.
    handoff_ids: frozenset[str] | None = None
    admitted: bool = False


class ConsoleSendAcknowledgement:
    """Each session's Enter send acknowledged ahead of its store echo."""

    def __init__(
        self,
        on_release: Callable[[], None],
        message_ids: Callable[[str], Iterable[str]],
    ) -> None:
        """Create an acknowledgement with no send pending.

        Args:
            on_release: Called after a row is released outside a projection,
                so the surfaces it changed are repainted.
            message_ids: Reads a session's current message ids; called once
                per acknowledged send, as its turn is handed to the runtime.
        """
        self._pending: dict[str, _PendingSend] = {}
        self._on_release = on_release
        self._message_ids = message_ids

    def begin(self, session_id: str, text: str) -> object | None:
        """Acknowledge one Enter send in ``session_id``.

        Args:
            session_id: The tab the Enter was pressed in.
            text: The draft as it will be sent, shown in the pending row.

        Returns:
            The send's token (bind its dispatch with :meth:`dispatching`), or
            ``None`` while this session already has a send pending.
        """
        if session_id in self._pending:
            return None
        token = object()
        row = ConsoleChatMessage(
            role=ConsoleMessageRole.USER,
            content=text,
            id=f"console-send-ack-{uuid4().hex}",
            status="pending",
        )
        self._pending[session_id] = _PendingSend(token, session_id, row)
        return token

    def active_for(self, session_id: str | None) -> bool:
        """Return whether ``session_id`` has an acknowledged send pending.

        Args:
            session_id: The session to ask about; ``None`` is never pending.

        Returns:
            True from :meth:`begin` until the send's row is released.
        """
        return session_id in self._pending

    def pending_row_id(self, session_id: str | None) -> str | None:
        """Return the id of ``session_id``'s pending "Sending…" row.

        Args:
            session_id: The session to ask about.

        Returns:
            The transcript row id, or ``None`` when no send is pending there.
        """
        pending = self._pending.get(session_id)
        return pending.row.id if pending is not None else None

    def run_copy(self, session_id: str | None) -> str:
        """Return the Run chip copy for ``session_id``'s pending send.

        Args:
            session_id: The session whose chip is shown.

        Returns:
            ``SENDING_RUN_COPY`` while a send is pending there, else ``""``.
        """
        return SENDING_RUN_COPY if self.active_for(session_id) else ""

    def overlay_run_markers(
        self, markers: dict[str, ConsoleRunMarker] | None
    ) -> dict[str, ConsoleRunMarker] | None:
        """Mark each acknowledged session's tab running until its run starts.

        Args:
            markers: The controller's run marker per session, or ``None``.

        Returns:
            A copy with each pending session marked running (a pending
            approval keeps its own marker), or ``markers`` itself when no
            send is pending.
        """
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
        """Append the pending row until the store's own echo replaces it.

        The echo is the first USER row ``session_id`` gains after the send's
        turn was handed to the runtime. Before that hand-off no row can be
        it, so a USER row another action adds (an Edit & resend sibling)
        leaves the pending row in place.

        Args:
            session_id: The session the rows belong to.
            messages: That session's transcript rows, in order.

        Returns:
            The rows, with the pending row last until its echo is among them.
        """
        rows = list(messages)
        pending = self._pending.get(session_id)
        if pending is None:
            return rows
        handoff_ids = pending.handoff_ids
        if (
            pending.admitted
            and handoff_ids is not None
            and any(
                row.role is ConsoleMessageRole.USER and row.id not in handoff_ids
                for row in rows
            )
        ):
            # The echo is in this very projection: no resync is owed.
            del self._pending[pending.session_id]
            return rows
        return [*rows, pending.row]

    @contextlib.contextmanager
    def dispatching(self, token: object | None) -> Iterator[None]:
        """Bind runtime admissions made inside this block to ``token``.

        Args:
            token: The token :meth:`begin` returned, or ``None`` for a send
                that was not acknowledged.

        Yields:
            Nothing; the binding is undone when the block exits.
        """
        reset = _DISPATCHING.set(token)
        try:
            yield
        finally:
            _DISPATCHING.reset(reset)

    def custody_callback(self, session_id: str) -> Callable[[bool], None] | None:
        """Hand the dispatching send's turn over; return its release callback.

        Called as the turn is handed to the runtime, before it can write the
        echo: the session's message ids now are what the echo is told apart
        from.

        Args:
            session_id: The session whose turn is being admitted.

        Returns:
            A runtime terminal callback that releases this send's row, or
            ``None`` when no acknowledged send is dispatching here.
        """
        pending = self._dispatching_send(session_id)
        if pending is None:
            return None
        pending.handoff_ids = frozenset(self._message_ids(session_id))
        return partial(self._custody_ended, pending.token)

    def mark_admitted(self, session_id: str) -> None:
        """Record that the dispatching send's turn was admitted.

        From here the row waits for its echo or the turn's custody to end,
        not for its dispatch to return.

        Args:
            session_id: The session whose turn the runtime accepted.
        """
        if (pending := self._dispatching_send(session_id)) is not None:
            pending.admitted = True

    def dispatch_finished(self, token: object | None) -> None:
        """Release a row whose dispatch admitted no runtime turn.

        Args:
            token: The finished dispatch's token (``None`` is ignored).
        """
        pending = self._by_token(token)
        if pending is not None and not pending.admitted:
            self.release(token)

    def release(self, token: object) -> None:
        """Drop ``token``'s pending row and request a repaint.

        Args:
            token: The send to release; a stale token releases nothing.
        """
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
        ack = ConsoleSendAcknowledgement(
            partial(_request_resync, screen), partial(_session_message_ids, screen)
        )
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
    token = ack.begin(session_id, text) if text is not None else None

    async def observed_send() -> bool:
        try:
            with ack.dispatching(token):
                return await send()
        finally:
            ack.dispatch_finished(token)

    def dispatch() -> None:
        screen.app.call_later(_start_send, screen, observed_send)

    if token is None or not screen.call_later(
        _paint_then, screen, session_id, dispatch
    ):
        dispatch()


def request_visible_send(
    screen: Any,
    *,
    guard: Callable[[], bool] | None = None,
    made_in: str | None = None,
) -> None:
    """Capture the visible draft now and send it, acknowledged (TASK-33620.15).

    Enter, the Send button and the Workbench's send all start here, so all
    three get the same "Sending…" acknowledgement (TASK-33620.5 gave it to
    Enter only) and the same send: a task the app pump starts and never
    awaits, so its admission never holds key delivery.

    Args:
        screen: The Console ``ChatScreen``.
        guard: Enter's check that the Send action is available, run once the
            draft is captured; ``False`` releases the capture unsent.
        made_in: Set on a deferred request's replay: the chat it was made in.
    """
    from tldw_chatbook.UI.Screens.chat_screen import _ConsolePendingSend

    if screen._console_pending_send is not None:
        # A send keypress is already scheduled on the app pump; a second
        # Enter in that window must not enqueue it twice.
        return
    session_id = screen._console_visible_send_session_id()
    flight = _flight(screen)
    if flight.tasks:
        # The app pump used to hold this request until the running send's
        # admission returned; replay the latest one then (one request, as the
        # second Enter of a double press would have found an empty draft).
        flight.deferred = partial(
            request_visible_send, screen, guard=guard, made_in=session_id
        )
        return
    if made_in is not None and made_in != session_id:
        # Its chat was left meanwhile (the held pump ran it there first): it
        # must never send the draft now on screen. That chat keeps its draft.
        return
    if session_id is None:
        screen.app_instance.notify("Console send is unavailable.", severity="error")
        return
    composer = screen._console_composer_or_none()
    # TASK-340: capture the payload now so printable keys handled before the
    # scheduled callback belong to the next draft.
    stash = composer.capture_draft_for_send() if composer is not None else None
    pending_send = _ConsolePendingSend(session_id, stash, object())
    screen._console_pending_send = pending_send
    if stash is not None:
        try:
            screen._ensure_console_chat_store().set_session_draft(
                session_id, stash.text
            )
        except KeyError:
            screen._console_pending_send = None
            return
    if guard is not None and not guard():
        screen._console_pending_send = None
        return
    schedule_acknowledged_send(screen, pending_send)


@dataclass
class _SendFlight:
    """The screen's visible sends still running, and one deferred request."""

    tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    deferred: Callable[[], None] | None = None


def _flight(screen: Any) -> _SendFlight:
    flight = getattr(screen, FLIGHT_ATTRIBUTE, None)
    if flight is None:
        flight = _SendFlight()
        setattr(screen, FLIGHT_ATTRIBUTE, flight)
    return flight


def _start_send(screen: Any, send: Callable[[], Awaitable[bool]]) -> None:
    """Run ``send`` as its own task; the app pump returns at its first await.

    Awaited from this callback instead, every await of the send -- the hook
    snapshot, the turn authority read off the UI pump -- kept the app pump
    from delivering keys, so moving work to a thread alone unblocked nothing.
    An exception still reaches the app as it did from the pump.
    """
    flight = _flight(screen)
    task = asyncio.get_running_loop().create_task(send())
    flight.tasks.add(task)
    task.add_done_callback(partial(_send_settled, screen, flight))


def _send_settled(screen: Any, flight: _SendFlight, task: asyncio.Task[Any]) -> None:
    flight.tasks.discard(task)
    if not task.cancelled() and (error := task.exception()) is not None:
        screen.app.call_later(_raise, error)
    if not flight.tasks and (deferred := flight.deferred) is not None:
        flight.deferred = None
        if not _torn_down(screen):
            screen.call_later(deferred)


def _raise(error: BaseException) -> None:
    raise error


async def _paint_then(
    screen: Any, session_id: str, dispatch: Callable[[], None]
) -> None:
    """Push the acknowledgement on this pump; send once a frame shows it."""
    try:
        await paint_acknowledgement(screen, session_id)
    except Exception as exc:  # noqa: BLE001 -- the send must never depend on its paint
        # Type only: an exception's text can carry the draft or session ids.
        logger.warning(
            "Console send acknowledgement paint failed (exception_type={})",
            type(exc).__name__,
        )
    finally:
        if not screen.call_after_refresh(
            _dispatch_once_laid_out, screen, session_id, dispatch, 0
        ):
            dispatch()


def _dispatch_once_laid_out(
    screen: Any, session_id: str, dispatch: Callable[[], None], hops: int
) -> None:
    """Dispatch once a refresh has laid the row and the Run chip out.

    The send must not start before the frame is out: its first synchronous
    stretch can be the admission itself, and nothing is drawn during it. The
    mounted test reads the frame written before the send starts; handing off
    straight after the paint is red there (nothing drawn yet), and live it
    left the tab dot off the frame for the whole admission block (80x24,
    first and warm sends). The mounted row and shown chip update from their
    own pumps after the first refresh, so after one ``call_after_refresh``
    each further hop is a frame-length timer until both have a region,
    bounded by ``_PAINT_HOPS``. The harness cannot tell a single hop from
    these hops; live it could (a new-tab send drew its dot only after
    admission with one hop). The tabs are relabelled before the row mounts:
    relabelled last, a warm 80x24 send still lost the dot with these hops.
    Final build, live: 9 of 9 sends had the whole acknowledgement on screen
    before admission.
    """
    if hops < _PAINT_HOPS and not _acknowledgement_laid_out(screen, session_id):
        try:
            screen.set_timer(
                _PAINT_HOP_SECONDS,
                partial(
                    _dispatch_once_laid_out, screen, session_id, dispatch, hops + 1
                ),
            )
            return
        except Exception:  # noqa: BLE001 -- a closing screen still sends
            pass
    dispatch()


def _acknowledgement_laid_out(screen: Any, session_id: str) -> bool:
    if screen._console_chat_store.active_session_id != session_id:
        return True  # Another tab is shown: nothing of this send to wait for.
    ack = getattr(screen, ACK_ATTRIBUTE, None)
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


async def paint_acknowledgement(screen: Any, session_id: str) -> None:
    """Push the acknowledged state straight to each surface it changes.

    Built from the screen's whole derivations (the full transcript sync, the
    Workbench state, the composer's readiness gating) this paint measured
    66-286 ms live; these targeted pushes measured 28-29 ms. They carry exactly
    what the next whole sync derives from the same acknowledgement, so that
    sync confirms them.

    Args:
        screen: The Console ``ChatScreen``.
        session_id: The tab the Enter was pressed in. When another tab is
            shown by now (a tab press landed first), nothing is painted: the
            composer, Run chip and header belong to the shown tab, whose own
            sync derives its state, and the send is refused for the change.
    """
    from tldw_chatbook.Chat.console_display_state import (
        QUEUE_REASON_PREPARING,
        SEND_LABEL_SENDING,
    )

    if screen._console_chat_store.active_session_id != session_id:
        return
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
    # Tabs before the row: a relabelled tab is drawn a frame after the change,
    # and the hand-off waits only for the mounted row (live warm send, tabs
    # last: the dot first appeared after admission).
    await screen._sync_console_native_session_tabs()
    await widget.refresh_messages()


def _session_message_ids(screen: Any, session_id: str) -> tuple[str, ...]:
    store = getattr(screen, "_console_chat_store", None)
    if store is None:
        return ()
    try:
        return tuple(message.id for message in store.messages_for_session(session_id))
    except KeyError:
        return ()
