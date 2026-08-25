"""TASK-3070.12 Console realtime's 56/0/1 non-DOM ownership boundary.

This seam owns the realtime session and close-worker state plus explicit,
late-bound application edges. The 56 orchestration methods move here in the
next step; there are no framework delegates, and the one repaint method stays
with the presentation owner.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from textual.worker import Worker

from tldw_chatbook.Chat.console_realtime_loop import RealtimeLoopController


@dataclass
class ConsoleRealtimeSession:
    """Everything the realtime (V4) hands-free loop needs while it runs.

    Constructed once per loop entry (`ChatScreen._enter_console_realtime_
    loop`) and dropped on `ExitLoop` (`ChatScreen._release_console_realtime_
    state`) -- never reused across entries, exactly like its V3 sibling
    `ConsoleHandsFreeSession`, so every entry gets a clean FSM.

    Attributes:
        controller: The headless FSM driving the loop.
        console_session_id: The Console chat session this loop is bound to,
            captured at entry. Every continuity row is written to THIS
            session, never to `store.active_session_id` re-read later --
            a tab switch mid-conversation must not scatter half a spoken
            exchange across two transcripts (the same discipline V3's
            `pending_session_id` enforces for its own send).
        buddy_generation: Monotonic app-owned loop generation used only
            to fence trusted Buddy lifecycle state from replaced loops.
        idle_timeout_seconds: The configured idle ceiling, kept here so the
            exit toast can name it without re-reading config at exit time.
        tap: The `RealtimeMicTap` streaming microphone PCM into the session.
        session: The live `RealtimeSession`, or None before the first
            connect completes and between a drop and its reconnect.
        sink: The `StreamingPcmSink` playing the CURRENT reply's audio, or
            None between replies.
        audio_queue: The `asyncio.Queue` feeding this reply's `pump` task;
            a `None` item is the end-of-reply sentinel that closes the
            async iterator.
        pump_worker: The worker running `pump(sink, aiter)` for this reply.
        tick_timer: The `set_interval(0.1, ...)` handle driving
            `controller.tick(now)` (the idle ceiling) and the chip repaint.
        connect_attempt: Monotonic per-loop counter, incremented for every
            connect (first and each reconnect). Callbacks are bound to the
            attempt that created them, so a superseded session's late
            events are dropped instead of driving the FSM (see
            `_console_realtime_marshal`).
        ready: True once the provider acknowledged the handshake and the
            tap was flushed; an adopted transcript arriving before that is
            held in `pending_text_turn` rather than enqueued into a session
            that cannot send it yet. Also the discriminator for what a
            close/error MEANS (see `_on_console_realtime_closed`): before
            it, a refused connect; after it, a transport drop.
        connect_returned_at: Monotonic stamp of the moment `connect()`
            returned for the outstanding attempt, or None when no attempt
            is waiting on `on_ready`. Drives the ready deadline in
            `_tick_console_realtime` -- the backstop for a no-ready path
            that arrives as nothing at all.
        mic_gated: The gate value last synced to `tap.set_gated(...)` --
            the wiring's record of rule 7, and what tests assert against
            (the tap's own flag is private).
        fed_bytes: Bytes of reply audio handed to the sink queue for the
            CURRENT reply. Drives `played_ms`; reset per reply.
        audio_failed_for_reply: True once this reply's audio sink failed to
            open -- every later delta of the SAME reply is then dropped
            without another attempt. Reset at the next reply start.
        audio_unavailable_notified: True once the user has been told, in
            THIS loop entry, that reply audio is unavailable. One toast per
            loop, not one per reply.
        reply_token: Monotonic per-reply counter. A reply's playback
            completion carries the token it started with, so a completion
            that lands after the next reply began is dropped instead of
            reporting that one finished.
        generation_done: True once `response.done` arrived for the current
            reply. Half of the rendezvous below.
        playback_pending: True while this reply's audio is still being fed
            or played. The other half: whichever of these two finishes
            LAST is what tells the FSM the reply is over -- see
            `_on_console_realtime_reply_done`.
        barged: True once the user cut this reply short. Mirrors Task 2's
            "a cancelled response fires no reply-done": the aborted pump's
            completion must report nothing.
        barge_trigger: Which input drove the barge-in currently being
            handled -- `"keypress"` or `"speech"`. Recorded here because
            the `SilenceSpeech` intent is shared by both and carries no
            trigger of its own, and "which one fired" is the first
            question any barge-in report raises.
        user_row_id: The transcript row created at turn-commit, waiting for
            its input transcript to land.
        assistant_row_id: The current reply's transcript row, or None
            between replies (closed by `_finish_console_realtime_reply_row`).
        last_reply_row_id: The most recent reply's row, NOT cleared when
            that reply closes -- usage arrives from the same provider event
            that ended the reply, so it always needs the row that just
            stopped being current.
        pending_text_turn: An adopted pipeline capture's transcript waiting
            for `on_ready` (see `ready`).
        adopt_capture: True while a live pipeline capture is being stopped
            so its transcript can become this loop's first turn.
        failure_text: Why the last connect attempt failed, in user-facing
            words -- consumed by the fallback toast.
        transcript_dirty: Set by every continuity write; consumed by the
            0.1 s tick, which is what actually repaints the transcript (a
            per-delta resync would be one full UI rebuild per audio
            transcript chunk).
    """

    controller: RealtimeLoopController
    console_session_id: str
    idle_timeout_seconds: float
    buddy_generation: int = 0
    tap: Any = None
    session: Any = None
    sink: Any = None
    audio_queue: Any = None
    pump_worker: Any = None
    tick_timer: Any = None
    connect_attempt: int = 0
    ready: bool = False
    connect_returned_at: float | None = None
    reply_token: int = 0
    generation_done: bool = False
    playback_pending: bool = False
    barged: bool = False
    barge_trigger: str = "unknown"
    mic_gated: bool = False
    fed_bytes: int = 0
    user_row_id: str | None = None
    assistant_row_id: str | None = None
    last_reply_row_id: str | None = None
    audio_failed_for_reply: bool = False
    audio_unavailable_notified: bool = False
    pending_text_turn: str | None = None
    adopt_capture: bool = False
    failure_text: str = ""
    transcript_dirty: bool = False


class ConsoleRealtimeController:
    """Own realtime orchestration state behind explicit late-bound edges."""

    def __init__(
        self,
        *,
        ensure_session_settings: Callable[[], Any],
        chat_store_accessor: Callable[[], Any],
        runtime_accessor: Callable[[], Any],
        dictation_state_accessor: Callable[[], str],
        request_dictation_stop: Callable[[], None],
        pipeline_blocker: Callable[[], str | None],
        enter_pipeline_loop: Callable[[bool], None],
        recorder_factory_accessor: Callable[[], Any],
        provider_session_factory_accessor: Callable[[], Any],
        sink_factory_accessor: Callable[[], Any],
        notify: Callable[..., Any],
        ui_thread_id_accessor: Callable[[], int],
        event_loop_accessor: Callable[[], asyncio.AbstractEventLoop | None],
        set_interval: Callable[..., Any],
        run_worker: Callable[..., Worker[Any]],
        defer_native_sync: Callable[[], bool],
        repaint_chip: Callable[[], None],
        restore_voice_chip: Callable[[], None],
    ) -> None:
        self._ensure_session_settings = ensure_session_settings
        self._chat_store_accessor = chat_store_accessor
        self._runtime_accessor = runtime_accessor
        self._dictation_state_accessor = dictation_state_accessor
        self._request_dictation_stop = request_dictation_stop
        self._pipeline_blocker = pipeline_blocker
        self._enter_pipeline_loop = enter_pipeline_loop
        self._recorder_factory_accessor = recorder_factory_accessor
        self._provider_session_factory_accessor = provider_session_factory_accessor
        self._sink_factory_accessor = sink_factory_accessor
        self._notify = notify
        self._ui_thread_id_accessor = ui_thread_id_accessor
        self._event_loop_accessor = event_loop_accessor
        self._set_interval = set_interval
        self._run_worker = run_worker
        self._defer_native_sync = defer_native_sync
        self._repaint_chip = repaint_chip
        self._restore_voice_chip = restore_voice_chip
        self.session: ConsoleRealtimeSession | None = None
        self.close_worker: Worker | None = None
