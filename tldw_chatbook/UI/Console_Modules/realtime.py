"""TASK-3070.12 Console realtime's 56/0/1 non-DOM ownership boundary.

This seam owns the realtime session and close-worker state plus explicit,
late-bound application edges. All 56 orchestration methods live here; there
are no framework delegates, and the one repaint method stays with the
presentation owner.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import logging
import re
import threading
import time
from typing import Any

from loguru import logger
from textual.worker import Worker

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_hands_free import ExitLoop, ModeChanged, SilenceSpeech
from tldw_chatbook.Chat.console_realtime_loop import RealtimeLoopController
from tldw_chatbook.Chat.console_voice_input import (
    acoustic_barge_in_enabled,
    realtime_idle_timeout_seconds,
    realtime_model,
    realtime_provider,
    realtime_turn_detection,
    realtime_vad_silence_ms,
    realtime_vad_threshold,
    realtime_voice,
)
from tldw_chatbook.Chat.message_metadata import MessageMetadata
from tldw_chatbook.Chat.provider_usage import ProviderUsage, as_seconds
from tldw_chatbook.LLM_Calls.realtime import RealtimeCallbacks, RealtimeSessionConfig
from tldw_chatbook.Utils.persistent_diagnostics import (
    persist_event,
    safe_metadata_token,
)
from tldw_chatbook.config import get_api_key


CONSOLE_REALTIME_SUPPORTED_PROVIDER = "openai"
CONSOLE_REALTIME_TRANSCRIPTION_MODEL = "whisper-1"
CONSOLE_REALTIME_CONNECT_TIMEOUT_SECONDS = 8.0
CONSOLE_REALTIME_READY_TIMEOUT_SECONDS = 8.0
CONSOLE_REALTIME_FAILURE_TEXT_MAX_CHARS = 120
_CONSOLE_REALTIME_CODE_RE = re.compile(r"\(code=([A-Za-z0-9_.\- ]{1,64})\)")
CONSOLE_REALTIME_ERROR_CATEGORY_ALIASES: dict[str, str] = {
    "invalid_api_key": "invalid_credentials",
    "missing_api_key": "missing_credentials",
}
_CONSOLE_REALTIME_SECRET_RE = re.compile(r"[A-Za-z0-9_\-]{24,}")
CONSOLE_REALTIME_SAMPLE_RATE = 24000
CONSOLE_REALTIME_BYTES_PER_SECOND = CONSOLE_REALTIME_SAMPLE_RATE * 2
CONSOLE_REALTIME_SEED_TURNS = 20
CONSOLE_REALTIME_SEED_CHARS = 8000
CONSOLE_REALTIME_INTERRUPTED_MARKER = " ⏹ interrupted"
CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER = "(no speech detected)"
CONSOLE_REALTIME_ENGINE = "realtime"
CONSOLE_REALTIME_CHIP_MESSAGES: dict[str, str] = {
    "connecting": "realtime · connecting…",
    "live": "realtime · listening",
    "thinking": "realtime · thinking…",
    "speaking": "realtime · speaking",
    "reconnecting": "realtime · reconnecting…",
}
CONSOLE_REALTIME_UNSUPPORTED_PROVIDER_TEMPLATE = (
    "Realtime voice provider '{provider}' is not supported. Only "
    "'{supported}' is implemented; hands-free did not start."
)
CONSOLE_REALTIME_MIC_FAILED_MESSAGE = "the microphone could not be opened"
CONSOLE_REALTIME_NO_API_KEY_MESSAGE = (
    f"no {CONSOLE_REALTIME_SUPPORTED_PROVIDER.title()} API key is configured"
)
CONSOLE_REALTIME_AUDIO_UNAVAILABLE_MESSAGE = (
    "Realtime reply audio is unavailable (no output device); the reply "
    "transcript still appears in the conversation."
)
CONSOLE_REALTIME_CONNECT_TIMEOUT_MESSAGE = "the connection timed out after {seconds:g}s"
CONSOLE_REALTIME_HANDSHAKE_INCOMPLETE_MESSAGE = (
    "the handshake never completed after {seconds:g}s"
)
CONSOLE_REALTIME_UNSPECIFIED_FAILURE_MESSAGE = (
    "the realtime session could not be opened"
)
CONSOLE_REALTIME_FALLBACK_TEMPLATE = (
    "Realtime voice unavailable ({reason}); using the pipeline hands-free loop instead."
)
CONSOLE_REALTIME_NO_LOOP_TEMPLATE = (
    "Hands-free unavailable. Realtime failed ({reason}); the pipeline loop "
    "is not usable either ({pipeline_reason})."
)
CONSOLE_REALTIME_RECONNECTING_MESSAGE = "Realtime reconnecting…"
CONSOLE_REALTIME_RECONNECTED_MESSAGE = "Realtime reconnected"
CONSOLE_REALTIME_EXIT_CONNECTION_LOST_MESSAGE = "Hands-free ended: connection lost"
CONSOLE_REALTIME_EXIT_IDLE_TEMPLATE = "Hands-free ended: idle for {minutes:g} minutes"


@dataclass
class ConsoleRealtimeSession:
    """Everything the realtime (V4) hands-free loop needs while it runs.

    Constructed once per loop entry (`ConsoleRealtimeController._enter_console_
    realtime_loop`) and dropped on `ExitLoop` (`ConsoleRealtimeController._release_
    console_realtime_state`) -- never reused across entries, exactly like its V3 sibling
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

    def _enter_console_realtime_loop(self, *, capture_live: bool) -> None:
        """Start the realtime hands-free loop.

        Order matters here and is load-bearing:

        1. Refuse an unsupported provider BEFORE anything is opened -- the
           config reader does not validate it (see
           `CONSOLE_REALTIME_SUPPORTED_PROVIDER`).
        2. Enter the FSM, which paints `connecting…` immediately, so the
           several seconds a handshake can take never look like a hang.
        3. Open the MICROPHONE, before the connect is even started. The tap
           buffers everything it captures until `mark_ready()`, so a user
           who starts talking the instant the chip appears keeps their
           first words instead of losing them to the handshake window.
        4. Only then connect, bounded by
           `CONSOLE_REALTIME_CONNECT_TIMEOUT_SECONDS`.

        Args:
            capture_live: True when a one-shot pipeline capture is already
                open (the key binding pressed while recording, or a spoken
                "hands free" mid-capture). That capture is stopped and
                transcribed through the existing V2 path, and its
                transcript becomes this loop's first turn -- see
                `_console_realtime_adopt_transcript`.
        """
        if self.session is not None:
            return
        provider = str(realtime_provider() or "").strip().lower()
        if provider != CONSOLE_REALTIME_SUPPORTED_PROVIDER:
            self._notify(
                CONSOLE_REALTIME_UNSUPPORTED_PROVIDER_TEMPLATE.format(
                    provider=realtime_provider(),
                    supported=CONSOLE_REALTIME_SUPPORTED_PROVIDER,
                ),
                severity="warning",
            )
            return

        # Bind the Console session ONCE, here: every continuity row this
        # loop writes goes to this id, never to a re-read `active_session_
        # id` (see `ConsoleRealtimeSession.console_session_id`).
        self._ensure_session_settings()
        store = self._chat_store_accessor()
        console_session_id = store.active_session_id
        if not console_session_id:
            logger.debug("Console realtime loop refused: no active Console session")
            return
        idle_timeout = realtime_idle_timeout_seconds()
        buddy_generation = (
            self._runtime_accessor().persona_buddy_sink.next_voice_generation(
                console_session_id
            )
        )
        if buddy_generation is None:
            return
        controller = RealtimeLoopController(
            self._handle_console_realtime_intent,
            acoustic_barge_in=acoustic_barge_in_enabled(),
            idle_timeout_seconds=idle_timeout,
        )
        session = ConsoleRealtimeSession(
            controller=controller,
            console_session_id=console_session_id,
            idle_timeout_seconds=idle_timeout,
            buddy_generation=buddy_generation,
        )
        self.session = session
        session.tick_timer = self._set_interval(0.1, self._tick_console_realtime)
        self._persist_console_realtime_event(
            "realtime_entry",
            operation="entry",
            provider=provider,
            model=str(realtime_model()),
        )
        controller.enter()

        if not self._start_console_realtime_tap(session):
            self._console_realtime_connect_failed(
                session,
                session.connect_attempt,
                RuntimeError(CONSOLE_REALTIME_MIC_FAILED_MESSAGE),
            )
            return

        if capture_live and self._dictation_state_accessor() == "recording":
            session.adopt_capture = True
            self._request_dictation_stop()

        self._start_console_realtime_connect(session)

    def _start_console_realtime_tap(self, session: ConsoleRealtimeSession) -> bool:
        """Open the microphone for `session`. Returns True on success.

        The tap is constructed with a lazily-imported `RealtimeMicTap`: its
        module reaches `Audio/recording_service.py` (and therefore NumPy
        plus the optional capture backends) at import time, which must not
        be paid at app start by every Console mount that never speaks.

        `recorder_factory` is left as None in production; the app-level
        `console_realtime_recorder_factory` seam exists so tests exercise
        the REAL tap (its buffering/ordering guarantees are what rule 3
        depends on) against a fake recorder rather than a real device.
        """
        from ...Audio.realtime_mic_tap import RealtimeMicTap

        recorder_factory = self._recorder_factory_accessor()
        tap = RealtimeMicTap(
            lambda frames: self._on_console_realtime_frames(session, frames),
            sample_rate=CONSOLE_REALTIME_SAMPLE_RATE,
            recorder_factory=recorder_factory if callable(recorder_factory) else None,
        )
        session.tap = tap
        try:
            started = bool(tap.start())
        except Exception:  # noqa: BLE001 - a device failure is a fallback, not a crash
            logger.opt(exception=True).warning(
                "Console realtime: microphone tap failed to start"
            )
            started = False
        return started

    def _on_console_realtime_frames(
        self, session: ConsoleRealtimeSession, frames: bytes
    ) -> None:
        """Forward one captured PCM chunk to the provider session.

        Runs on the RECORDER's own background thread (see
        `RealtimeMicTap`'s module docstring), which is exactly the call
        pattern `OpenAIRealtimeSession.append_audio` documents itself
        thread-safe for -- it marshals onto its own loop internally, so
        nothing is marshalled here. Both reads below are plain attribute
        loads, safe from any thread, and a stale session (the loop exited
        while a frame was in flight) is dropped rather than resurrected.
        """
        if self.session is not session:
            return
        provider_session = session.session
        if provider_session is None:
            return
        try:
            provider_session.append_audio(frames)
        except Exception:  # noqa: BLE001 - never kill the recorder thread
            logger.opt(exception=True).debug(
                "Console realtime: append_audio failed; dropping this chunk"
            )

    def _console_realtime_instructions(self) -> str | None:
        """The active session's system prompt, as realtime `instructions`.

        A realtime session has no per-request message list to carry a
        system prompt in -- instructions are session-level -- so the
        Console's own system prompt has to be handed over at handshake and
        re-handed on every reconnect, or the model silently loses its
        persona the moment the transport blips.
        """
        try:
            settings = self._ensure_session_settings()
        except Exception:  # noqa: BLE001 - a settings failure must not block voice
            logger.opt(exception=True).debug(
                "Console realtime: could not read the session system prompt"
            )
            return None
        prompt = str(getattr(settings, "system_prompt", "") or "").strip()
        return prompt or None

    def _console_realtime_seed_items(
        self, console_session_id: str
    ) -> list[tuple[str, str]]:
        """Build the conversation seed for a fresh (or reconnected) session.

        Newest-first selection under BOTH budgets
        (`CONSOLE_REALTIME_SEED_TURNS`, `CONSOLE_REALTIME_SEED_CHARS`),
        then reversed back into transcript order: what a returning session
        most needs is the recent thread, and an unbounded replay of a long
        Console conversation is billed context on every reconnect.

        Only user/assistant rows with real text are replayed -- tool
        markers would seed noise the user never said. A row whose
        transcript came back empty (`transcript_status == "empty"`,
        task-2391) is excluded the same way even though its content is no
        longer blank: that content is now the empty-transcript placeholder
        (`CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER`), UI chrome written
        so the row could persist at all, not something the user said --
        replaying it would teach the model the user typed that literal
        phrase.

        An over-budget message is SKIPPED, not treated as the end of the
        walk (fix round 1, F6): stopping there meant one long newest reply
        -- routine, a realtime reply is a monologue -- shipped ZERO history
        on reconnect, silently amnesiac exactly when continuity matters
        most. Skipping keeps every older turn that still fits.
        """
        store = self._chat_store_accessor()
        try:
            messages = store.messages_for_session(console_session_id)
        except KeyError:
            return []
        selected: list[tuple[str, str]] = []
        used_chars = 0
        for message in reversed(messages):
            if message.role not in (
                ConsoleMessageRole.USER,
                ConsoleMessageRole.ASSISTANT,
            ):
                continue
            metadata = message.metadata
            if metadata is not None and metadata.transcript_status == "empty":
                continue
            text = self._console_realtime_seed_text(message)
            if not text:
                continue
            if used_chars + len(text) > CONSOLE_REALTIME_SEED_CHARS:
                continue
            selected.append((message.role.value, text))
            used_chars += len(text)
            if len(selected) >= CONSOLE_REALTIME_SEED_TURNS:
                break
        selected.reverse()
        return selected

    @staticmethod
    def _console_realtime_seed_text(message: ConsoleChatMessage) -> str:
        """The model-facing text of one prior turn, without our chrome.

        The interrupted marker is OUR chrome for the human reader (final
        review M4): replaying it into the model's context on every reseed
        would teach it that "⏹ interrupted" is part of how the assistant
        speaks. So it is removed here -- as a TRAILING marker, always, on
        every row, with no condition attached.

        Trimming a suffix rather than matching the text anywhere is what
        makes that safe: `_finish_console_realtime_reply_row` only ever
        APPENDS the marker (via `append_stream_chunk`), so a suffix trim
        removes every marker this app has written while leaving alone the
        same characters occurring in a turn's actual words. A user who
        types "the docs say ⏹ interrupted means cut off" gets their
        sentence seeded intact; the earlier global replace ate it.

        Deliberately NOT gated on `metadata.interrupted` (task-2364, review
        round 1). Only the realtime loop stamps metadata onto rows, so
        every ordinary typed turn -- past, present and future -- arrives
        here with `metadata is None`: a gate reading "no metadata means a
        legacy interrupted reply" would mangle live user text forever, and
        a gate reading the flag alone would leak chrome whenever the marker
        append succeeded but the metadata write was swallowed (they are
        separate, separately-swallowed calls). `interrupted` remains the
        SEMANTIC record -- what exports, summaries and later readers
        consult; removing chrome this code appended is a mechanical undo,
        not an inference, so it needs no fact to consult.

        Where the two disagree, that is logged rather than acted on: it is
        the only place the divergence is observable, and each direction
        means something different (a marker without the flag is a stale
        marker; a flag without the marker is a LOST one, so the reader
        never saw the reply was cut).

        Args:
            message: A transcript row from the loop's Console session.

        Returns:
            The row's text with a trailing interruption marker removed,
            stripped.
        """
        raw = str(message.content or "")
        trimmed = raw.removesuffix(CONSOLE_REALTIME_INTERRUPTED_MARKER)
        metadata = message.metadata
        if metadata is not None:
            if trimmed != raw and not metadata.interrupted:
                logger.debug(
                    "Console realtime: seeded a row carrying the interrupted "
                    "marker without the flag; the metadata write was likely "
                    "swallowed: op=realtime_seed_text"
                )
            elif trimmed == raw and metadata.interrupted:
                logger.debug(
                    "Console realtime: seeded a row flagged interrupted with no "
                    "marker in its text; the marker append was likely "
                    "swallowed, so the reader never saw the cut: "
                    "op=realtime_seed_text"
                )
        return trimmed.strip()

    def _console_realtime_row_metadata(
        self,
        *,
        model: str,
        interrupted: bool = False,
        transcript_status: str = "",
    ) -> MessageMetadata:
        """Build the provenance record every realtime row carries.

        The V4 spec puts engine/provider/model provenance on the row
        itself; before task-2364 it could only ride the attached usage and
        a visible marker (spec "Turn metadata deferred").

        Args:
            model: Model this row is attributed to -- the realtime model
                for a reply, the transcription model for a user row, which
                is exactly how each row's usage is attributed too.
            interrupted: Whether the row's generation was cut short.
            transcript_status: One of ``MessageMetadata``'s closed
                vocabulary; ``""`` for rows that are not transcriptions.

        Returns:
            The metadata record to store on the row.
        """
        return MessageMetadata(
            engine=CONSOLE_REALTIME_ENGINE,
            provider=CONSOLE_REALTIME_SUPPORTED_PROVIDER,
            model=model,
            interrupted=interrupted,
            transcript_status=transcript_status,
        )

    def _build_console_realtime_session(
        self, config: RealtimeSessionConfig, callbacks: RealtimeCallbacks
    ) -> Any:
        """Construct the provider session, honoring the test seam.

        `console_realtime_session_factory` mirrors `console_provider_
        gateway_factory`'s getattr idiom exactly. The real session is
        imported inside this method, not at module scope: it owns a
        WebSocket transport, and a Console mount that never opens a
        realtime loop must not pay for it.
        """
        factory = self._provider_session_factory_accessor()
        if callable(factory):
            return factory(config, callbacks)
        from ...LLM_Calls.realtime.openai_session import OpenAIRealtimeSession

        return OpenAIRealtimeSession(config, callbacks)

    def _console_realtime_api_key(self) -> str:
        """The configured API key for the realtime provider, or `""`.

        Never raises and never logs the key itself.
        """
        try:
            return str(get_api_key(CONSOLE_REALTIME_SUPPORTED_PROVIDER) or "")
        except Exception:  # noqa: BLE001 - config trouble is a connect failure
            logger.opt(exception=True).debug(
                "Console realtime: could not resolve the provider API key"
            )
            return ""

    def _build_console_realtime_callbacks(
        self, session: ConsoleRealtimeSession, attempt: int
    ) -> RealtimeCallbacks:
        """Wire this connect attempt's callbacks onto the screen.

        Every callback is bound to `attempt`, so a session superseded by a
        reconnect can never drive the FSM afterward (see
        `_console_realtime_marshal`), and every one of them is marshalled
        rather than called inline -- they arrive on the session's own
        asyncio task.
        """

        def _route(handler: Callable[..., None]) -> Callable[..., None]:
            def _fire(*args: Any) -> None:
                self._console_realtime_marshal(handler, session, attempt, *args)

            return _fire

        return RealtimeCallbacks(
            on_ready=_route(self._on_console_realtime_ready),
            on_turn_committed=_route(self._on_console_realtime_turn_committed),
            on_input_transcript=_route(self._on_console_realtime_input_transcript),
            on_reply_started=_route(self._on_console_realtime_reply_started),
            on_output_transcript_delta=_route(
                self._on_console_realtime_output_transcript_delta
            ),
            on_audio_delta=_route(self._on_console_realtime_audio_delta),
            on_first_audio=_route(self._on_console_realtime_first_audio),
            on_reply_done=_route(self._on_console_realtime_reply_done),
            on_usage=_route(self._on_console_realtime_usage),
            on_transcription_usage=_route(
                self._on_console_realtime_transcription_usage
            ),
            on_speech_started=_route(self._on_console_realtime_speech_started),
            on_error=_route(self._on_console_realtime_error),
            on_closed=_route(self._on_console_realtime_closed),
        )

    def _console_realtime_marshal(
        self,
        handler: Callable[..., None],
        session: ConsoleRealtimeSession,
        attempt: int,
        *args: Any,
    ) -> None:
        """Run `handler(session, *args)` on the app's own thread.

        Realtime callbacks fire from the session's asyncio task. In
        production that task runs on the app's event loop (the connect
        worker is dispatched there), so the fast path below is a direct
        call -- but the contract does not promise it, and a foreign-thread
        callback must never touch widgets. `call_soon_threadsafe` is used
        rather than `App.call_from_thread` on purpose: `call_from_thread`
        BLOCKS its caller until the callback completes, and blocking a
        provider's receive loop on the UI thread would stall inbound audio
        for the whole conversation.

        The staleness check runs at DELIVERY time, not schedule time: a
        callback queued just before a reconnect must be judged against the
        state it will actually land in.
        """

        def _run() -> None:
            if self.session is not session:
                return
            if session.connect_attempt != attempt:
                return
            try:
                handler(session, *args)
            except Exception:  # noqa: BLE001 - a wiring fault must not kill the loop
                logger.opt(exception=True).warning(
                    "Console realtime: callback handler failed; dropping it"
                )

        if threading.get_ident() == self._ui_thread_id_accessor():
            _run()
            return
        loop = self._event_loop_accessor()
        if loop is None:
            logger.debug(
                "Console realtime: no app loop to marshal onto; dropping callback"
            )
            return
        try:
            loop.call_soon_threadsafe(_run)
        except Exception:  # noqa: BLE001 - a closing loop is not an error here
            logger.opt(exception=True).debug(
                "Console realtime: marshal onto the app loop failed"
            )

    def _start_console_realtime_connect(self, session: ConsoleRealtimeSession) -> None:
        """Dispatch one connect attempt (first connect or reconnect).

        ONE code path serves both, which is exactly what
        `RealtimeLoopController.on_connect_failed`'s docstring expects: it
        routes a `connecting` failure to `connect-failed` and a
        `reconnecting` failure to the same give-up exit a second transport
        drop takes.
        """
        session.connect_attempt += 1
        # No credential, no connect (fix round 1): dispatching one anyway
        # would spend the connect timeout to come back with whatever 401
        # text the provider chose, and the fallback toast would quote THAT
        # instead of the one thing the user can act on. Same
        # blocker-shaped check as `_console_pipeline_hands_free_blocker`,
        # routed through the SAME failure path so the fallback behaves
        # identically.
        if not self._console_realtime_api_key():
            self._console_realtime_connect_failed(
                session,
                session.connect_attempt,
                RuntimeError(CONSOLE_REALTIME_NO_API_KEY_MESSAGE),
            )
            return
        self._run_worker(
            self._connect_console_realtime(session, attempt=session.connect_attempt),
            exclusive=False,
            group="console-realtime-connect",
            exit_on_error=False,
        )

    async def _connect_console_realtime(
        self, session: ConsoleRealtimeSession, *, attempt: int
    ) -> None:
        """Build and connect one provider session, bounded by a timeout."""
        config = RealtimeSessionConfig(
            api_key=self._console_realtime_api_key(),
            model=realtime_model(),
            # `or None` rather than the raw value: an empty configured
            # voice means "use the provider default", which is what None
            # means on the wire -- sending `""` would ask for a voice named
            # nothing.
            voice=realtime_voice() or None,
            input_sample_rate=CONSOLE_REALTIME_SAMPLE_RATE,
            output_sample_rate=CONSOLE_REALTIME_SAMPLE_RATE,
            instructions=self._console_realtime_instructions(),
            turn_detection=realtime_turn_detection(),
            vad_threshold=realtime_vad_threshold(),
            vad_silence_ms=realtime_vad_silence_ms(),
            # Read per attempt, not captured at loop entry: a reconnect
            # that reverted to the provider's defaults would bring back
            # the fragmenting these settings exist to stop, halfway
            # through a conversation, with nothing to show for it.
        )
        callbacks = self._build_console_realtime_callbacks(session, attempt)
        try:
            provider_session = self._build_console_realtime_session(config, callbacks)
        except Exception as exc:  # noqa: BLE001 - reported, never raised at the user
            self._console_realtime_connect_failed(session, attempt, exc)
            return
        if self.session is not session or session.connect_attempt != attempt:
            # Superseded before we even connected (exit, or another
            # reconnect): release what was just built rather than leaking
            # a live transport nobody owns.
            await self._close_console_realtime_session(provider_session)
            return
        session.session = provider_session
        try:
            await asyncio.wait_for(
                provider_session.connect(),
                timeout=CONSOLE_REALTIME_CONNECT_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            await self._close_console_realtime_session(provider_session)
            self._console_realtime_connect_failed(
                session,
                attempt,
                TimeoutError(
                    CONSOLE_REALTIME_CONNECT_TIMEOUT_MESSAGE.format(
                        seconds=CONSOLE_REALTIME_CONNECT_TIMEOUT_SECONDS
                    )
                ),
            )
        except Exception as exc:  # noqa: BLE001 - every failure is a fallback
            await self._close_console_realtime_session(provider_session)
            self._console_realtime_connect_failed(session, attempt, exc)
            return
        if self.session is not session or session.connect_attempt != attempt:
            return
        # The transport is up, but the provider has NOT accepted the
        # session yet (`on_ready` is the acknowledgement). Arm the ready
        # deadline for that window -- see `CONSOLE_REALTIME_READY_TIMEOUT_
        # SECONDS`; a refusal usually arrives as a callback long before
        # this fires, and this exists for the case where nothing arrives
        # at all.
        session.connect_returned_at = time.monotonic()

    @staticmethod
    def _persist_console_realtime_event(event: str, **fields: Any) -> None:
        """Record one realtime lifecycle event to the persistent log.

        The persistent log admits ONLY `tldw_chatbook.diagnostics.*`
        records (`Utils/persistent_diagnostics.py`), so without this a
        realtime run left no durable trace at all -- the owner's
        stuck-at-connecting session had to be reconstructed from a
        screenshot. Same shape as the dictation-failure site above, for
        the same reason.

        Every field goes through the persistent schema, which is bounded
        tokens only: a provider's error prose (which quotes API keys)
        cannot be passed here even by accident. Failures to persist are
        swallowed -- diagnostics must never break the voice loop.
        """
        try:
            persist_event("realtime", event, **fields)
        except Exception:  # noqa: BLE001 - diagnostics never break the loop
            logger.opt(exception=True).debug(
                "Could not persist a realtime diagnostics event"
            )

    @staticmethod
    def _console_realtime_failure_token(text: str) -> str:
        """Reduce a sanitized failure to a bounded token for the log.

        Prefers the provider's own `(code=…)` -- the single most
        diagnostic word available -- and falls back to `unspecified`
        rather than forcing prose through `safe_metadata_token`, which
        would write a useless `invalid`.

        The alias table exists because the persistent schema REFUSES any
        token containing `api_key` (`_PRIVATE_TOKEN_MARKERS`): from the
        admission boundary's seat, "invalid_api_key" is indistinguishable
        from a leaked credential, and it is right to refuse it. So the
        credential-failure case -- the one that actually brought this
        logging into existence -- is recorded under a marker-free synonym
        instead of defeating the guard that protects the log.
        """
        match = _CONSOLE_REALTIME_CODE_RE.search(text or "")
        candidate = match.group(1).strip() if match else ""
        candidate = CONSOLE_REALTIME_ERROR_CATEGORY_ALIASES.get(candidate, candidate)
        token = safe_metadata_token(candidate) if candidate else "invalid"
        return "unspecified" if token == "invalid" else token

    @staticmethod
    def _sanitize_console_realtime_failure(raw: object) -> str:
        """Reduce a provider failure to something safe to show and log.

        Provider error text quotes credentials. OpenAI's own invalid-key
        message is literally `Incorrect API key provided: sk-proj-…` --
        so the raw string can never reach a toast, and (the discipline
        this codebase already keeps for `loguru`'s frame dumps) can never
        reach a log line either.

        Three steps, in order:
          1. Keep the code the session appended (`(code=invalid_api_key)`)
             -- provider vocabulary, never user material, and the single
             most useful token in the whole message.
          2. Keep only the LEADING clause, up to the first `:` or newline.
             That is where providers put the human summary and after which
             they put the offending value.
          3. Scrub any long unbroken token that survived anyway, and cap
             the length.

        Args:
            raw: An exception or reason string from the provider.

        Returns:
            Sanitized text, never empty.
        """
        text = str(raw or "").strip()
        if not text:
            return CONSOLE_REALTIME_UNSPECIFIED_FAILURE_MESSAGE
        code_match = _CONSOLE_REALTIME_CODE_RE.search(text)
        code = code_match.group(1).strip() if code_match else ""
        lead = text.splitlines()[0].split(":", 1)[0].strip()
        lead = _CONSOLE_REALTIME_SECRET_RE.sub("…", lead).strip()
        if code and code not in lead:
            lead = f"{lead} ({code})".strip() if lead else code
        if len(lead) > CONSOLE_REALTIME_FAILURE_TEXT_MAX_CHARS:
            lead = lead[: CONSOLE_REALTIME_FAILURE_TEXT_MAX_CHARS - 1].rstrip() + "…"
        return lead or CONSOLE_REALTIME_UNSPECIFIED_FAILURE_MESSAGE

    def _console_realtime_connect_failed(
        self, session: ConsoleRealtimeSession, attempt: int, exc: BaseException
    ) -> None:
        """Record why a connect attempt failed and tell the FSM.

        The FSM decides what that MEANS (a first-connect failure exits with
        `connect-failed`, which the exit handler turns into the loud
        fallback; a failed reconnect exits with `connection-lost`), so this
        never decides for it.

        The SINGLE choke point for every way a connect can fail -- a
        raising `connect()`, a timeout, a close or an error arriving before
        the handshake was acknowledged, or the ready deadline -- so
        sanitization happens here, once, and no caller can forget it.
        """
        if self.session is not session or session.connect_attempt != attempt:
            return
        session.connect_returned_at = None
        session.failure_text = self._sanitize_console_realtime_failure(
            str(exc) or type(exc).__name__
        )
        self._persist_console_realtime_event(
            "realtime_connect_failed",
            level=logging.ERROR,
            operation="connect",
            status="failed",
            exception_type=type(exc).__name__,
            error_category=self._console_realtime_failure_token(str(exc)),
            retry_count=max(attempt - 1, 0),
        )
        logger.warning(
            "Console realtime: connect attempt failed: "
            f"op=realtime_connect attempt={attempt} reason={session.failure_text!r}"
        )
        session.session = None
        session.controller.on_connect_failed()

    def _on_console_realtime_ready(self, session: ConsoleRealtimeSession) -> None:
        """`on_ready`: seed the session, release the buffered audio, go live.

        Seeding happens BEFORE `mark_ready()` on purpose: the tap flushes
        its pre-ready buffer synchronously into `append_audio`, and the
        provider must already hold the conversation history (and the
        instructions) when the user's first words arrive, not after them.

        Arriving here from `reconnecting` also closes the loop the
        "Realtime reconnecting…" toast opened (final review M6): without a
        matching success toast, a reconnect that WORKED is
        indistinguishable from one still in progress -- the chip returns
        to `listening` either way, and the user is left unsure whether to
        keep talking.
        """
        reconnected = session.controller.state == "reconnecting"
        provider_session = session.session
        if provider_session is not None:
            try:
                provider_session.send_seed(
                    self._console_realtime_seed_items(session.console_session_id),
                    self._console_realtime_instructions(),
                )
            except Exception:  # noqa: BLE001 - a seed failure is not fatal
                logger.opt(exception=True).warning(
                    "Console realtime: seeding the session failed"
                )
        tap = session.tap
        if tap is not None:
            try:
                tap.mark_ready()
            except Exception:  # noqa: BLE001
                logger.opt(exception=True).warning(
                    "Console realtime: flushing the mic tap failed"
                )
        session.ready = True
        session.connect_returned_at = None
        self._persist_console_realtime_event(
            "realtime_ready",
            operation="ready",
            status="reconnected" if reconnected else "connected",
            retry_count=max(session.connect_attempt - 1, 0),
        )
        session.controller.on_session_ready()
        if reconnected:
            self._notify(CONSOLE_REALTIME_RECONNECTED_MESSAGE, severity="information")
        pending, session.pending_text_turn = session.pending_text_turn, None
        if pending:
            # An adopted capture whose transcript landed while the
            # handshake was still in flight (see
            # `_console_realtime_adopt_transcript`).
            self._send_console_realtime_text_turn(session, pending)

    def _on_console_realtime_turn_committed(
        self, session: ConsoleRealtimeSession
    ) -> None:
        """`on_turn_committed`: the provider closed the user's input turn.

        The transcript row is created HERE, empty, rather than when the
        transcript itself finally arrives: input transcription runs
        asynchronously and routinely lands AFTER the assistant has already
        started replying, so a row created on arrival would sit below the
        answer it asked for. Creating it at commit fixes its place in the
        transcript; `_on_console_realtime_input_transcript` fills it in.

        `phase` records the state this arrived IN, before the FSM sees it:
        `on_turn_committed` is a no-op outside `live`, so a commit landing
        in `thinking` is silently dropped -- which is exactly the shape of
        the owner's "I spoke and nothing came back" incident, and was
        invisible in the log.
        """
        self._persist_console_realtime_event(
            "realtime_turn_committed",
            operation="turn_committed",
            initiator="audio",
            phase=session.controller.state,
        )
        session.user_row_id = self._append_console_realtime_row(
            session,
            ConsoleMessageRole.USER,
            "",
            # The row is deliberately empty until its transcript lands, so
            # it records WHY it is empty from the moment it exists
            # (task-2364): a transcript that never arrives leaves a row
            # saying "pending", not an unexplained blank.
            metadata=self._console_realtime_row_metadata(
                model=CONSOLE_REALTIME_TRANSCRIPTION_MODEL,
                transcript_status="pending",
            ),
        )
        session.controller.on_turn_committed(time.monotonic())

    def _on_console_realtime_input_transcript(
        self, session: ConsoleRealtimeSession, text: str
    ) -> None:
        """`on_input_transcript`: fill in what the user actually said.

        `update_message_content`, NOT `append_stream_chunk`: the store
        refuses stream chunks on anything but an assistant row
        (`_validate_can_stream`), and this callback delivers the whole
        transcript exactly once (the provider's `...transcription.
        completed` event; the incremental `.delta` sibling is deliberately
        not wired). So there is nothing to append -- there is one final
        text to set.

        A transcript with no row to land in (a commit this wiring never
        saw, e.g. one that arrived during a reconnect) creates its own row
        rather than being dropped: losing what the user said is worse than
        a row slightly out of order.

        An ALREADY-FILLED row is never overwritten (fix round 1, F5). This
        callback carries no item id, and `user_row_id` moves to each new
        commit, so a transcription that finishes late -- after the next
        turn committed AND after that turn's own transcript landed --
        would otherwise replace a correct transcript with a stale one,
        putting words in the user's mouth in the durable record. Dropped
        instead, with the row id, because a wrong transcript is worse than
        a missing one and this is the only place it can be diagnosed.

        Every outcome is RECORDED on the row (task-2364): a transcript that
        legitimately came back empty marks its row `empty`, a write that
        failed marks it `failed`, and a filled row becomes `final`. Before
        the metadata field, the empty case simply returned here and left an
        empty user row stranded forever with nothing saying whether the
        user had been silent or the pipeline had broken. The empty case is
        now also durable (task-2391): see
        `_mark_console_realtime_transcript_empty`.
        """
        spoken = str(text or "").strip()
        row_id = session.user_row_id
        if not spoken:
            self._mark_console_realtime_transcript_empty(session, row_id)
            return
        if row_id is None:
            session.user_row_id = self._append_console_realtime_row(
                session,
                ConsoleMessageRole.USER,
                spoken,
                metadata=self._console_realtime_row_metadata(
                    model=CONSOLE_REALTIME_TRANSCRIPTION_MODEL,
                    transcript_status="final",
                ),
            )
            return
        store = self._chat_store_accessor()
        try:
            existing = str(store.get_message(row_id).content or "").strip()
        except Exception:  # noqa: BLE001 - an unreadable row is a dropped one
            logger.opt(exception=True).warning(
                "Console realtime: could not read the input-transcript row: "
                f"op=realtime_input_transcript row_id={row_id}"
            )
            return
        if existing:
            logger.warning(
                "Console realtime: dropping a late input transcript; its row "
                "already holds another turn's text: "
                f"op=realtime_input_transcript row_id={row_id}"
            )
            return
        try:
            store.finalize_deferred_user_message_content(row_id, spoken)
        except Exception:  # noqa: BLE001 - transcript upkeep is never fatal
            logger.opt(exception=True).warning(
                "Console realtime: could not write the input transcript"
            )
            self._set_console_realtime_transcript_status(row_id, "failed")
            return
        # AFTER the content write, never before: a status of "final" on a
        # row whose text never landed would be a lie of exactly the kind
        # this field exists to prevent.
        self._set_console_realtime_transcript_status(row_id, "final")
        session.transcript_dirty = True

    def _mark_console_realtime_transcript_empty(
        self, session: ConsoleRealtimeSession, row_id: str | None
    ) -> None:
        """Record a committed turn whose transcript came back with no words.

        task-2391: `set_message_metadata` alone (the pre-fix behavior) only
        ever reached a row that was ALREADY persisted -- an empty realtime
        user row never is, because the store defers persistence for
        content-less rows and the DB layer refuses to create a message with
        neither text nor an image at all (`CharactersRAGDB.add_message`).
        So the metadata write landed in memory only and vanished on
        restart. `CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER` is written
        as the row's CONTENT instead, through the same
        `update_message_content` call the "final" (real transcript) branch
        above uses -- which flushes the deferred create exactly as a real
        transcript would. The status write follows the content write, same
        order and same reason as the "final" branch: a status of "empty" on
        a row whose placeholder never landed would be a lie.

        Race-safe against a REAL transcript (matching the late-final-
        transcript guard above): a row already carrying different non-blank
        text is left alone, never overwritten.

        Retry-safe against a SWALLOWED status write (Qodo review, task-2391
        follow-up): the content write and the status write are two separate
        store calls, and `_set_console_realtime_transcript_status` (below)
        deliberately never raises -- a metadata-write failure there is
        logged and swallowed, not surfaced. An earlier version of this
        method used "does the row already have text" as its sole retry
        guard, which -- once the placeholder itself IS that text -- also
        blocked every later retry from ever reaching the status write
        again, permanently stranding a row whose content says "empty" but
        whose `transcript_status` never does (invisible to
        `_is_empty_transcript_row`, and so reachable by a provider as a
        fabricated user turn: the exact leak the placeholder was written to
        avoid, reopened by a different route). So content and status are
        each retried independently: content is written only when the row
        is genuinely still blank; status is (re-)written whenever the
        content is blank OR already the placeholder, never when it holds
        something else.

        Args:
            session: The live realtime loop state, for the repaint flag.
            row_id: Native store id of the committed turn's user row, or
                ``None`` when no row exists to mark (a commit this wiring
                never saw).
        """
        if row_id is None:
            return
        store = self._chat_store_accessor()
        try:
            existing = str(store.get_message(row_id).content or "").strip()
        except Exception:  # noqa: BLE001 - an unreadable row is left untouched
            logger.opt(exception=True).debug(
                "Console realtime: could not read a transcript row's text: "
                f"op=realtime_transcript_status row_id={row_id}"
            )
            return
        if existing and existing != CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER:
            # A real transcript is already there -- never relabel it "empty".
            return
        if not existing:
            try:
                store.finalize_deferred_user_message_content(
                    row_id, CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER
                )
            except Exception:  # noqa: BLE001 - transcript upkeep is never fatal
                logger.opt(exception=True).warning(
                    "Console realtime: could not record the empty-transcript row"
                )
                return
        # Reached with the placeholder now in place -- either just written
        # above, or already there from an earlier call whose status write
        # was swallowed. Either way, (re-)stamp the status: idempotent when
        # it already succeeded, and the only way a stranded row recovers
        # when it did not.
        self._set_console_realtime_transcript_status(row_id, "empty")
        session.transcript_dirty = True

    def _set_console_realtime_transcript_status(self, row_id: str, status: str) -> None:
        """Record what became of a user row's transcript (task-2364).

        Args:
            row_id: Native store id of the user row.
            status: A `MessageMetadata` transcript status
                ("final"/"empty"/"failed").
        """
        store = self._chat_store_accessor()
        try:
            store.set_message_metadata(
                row_id,
                self._console_realtime_row_metadata(
                    model=CONSOLE_REALTIME_TRANSCRIPTION_MODEL,
                    transcript_status=status,
                ),
            )
        except Exception:  # noqa: BLE001 - bookkeeping is never worth a crash
            logger.opt(exception=True).debug(
                "Console realtime: could not record a transcript status: "
                f"op=realtime_transcript_status row_id={row_id} status={status}"
            )

    def _on_console_realtime_reply_started(
        self, session: ConsoleRealtimeSession, item_id: str
    ) -> None:
        """`on_reply_started`: open the assistant's transcript row.

        Also the per-reply reset point for the audio accounting behind
        `played_ms` -- a barge-in must be measured against THIS reply's
        audio, not everything played since the loop started.
        """
        self._persist_console_realtime_event(
            "realtime_reply_started",
            operation="reply_started",
            phase=session.controller.state,
        )
        row_id = self._append_console_realtime_row(
            session,
            ConsoleMessageRole.ASSISTANT,
            "",
            metadata=self._console_realtime_row_metadata(model=str(realtime_model())),
        )
        session.assistant_row_id = row_id
        session.last_reply_row_id = row_id or session.last_reply_row_id
        session.fed_bytes = 0
        # A fresh attempt at the output device for this reply: the latch is
        # per-reply, not per-loop (the toast is the per-loop half).
        session.audio_failed_for_reply = False
        session.reply_token += 1
        session.generation_done = False
        session.playback_pending = False
        session.barged = False
        session.controller.on_reply_started()

    def _on_console_realtime_output_transcript_delta(
        self, session: ConsoleRealtimeSession, text: str
    ) -> None:
        """`on_output_transcript_delta`: stream the reply's own words in."""
        row_id = session.assistant_row_id
        if row_id is None or not text:
            return
        store = self._chat_store_accessor()
        try:
            store.append_stream_chunk(row_id, text)
        except Exception:  # noqa: BLE001
            logger.opt(exception=True).warning(
                "Console realtime: could not stream the reply transcript"
            )
            return
        session.transcript_dirty = True

    def _on_console_realtime_usage(
        self, session: ConsoleRealtimeSession, payload: dict
    ) -> None:
        """`on_usage`: attach billing to the reply it belongs to.

        Read from `last_reply_row_id`, not `assistant_row_id`: the provider
        fires this from the SAME `response.done` event that already fired
        `on_reply_done`, which closes the row -- so the usage for a reply
        always arrives just after that reply stopped being "current".
        """
        row_id = session.last_reply_row_id
        if row_id is None:
            return
        usage = ProviderUsage.from_provider_payload(
            payload,
            provider=CONSOLE_REALTIME_SUPPORTED_PROVIDER,
            model=str(realtime_model()),
        )
        if usage is None:
            return
        store = self._chat_store_accessor()
        try:
            store.set_message_usage(row_id, usage)
        except Exception:  # noqa: BLE001 - cost display is never worth a crash
            logger.opt(exception=True).debug(
                "Console realtime: could not attach usage to the reply"
            )

    def _on_console_realtime_transcription_usage(
        self, session: ConsoleRealtimeSession, payload: dict
    ) -> None:
        """`on_transcription_usage`: attach the USER turn's spoken-audio
        duration -- distinct from `_on_console_realtime_usage` (the
        ASSISTANT reply's token usage, from `response.done`).

        `payload` is `{"type": "duration", "seconds": N}` (live-confirmed,
        see `openai_session.py`'s ground-truth header) -- a duration, not a
        token count, so it is captured on `ProviderUsage.transcription_
        seconds` rather than any of the token buckets. Attached to
        `user_row_id` (this transcript's own row), never `last_reply_row_
        id` (the assistant's): confusing the two would bill the user's
        spoken-audio duration onto the assistant's reply.

        `pricing_catalog.py`'s cost math does not read `transcription_
        seconds` -- capturing it here does not make it billable; wiring a
        cost display for it is a separate follow-up task (task-2363's own
        AC treats cost-chip integration as explicitly out of scope).

        Mirrors `_on_console_realtime_input_transcript`'s late-arrival
        guard: a duration payload landing after `user_row_id` has already
        moved to the NEXT turn (and that turn's own duration usage, if any,
        already landed) must not clobber it -- dropped instead, loudly
        enough to diagnose.
        """
        if not isinstance(payload, dict) or payload.get("type") != "duration":
            return
        if "seconds" not in payload:
            return
        # `as_seconds` is `ProviderUsage`'s OWN sanitizer, shared rather than
        # re-implemented here so a duration means the same thing however it
        # enters the record. A bare `float()` let a negative, NaN or +/-inf
        # value off the wire into `transcription_seconds`, where it survived
        # `plus()` and was persisted -- as bare `NaN`/`Infinity` tokens that
        # strict JSON readers reject (Qodo Q2). Anything unusable becomes
        # 0.0: the turn still records WHICH provider/model transcribed it,
        # with no duration claimed.
        seconds = as_seconds(payload.get("seconds"))
        row_id = session.user_row_id
        if row_id is None:
            return
        store = self._chat_store_accessor()
        try:
            existing = store.get_message(row_id).usage
        except Exception:  # noqa: BLE001 - an unreadable row is a dropped one
            logger.opt(exception=True).warning(
                "Console realtime: could not read the transcription-usage row: "
                f"op=realtime_transcription_usage row_id={row_id}"
            )
            return
        if existing is not None:
            logger.warning(
                "Console realtime: dropping a late transcription usage; its "
                "row already holds another turn's usage: "
                f"op=realtime_transcription_usage row_id={row_id}"
            )
            return
        usage = ProviderUsage(
            transcription_seconds=seconds,
            provider=CONSOLE_REALTIME_SUPPORTED_PROVIDER,
            model=CONSOLE_REALTIME_TRANSCRIPTION_MODEL,
        )
        try:
            store.set_message_usage(row_id, usage)
        except Exception:  # noqa: BLE001 - cost display is never worth a crash
            logger.opt(exception=True).debug(
                "Console realtime: could not attach transcription usage"
            )

    def _append_console_realtime_row(
        self,
        session: ConsoleRealtimeSession,
        role: ConsoleMessageRole,
        content: str,
        *,
        metadata: MessageMetadata | None = None,
    ) -> str | None:
        """Append one continuity row to the loop's OWN Console session.

        Persisted like any other Console turn: a spoken conversation is a
        conversation, and a realtime exchange that vanished on restart
        would be the only kind that does.

        Args:
            session: The live realtime loop state.
            role: Transcript role for the new row.
            content: Row text ("" for a placeholder filled in later).
            metadata: Structured provenance/state to store with the row
                (task-2364). Passed at creation so the row's engine,
                provider and model are written by the same DB write as its
                text rather than chased with a second update.

        Returns:
            The new row's id, or None when the write failed (already
            logged) -- callers treat None as "no row to fill in later".
        """
        store = self._chat_store_accessor()
        try:
            message = store.append_message(
                session.console_session_id,
                role=role,
                content=content,
                persist=True,
                metadata=metadata,
            )
        except Exception:  # noqa: BLE001 - a store failure must not end the call
            logger.opt(exception=True).warning(
                "Console realtime: could not append a transcript row: "
                f"op=realtime_row role={role.value}"
            )
            return None
        session.transcript_dirty = True
        return message.id

    def _finish_console_realtime_reply_row(
        self, session: ConsoleRealtimeSession, *, interrupted: bool
    ) -> None:
        """Close the current reply's transcript row, marking a barge-in.

        The marker is appended BEFORE the terminal mark (the store refuses
        chunks on a completed row) and is what keeps the stored transcript
        honest: the user heard half a sentence, and everything downstream
        -- the seed on the next reconnect, an export, a summary -- reads
        this row as if it were the whole reply otherwise.
        """
        row_id, session.assistant_row_id = session.assistant_row_id, None
        if row_id is None:
            return
        store = self._chat_store_accessor()
        # The structured record (task-2364) is what the reseed builder,
        # exports and summaries read; the marker below stays because the
        # HUMAN reading the transcript needs to see it too. Written before
        # the terminal mark so the flush that persists the final text
        # carries the flag in the same write.
        try:
            store.set_message_metadata(
                row_id,
                self._console_realtime_row_metadata(
                    model=str(realtime_model()),
                    interrupted=interrupted,
                ),
            )
        except Exception:  # noqa: BLE001 - bookkeeping is never worth a crash
            logger.opt(exception=True).debug(
                "Console realtime: could not record the reply's metadata"
            )
        if interrupted:
            try:
                store.append_stream_chunk(row_id, CONSOLE_REALTIME_INTERRUPTED_MARKER)
            except Exception:  # noqa: BLE001
                logger.opt(exception=True).debug(
                    "Console realtime: could not mark the reply interrupted"
                )
        try:
            store.mark_message_complete(row_id)
        except Exception:  # noqa: BLE001
            logger.opt(exception=True).debug(
                "Console realtime: could not complete the reply row"
            )
        session.transcript_dirty = True

    def _console_realtime_adopt_transcript(self, transcript: str) -> bool:
        """Claim a just-finished pipeline capture as this loop's first turn.

        Returns True when the realtime loop CONSUMED the transcript, which
        is the caller's signal not to insert it into the composer draft as
        well -- the words were spoken as a turn, not typed as a draft, and
        leaving a copy behind would re-send them the next time the user
        pressed Enter.

        A transcript that lands before the handshake completes is held
        (`pending_text_turn`) rather than enqueued into a session that
        cannot send it yet; `_on_console_realtime_ready` releases it.
        """
        session = self.session
        if session is None or not session.adopt_capture:
            return False
        session.adopt_capture = False
        spoken = str(transcript or "").strip()
        if not spoken:
            return True
        if session.ready:
            self._send_console_realtime_text_turn(session, spoken)
        else:
            session.pending_text_turn = spoken
        return True

    def _send_console_realtime_text_turn(
        self, session: ConsoleRealtimeSession, text: str
    ) -> None:
        """Send one TEXT turn (an adopted capture) into the live session.

        `on_turn_committed` is a server-side signal about the AUDIO input
        buffer, so it never fires for a text item -- which would leave the
        FSM sitting in `live` while a reply streamed, never gating the mic
        and never painting `thinking`. Driving the same input directly
        here is what makes an adopted turn behave like any other turn.
        """
        self._append_console_realtime_row(
            session,
            ConsoleMessageRole.USER,
            text,
            # An adopted capture's WORDS came from the pipeline engine's
            # STT, not from the realtime provider's transcription, so no
            # transcription model is claimed here (task-2364) -- the row
            # belongs to this realtime session and its text is already
            # final, and that is all this record asserts. `set_message_
            # metadata` replaces a record wholesale, but this row is never
            # re-stamped: its id is deliberately not kept as
            # `user_row_id` (that tracks AUDIO turns), so nothing later
            # overwrites the blank model with the transcription model.
            metadata=self._console_realtime_row_metadata(
                model="",
                transcript_status="final",
            ),
        )
        provider_session = session.session
        if provider_session is None:
            return
        try:
            provider_session.send_text_item(text, request_response=True)
        except Exception:  # noqa: BLE001
            logger.opt(exception=True).warning(
                "Console realtime: could not send the adopted transcript"
            )
            return
        session.controller.on_turn_committed(time.monotonic())

    def _on_console_realtime_audio_delta(
        self, session: ConsoleRealtimeSession, pcm: bytes
    ) -> None:
        """`on_audio_delta`: hand one chunk of reply audio to the sink.

        The sink and its pump task are created lazily, on the FIRST chunk
        of a reply rather than at reply start: a reply that never produces
        audio (a cancelled or failed one) must not open an output device
        for nothing.

        `fed_bytes` is counted HERE, at the queue, which is what makes
        `played_ms` over-count rather than under-count -- see
        `_console_realtime_played_ms` for why that direction is the safe
        one.

        A sink that could not be opened is LATCHED for the rest of the
        reply (fix round 1, F2). Audio deltas arrive roughly per 20 ms of
        speech, so retrying the open per delta meant one construction --
        and one logged traceback, on the UI thread -- every 20 ms for as
        long as the assistant talked. The device is not coming back
        mid-reply; the next reply gets a fresh attempt.
        """
        if not pcm:
            return
        if session.audio_failed_for_reply:
            return
        if session.audio_queue is None:
            self._begin_console_realtime_reply_audio(session)
        queue = session.audio_queue
        if queue is None:
            return
        session.fed_bytes += len(pcm)
        try:
            queue.put_nowait(pcm)
        except Exception:  # noqa: BLE001 - a full/closed queue is not fatal
            logger.opt(exception=True).debug("Console realtime: dropped an audio chunk")

    def _begin_console_realtime_reply_audio(
        self, session: ConsoleRealtimeSession
    ) -> None:
        """Open this reply's audio sink and start its pump task.

        One sink and one pump per reply: `StreamingPcmSink` instances are
        single-use by contract (open -> feed -> close/stop, then discard),
        and a per-reply pump is what lets a barge-in abort exactly this
        reply's audio without disturbing anything else.

        Failure is latched rather than retried (see
        `_on_console_realtime_audio_delta`), logged ONCE per reply and
        toasted ONCE per loop entry -- a device that is missing will be
        missing for every reply, and one toast per reply would bury the
        conversation the user is still having.
        """
        try:
            sink = self._build_console_realtime_sink()
        except Exception:  # noqa: BLE001 - the conversation survives mute audio
            sink = None
            logger.opt(exception=True).warning(
                "Console realtime: could not build the audio sink"
            )
        if sink is None:
            self._note_console_realtime_audio_unavailable(session)
            return
        try:
            sink.open(CONSOLE_REALTIME_SAMPLE_RATE, 1)
        except Exception:  # noqa: BLE001 - the conversation survives mute audio
            logger.opt(exception=True).warning(
                "Console realtime: could not open the audio sink"
            )
            self._note_console_realtime_audio_unavailable(session)
            return
        queue: asyncio.Queue = asyncio.Queue()
        session.sink = sink
        session.audio_queue = queue
        session.fed_bytes = 0
        # From here until the pump reports back, this reply is not over --
        # however long ago the provider stopped generating it.
        session.playback_pending = True
        session.pump_worker = self._run_worker(
            self._pump_console_realtime_audio(
                session, session.reply_token, sink, queue
            ),
            exclusive=False,
            group="console-realtime-audio",
            exit_on_error=False,
        )

    def _note_console_realtime_audio_unavailable(
        self, session: ConsoleRealtimeSession
    ) -> None:
        """Latch "no reply audio this reply", and say so once per loop."""
        session.audio_failed_for_reply = True
        # Persisted every time, not just the first: the toast is
        # deduplicated for the user's sake, but "which replies were
        # silent" is exactly what a support log needs.
        self._persist_console_realtime_event(
            "realtime_audio_begin_failed",
            operation="audio_begin",
            status="failed",
            error_category="sink_unavailable",
        )
        if session.audio_unavailable_notified:
            return
        session.audio_unavailable_notified = True
        self._notify(CONSOLE_REALTIME_AUDIO_UNAVAILABLE_MESSAGE, severity="warning")

    def _build_console_realtime_sink(self) -> Any:
        """Construct the reply-audio sink, honoring the test seam.

        Imported inside the method for the same reason the mic tap is: the
        sink module reaches an audio backend, and a Console mount that
        never speaks must not pay for it.
        """
        factory = self._sink_factory_accessor()
        if callable(factory):
            return factory()
        from ...Audio.streaming_sink import StreamingPcmSink

        return StreamingPcmSink(on_event=self._on_console_realtime_sink_event)

    def _on_console_realtime_sink_event(self, event: object) -> None:
        """Sink lifecycle events. Logged only -- fired on the sink's own
        notify thread, so nothing here may touch widgets."""
        logger.debug(f"Console realtime: sink event: op=sink_event event={event!r}")

    async def _pump_console_realtime_audio(
        self, session: ConsoleRealtimeSession, token: int, sink: Any, queue: Any
    ) -> None:
        """Feed one reply's queued audio into `sink`, then report playback end.

        The queue's `None` item is the end-of-reply sentinel: it ends the
        async iterator, which is what tells `pump` to close the sink and
        let the buffered tail actually finish playing (rather than cutting
        it off the way an abort does).

        `pump` returning is the sink reaching a terminal state -- drained
        (the device played everything), stopped (a barge-in or teardown
        aborted it), or failed. `settle()` then waits for that terminal
        EVENT to have been delivered, which `pump` explicitly does not
        promise (its own N4 note): the same "playback is really over"
        signal the V3 TTS path waits on before reporting an utterance
        finished. It blocks, so it runs off-thread.

        Whatever the outcome, this reply's audio is over exactly once, so
        `_console_realtime_playback_finished` is called on every exit --
        it owns the decision about whether that means anything to the FSM.
        """
        from ...Audio.streaming_sink import pump

        async def _chunks():
            while True:
                chunk = await queue.get()
                if chunk is None:
                    return
                yield chunk

        try:
            await pump(sink, _chunks())
            settle = getattr(sink, "settle", None)
            if callable(settle):
                await asyncio.to_thread(settle)
        except Exception:  # noqa: BLE001 - a pump failure still ends playback
            logger.opt(exception=True).warning(
                "Console realtime: reply audio playback failed"
            )
        finally:
            self._console_realtime_playback_finished(session, token)

    def _end_console_realtime_reply_audio(
        self, session: ConsoleRealtimeSession, *, abort: bool
    ) -> None:
        """End this reply's audio: drain it, or cut it off.

        `abort=False` (the reply finished) closes the source and lets the
        already-buffered tail play out. `abort=True` (a barge-in) stops the
        sink outright -- the whole point of barging in is that the
        assistant stops talking NOW, not at the end of the buffer.

        `session.sink` is deliberately NOT cleared on the drain path: the
        sink is still playing, and exit teardown must still be able to
        silence it. The next reply replaces it.
        """
        queue, session.audio_queue = session.audio_queue, None
        if queue is not None:
            try:
                queue.put_nowait(None)
            except Exception:  # noqa: BLE001
                logger.opt(exception=True).debug(
                    "Console realtime: could not close the audio source"
                )
        if not abort:
            return
        sink = session.sink
        if sink is not None:
            try:
                sink.stop()
            except Exception:  # noqa: BLE001
                logger.opt(exception=True).warning(
                    "Console realtime: could not stop the audio sink"
                )

    def _on_console_realtime_first_audio(self, session: ConsoleRealtimeSession) -> None:
        """`on_first_audio`: reply audio started -- `thinking` -> `speaking`."""
        self._persist_console_realtime_event(
            "realtime_first_audio",
            operation="first_audio",
            phase=session.controller.state,
        )
        session.controller.on_first_audio()

    def _on_console_realtime_reply_done(self, session: ConsoleRealtimeSession) -> None:
        """`on_reply_done`: GENERATION finished. Not necessarily the reply.

        Never fires for a response this client cancelled (Task 2's
        semantics), so there is no barge-in case to disambiguate here.

        It does NOT go straight to the FSM (live-gate defect, default
        speaker-safe mode: the model heard itself and answered its own
        voice). `response.done` means the provider finished GENERATING,
        and 24 kHz audio generates far faster than it plays -- the sink
        still holds seconds of the reply at this point. Telling the FSM
        the reply was over here left `speaking` early, which ungated the
        mic straight into the reply's own audible tail; the provider's
        server-side VAD then committed the model's voice as the user's
        next turn.

        So this half only records that generation is done and closes the
        audio source (letting the buffered tail play out). Whichever of
        the two halves finishes LAST -- this one or
        `_console_realtime_playback_finished` -- is what tells the FSM.
        A reply that produced no audio at all has no playback half, and
        completes here immediately.
        """
        session.generation_done = True
        self._end_console_realtime_reply_audio(session, abort=False)
        self._finish_console_realtime_reply_row(session, interrupted=False)
        self._persist_console_realtime_event(
            "realtime_reply_done",
            operation="reply_done",
            initiator="generation",
            decision="deferred" if session.playback_pending else "fired",
            phase=session.controller.state,
            cancelled=session.barged,
        )
        if session.playback_pending:
            return
        session.controller.on_reply_done(time.monotonic())

    def _console_realtime_playback_finished(
        self, session: ConsoleRealtimeSession, token: int
    ) -> None:
        """This reply's audio has finished playing (or was aborted).

        The other half of the rendezvous in
        `_on_console_realtime_reply_done`. Three guards, each for a real
        case:

          * a different loop owns the screen now (exit/teardown, whose
            abort makes the pump return) -- report nothing;
          * a NEWER reply is in flight (`token`), so this completion
            belongs to a reply the FSM has already moved past -- reporting
            it would end the current one;
          * the user barged in, and Task 2's contract is that a cancelled
            response completes nothing. The FSM already returned to `live`
            through its own barge-in input.
        """
        if self.session is not session:
            return
        if session.reply_token != token:
            return
        session.playback_pending = False
        fires = session.generation_done and not session.barged
        self._persist_console_realtime_event(
            "realtime_reply_done",
            operation="reply_done",
            initiator="playback",
            decision="fired" if fires else "dropped",
            phase=session.controller.state,
            cancelled=session.barged,
        )
        if not fires:
            return
        session.controller.on_reply_done(time.monotonic())

    def _on_console_realtime_speech_started(
        self, session: ConsoleRealtimeSession
    ) -> None:
        """`on_speech_started`: server-side VAD heard the user start talking.

        The FSM itself decides whether that is a barge-in (acoustic mode
        only) or noise to ignore.
        """
        session.barge_trigger = "speech"
        session.controller.on_speech_started()

    def _on_console_realtime_error(
        self, session: ConsoleRealtimeSession, exc: Exception
    ) -> None:
        """`on_error`: terminal before the handshake, logged after it.

        Once the session is live, a provider error that actually ends it
        arrives separately as `on_closed`, and treating every error event
        as terminal would end a working conversation over one recoverable
        event.

        BEFORE `on_ready`, the same event means the opposite: the
        handshake did not succeed, and (live-confirmed) it is how an
        invalid key is reported -- OpenAI accepts the WebSocket upgrade,
        so `connect()` returns cleanly and the refusal arrives here. There
        is no reply-in-flight to protect at that point, so it routes to
        the connect-failure path rather than being logged into a chip that
        would otherwise say `connecting…` forever.
        """
        if not session.ready:
            self._console_realtime_connect_failed(session, session.connect_attempt, exc)
            return
        logger.warning(
            "Console realtime: provider error: op=realtime_error "
            f"reason={self._sanitize_console_realtime_failure(exc)!r}"
        )

    def _on_console_realtime_closed(
        self, session: ConsoleRealtimeSession, reason: str
    ) -> None:
        """`on_closed`: the transport ended.

        A close this wiring performed deliberately (exit, reconnect) can
        never reach here -- both paths supersede the attempt first, and the
        marshal drops the callback before it lands. So anything arriving
        here is an unexpected end.

        WHEN it arrives decides what it means. After the handshake, it is
        a transport drop and the FSM's reconnect-once policy decides
        between a retry and giving up. BEFORE the handshake was
        acknowledged, it is a REFUSED CONNECT wearing a close's clothes:
        the provider accepted the upgrade and then rejected the session
        (an invalid key closes with 3000/`invalid_api_key`). The FSM
        deliberately ignores a transport-closed input while `connecting`
        -- Task 4's state table assumes connect failures surface as
        `connect()` raising -- so routing it there left the loop parked in
        `connecting` with no toast, forever. It goes to the same
        connect-failure path a raising `connect()` takes, which is where
        the reasoned exit and the loud fallback already live.
        """
        if not session.ready:
            self._console_realtime_connect_failed(
                session, session.connect_attempt, RuntimeError(reason)
            )
            return
        session.failure_text = self._sanitize_console_realtime_failure(reason)
        logger.info(
            "Console realtime: transport closed: op=realtime_closed "
            f"reason={session.failure_text!r}"
        )
        session.controller.on_transport_closed(error=True)

    def _handle_console_realtime_intent(self, intent: object) -> None:
        """Route one intent emitted synchronously by `RealtimeLoopController`.

        The V4 FSM emits a strict subset of V3's vocabulary
        (`ModeChanged`/`ExitLoop`/`SilenceSpeech`, imported from
        `console_hands_free.py` rather than redefined), so this dispatcher
        mirrors `_handle_console_hands_free_intent`'s shape exactly.
        """
        if isinstance(intent, SilenceSpeech):
            self._console_realtime_silence_speech()
        elif isinstance(intent, ModeChanged):
            self._console_realtime_mode_changed(intent.state, intent.reason)
        elif isinstance(intent, ExitLoop):
            self._console_realtime_exit_loop(intent.reason)

    def _console_realtime_mode_changed(self, state: str, reason: str | None) -> None:
        """`ModeChanged`: sync the mic gate, handle reconnects, repaint.

        The mic gate is synced on EVERY transition, unconditionally (rule
        7): `mic_gated` is a derived property of the FSM's state, so
        syncing it anywhere less than every transition would let the two
        drift -- and a mic left hot while the assistant speaks feeds the
        reply's own audio straight back into the provider.
        """
        session = self.session
        if session is None:
            return
        self._runtime_accessor().persona_buddy_sink.voice_state(
            session.console_session_id,
            session.buddy_generation,
            state,
        )
        gated = session.controller.mic_gated
        session.mic_gated = gated
        tap = session.tap
        if tap is not None:
            try:
                tap.set_gated(gated)
            except Exception:  # noqa: BLE001
                logger.opt(exception=True).debug(
                    "Console realtime: could not sync the mic gate"
                )
        if reason == "reconnecting":
            self._notify(CONSOLE_REALTIME_RECONNECTING_MESSAGE, severity="warning")
            self._console_realtime_begin_reconnect(session)
        self._repaint_chip()

    def _console_realtime_begin_reconnect(
        self, session: ConsoleRealtimeSession
    ) -> None:
        """Open a fresh session for the same loop after a transport drop.

        The old session is released and a new one built through the SAME
        factory and the SAME connect path, so a reconnect re-seeds from the
        store (including everything said since the loop started) exactly
        the way the first connect did. Incrementing the attempt inside
        `_start_console_realtime_connect` is what retires the dead
        session's callbacks.

        `tap.begin_buffering()` runs FIRST, before anything else here
        (task-2360): the mic tap is never rebuilt across a reconnect (it
        is the SAME device stream for the whole loop entry), so without
        this, speech captured in the window between here and the new
        session's `on_ready` would either reach nobody (`session.session`
        is momentarily None below) or reach a session that has not
        finished its handshake yet (`session.session` is reassigned to
        the new, not-yet-connected provider session inside `_connect_
        console_realtime`, well before it calls `connect()` -- a real
        session's `append_audio` silently drops anything sent before that
        completes). Buffering at the tap, rather than depending on either
        of those downstream behaviors, mirrors the ENTRY-time first-words
        guarantee exactly: `_on_console_realtime_ready`'s existing `tap.
        mark_ready()` call (unconditionally run for both a first connect
        and every reconnect) is what releases it, in order, once the new
        session is actually ready -- no other change needed there.
        """
        tap = session.tap
        if tap is not None:
            try:
                tap.begin_buffering()
            except Exception:  # noqa: BLE001
                logger.opt(exception=True).debug(
                    "Console realtime: could not re-arm the mic tap buffer "
                    "for reconnect"
                )
        provider_session, session.session = session.session, None
        session.ready = False
        self._persist_console_realtime_event(
            "realtime_reconnect",
            operation="reconnect",
            status="started",
            error_category=self._console_realtime_failure_token(session.failure_text),
        )
        # A reply that was in flight when the transport died is over, and
        # over abruptly: close its audio and its transcript row as an
        # interruption rather than leaving a `pending` row that will never
        # complete and a pump parked on a queue nobody feeds.
        self._end_console_realtime_reply_audio(session, abort=True)
        self._finish_console_realtime_reply_row(session, interrupted=True)
        if provider_session is not None:
            self._run_worker(
                self._close_console_realtime_session(provider_session),
                exclusive=False,
                group="console-realtime-close",
                exit_on_error=False,
            )
        self._start_console_realtime_connect(session)

    def _console_realtime_silence_speech(self) -> None:
        """`SilenceSpeech`: barge-in -- stop talking, tell the provider.

        `cancel_response(played_ms)` is what keeps the provider's record of
        the conversation honest: without it the model believes the user
        heard the whole reply it was midway through generating.
        """
        session = self.session
        if session is None:
            return
        # Read the count BEFORE tearing the audio down, then silence, then
        # tell the provider -- in that order: the user must stop hearing the
        # reply first, and `played_ms` must describe what they heard up to
        # that moment.
        played_ms = self._console_realtime_played_ms(session)
        self._persist_console_realtime_event(
            "realtime_barge",
            operation="barge",
            # Which input barged is the FIRST question asked of any
            # barge-in report, and the intent itself does not carry it --
            # `SilenceSpeech` is shared by both triggers, so the wiring
            # records which one it just handed the FSM.
            initiator=session.barge_trigger,
            phase=session.controller.state,
            duration_ms=played_ms,
        )
        # Latched before the abort: the pump is about to unwind and report
        # playback finished, and a cancelled reply must complete nothing
        # (Task 2's contract, mirrored in
        # `_console_realtime_playback_finished`).
        session.barged = True
        self._end_console_realtime_reply_audio(session, abort=True)
        self._finish_console_realtime_reply_row(session, interrupted=True)
        provider_session = session.session
        if provider_session is not None:
            try:
                sent = provider_session.cancel_response(played_ms)
            except Exception:  # noqa: BLE001
                logger.opt(exception=True).warning(
                    "Console realtime: cancel_response failed"
                )
            else:
                # The provider's own guard refuses a cancel for a response
                # that already ended. "Told the provider" and "there was
                # nothing left to cancel" are different incidents and were
                # indistinguishable from outside the session.
                self._persist_console_realtime_event(
                    "realtime_cancel_sent"
                    if sent is not False
                    else "realtime_cancel_noop",
                    operation="cancel",
                    decision="sent" if sent is not False else "noop",
                    duration_ms=played_ms,
                )

    def _console_realtime_played_ms(self, session: ConsoleRealtimeSession) -> int:
        """Milliseconds of THIS reply's audio the user has plausibly heard.

        Counted from bytes handed to the sink, not from bytes the device
        actually rendered, so it OVER-counts by at most the sink's own
        buffered depth. That is the safe direction on purpose: `played_ms`
        drives the provider's `conversation.item.truncate`, and truncating
        slightly LATE leaves a few words in the model's record that the
        user nearly heard, while truncating early would delete words they
        definitely did hear -- which then reads as the model denying it
        ever said them.
        """
        return int(session.fed_bytes * 1000 / CONSOLE_REALTIME_BYTES_PER_SECOND)

    def _console_realtime_exit_loop(self, reason: str | None) -> None:
        """`ExitLoop`: tear the loop down, then say why it ended.

        Teardown happens FIRST so nothing can keep streaming into a loop
        the user has already been told is over.
        """
        session = self.session
        if session is None:
            return
        failure = session.failure_text
        self._persist_console_realtime_event(
            "realtime_exit",
            operation="exit",
            # The FSM's own reason vocabulary is already token-shaped
            # ("connect-failed", "connection-lost", "idle-timeout"); a
            # user-initiated exit has no reason, which is itself the fact
            # worth recording.
            status=safe_metadata_token(reason or "user"),
        )
        self._teardown_console_realtime_loop()
        if reason == "connect-failed":
            self._console_realtime_fallback_to_pipeline(failure)
            return
        message = self._console_realtime_exit_message(reason, session)
        if message:
            self._notify(message, severity="warning")

    def _console_realtime_exit_message(
        self, reason: str | None, session: ConsoleRealtimeSession
    ) -> str:
        """Turn an `ExitLoop` reason into user-facing copy.

        A reasonless exit (the user pressed Esc or the mic) gets NO toast:
        they know what they just did, and narrating it back is noise.
        """
        if reason == "connection-lost":
            return CONSOLE_REALTIME_EXIT_CONNECTION_LOST_MESSAGE
        if reason == "idle-timeout":
            return CONSOLE_REALTIME_EXIT_IDLE_TEMPLATE.format(
                minutes=round(session.idle_timeout_seconds / 60.0, 1)
            )
        return ""

    def _console_realtime_fallback_to_pipeline(self, failure: str) -> None:
        """The realtime engine could not start: fall back, loudly, or refuse.

        "Loudly" is the whole point (rule 4). Silently downgrading to the
        pipeline engine would leave the user believing they are talking to
        a realtime session -- with its latency, its barge-in, and its
        billing -- when they are not. And when the pipeline stack is not
        usable either, BOTH reasons are named: a bare "hands-free
        unavailable" sends the user hunting through the realtime config
        for a fault that is really a missing microphone or speech model.
        """
        reason = failure or "the realtime session could not be opened"
        # Both pipeline actions are explicit late-bound dependencies on the
        # hands-free owner; the realtime controller never reaches through a
        # screen or sibling controller.
        pipeline_reason = self._pipeline_blocker()
        if pipeline_reason is None:
            self._notify(
                CONSOLE_REALTIME_FALLBACK_TEMPLATE.format(reason=reason),
                severity="warning",
            )
            self._enter_pipeline_loop(self._dictation_state_accessor() == "recording")
            return
        self._notify(
            CONSOLE_REALTIME_NO_LOOP_TEMPLATE.format(
                reason=reason, pipeline_reason=pipeline_reason
            ),
            severity="error",
        )

    def _tick_console_realtime(self) -> None:
        """`set_interval(0.1, ...)`: the FSM's only clock input.

        Also the transcript's repaint cadence. The ordinary Console
        transcript timer is gated on a chat-controller run being in flight
        and self-stops when there is none -- a realtime conversation has no
        such run, so it would never repaint. Coalescing here (rather than
        resyncing per delta) keeps one full UI rebuild per 0.1 s instead of
        one per audio-transcript chunk.
        """
        session = self.session
        if session is None:
            return
        now = time.monotonic()
        if (
            not session.ready
            and session.connect_returned_at is not None
            and now - session.connect_returned_at
            >= CONSOLE_REALTIME_READY_TIMEOUT_SECONDS
        ):
            # `connect()` returned and then NOTHING arrived -- no ready, no
            # error, no close. Whatever that is, it is not a live session,
            # and the entry must not sit at `connecting…` waiting for it.
            self._console_realtime_connect_failed(
                session,
                session.connect_attempt,
                TimeoutError(
                    CONSOLE_REALTIME_HANDSHAKE_INCOMPLETE_MESSAGE.format(
                        seconds=CONSOLE_REALTIME_READY_TIMEOUT_SECONDS
                    )
                ),
            )
            return
        session.controller.tick(now)
        self._repaint_chip()
        if session.transcript_dirty:
            session.transcript_dirty = False
            # `call_later`, not `run_worker`: this repaint is ordinary screen
            # work with no lifetime of its own, and a worker outliving the
            # screen (a repaint still mounting rows while the transcript is
            # being pruned) is a teardown hazard -- a queued callback is
            # simply dropped when the screen goes away.
            self._defer_native_sync()

    def _release_console_realtime_state(self) -> tuple[Any, Any, Any, Any] | None:
        """Drop the loop and hand its resources to the async release.

        What happens synchronously here is only what is instant: the tick
        timer stops, the reply row closes, and the tap is GATED -- a plain
        flag flip that stops it feeding the session immediately.

        `tap.stop()` itself is deliberately NOT called here (fix round 1,
        F3). It waits up to 2 s for in-flight `on_frames` callbacks to
        quiesce and then joins the recorder thread, which is the exact
        ~4 s frozen-UI class `_discard_console_dictation_session` already
        documents. It moves to the async release, where it still runs
        FIRST -- before the session close -- so the teardown ORDER (tap ->
        session -> sink) is unchanged.

        Returns:
            The `(tap, provider_session, sink, audio_queue)` tuple still
            needing an async release, or None when no loop was running.
            The queue rides along so the reply's pump task -- parked on
            `queue.get()` and therefore blind to a sink that went terminal
            underneath it -- can be released once, at the END of teardown,
            without racing the sink ordering above.
        """
        session = self.session
        if session is None:
            return None
        self._runtime_accessor().persona_buddy_sink.release_voice(
            session.console_session_id,
            session.buddy_generation,
        )
        self.session = None
        # Exiting mid-reply IS an interruption: close the row that way
        # rather than leaving a `pending` assistant message that nothing
        # will ever complete.
        self._finish_console_realtime_reply_row(session, interrupted=True)
        if session.tick_timer is not None:
            try:
                session.tick_timer.stop()
            except Exception:  # noqa: BLE001
                logger.opt(exception=True).debug(
                    "Console realtime: stopping the tick timer failed"
                )
        tap, session.tap = session.tap, None
        if tap is not None:
            try:
                # Instant, non-blocking: frames are dropped from now on, so
                # nothing reaches a session that is about to close even
                # though the real `stop()` happens off-thread below.
                tap.set_gated(True)
            except Exception:  # noqa: BLE001
                logger.opt(exception=True).debug(
                    "Console realtime: could not gate the mic tap for teardown"
                )
        provider_session, session.session = session.session, None
        sink, session.sink = session.sink, None
        queue, session.audio_queue = session.audio_queue, None
        session.pump_worker = None
        return tap, provider_session, sink, queue

    def _teardown_console_realtime_loop(self) -> None:
        """Exit teardown.

        Order, end to end: gate + drop the loop state (sync, instant) ->
        repaint the chip back to the ordinary dictation state (sync, so
        the user sees the loop end immediately rather than after the
        device teardown) -> tap.stop -> provider session close -> sink
        stop -> pump released, all on a worker because the first three of
        those block (fix round 1, F3/F10).
        """
        released = self._release_console_realtime_state()
        if released is None:
            return
        tap, provider_session, sink, queue = released
        # Handle retained (fix round 1, F7): once the loop state is
        # dropped, this worker is the ONLY thing still holding the
        # WebSocket and the microphone. An unmount landing before it runs
        # -- exiting the loop and leaving the screen in the same breath is
        # an ordinary thing to do -- has nothing else left to release them
        # by, so `on_unmount` waits on this.
        self.close_worker = self._run_worker(
            self._close_console_realtime_resources(tap, provider_session, sink, queue),
            exclusive=False,
            group="console-realtime-close",
            exit_on_error=False,
        )
        self._restore_voice_chip()

    async def _close_console_realtime_resources(
        self, tap: Any, provider_session: Any, sink: Any, queue: Any = None
    ) -> None:
        """Release the tap, then the session, then the sink -- in that order.

        `tap.stop()` runs through `asyncio.to_thread`: it waits for
        in-flight `on_frames` callbacks to quiesce (bounded at 2 s) and
        then joins the recorder thread, which is seconds of frozen UI if
        called inline -- the same reason `_discard_console_dictation_
        session` exists. Still FIRST, so the microphone is released before
        the session it was feeding.

        Session before sink: closing it stops new audio arriving, so the
        sink is never asked to play a chunk that outlived the
        conversation. The pump's source is closed LAST, once the sink is
        already terminal, so the pump returns immediately instead of
        draining a reply the user has already left.
        """
        if tap is not None:
            try:
                await asyncio.to_thread(tap.stop)
            except Exception:  # noqa: BLE001
                logger.opt(exception=True).warning(
                    "Console realtime: stopping the mic tap failed"
                )
        if provider_session is not None:
            await self._close_console_realtime_session(provider_session)
        if sink is not None:
            try:
                sink.stop()
            except Exception:  # noqa: BLE001
                logger.opt(exception=True).debug(
                    "Console realtime: stopping the audio sink failed"
                )
        if queue is not None:
            try:
                queue.put_nowait(None)
            except Exception:  # noqa: BLE001
                logger.opt(exception=True).debug(
                    "Console realtime: could not release the audio pump"
                )

    async def _close_console_realtime_session(self, provider_session: Any) -> None:
        """Close one provider session; failures are logged, never raised."""
        try:
            await provider_session.close()
        except Exception:  # noqa: BLE001 - teardown must never raise at the user
            logger.opt(exception=True).warning(
                "Console realtime: closing the provider session failed"
            )

    def handle_key(self, key: str) -> bool:
        """Handle realtime key policy and report whether Escape was consumed."""
        session = self.session
        if session is None:
            return False
        if key == "escape":
            session.controller.on_exit_request()
            return True
        session.barge_trigger = "keypress"
        session.controller.on_keypress()
        return False

    async def teardown(self) -> None:
        """Release an active loop and await any retained close worker."""
        released = self._release_console_realtime_state()
        if released is not None:
            tap, provider_session, sink, queue = released
            await self._close_console_realtime_resources(
                tap, provider_session, sink, queue
            )
        close_worker, self.close_worker = self.close_worker, None
        if close_worker is not None:
            try:
                await close_worker.wait()
            except Exception:  # noqa: BLE001 - cancelled release is harmless
                logger.opt(exception=True).debug(
                    "Console realtime: waiting for the release worker failed"
                )
