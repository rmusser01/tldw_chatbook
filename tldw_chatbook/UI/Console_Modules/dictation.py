"""Console dictation controller.

Extracted out of `ChatScreen` (wave-1 console decomposition, task 5): the
one-shot mic-button dictation lifecycle and its streaming session adapter.
This is wave 1's proof of the OTHER collaborator kind the design spec
defines alongside region widgets
(`Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md`) -- a
controller, "a plain object that owns state and behaviour with no region
of its own."

Moved verbatim:

- The three module-level types dictation posts/holds: `ConsoleDictationEvent`,
  `ConsoleDictationLimitSignal` (both `Message` subclasses the screen still
  receives -- see the two one-line `@on` delegations `ChatScreen` keeps)
  and `ConsoleStreamingDictationSession` (the streaming session adapter
  over `ConsoleVoiceInputController`).
- The private module-level helpers only that session and the controller's
  own event handling use: `_INLINE_BREAK_COMMANDS`, `_join_segments`,
  `_voice_command_chip_ack`, and the four `_VOICE_ACK_*` acknowledgement
  strings (`ChatScreen` imports the two of those it still needs, for its
  own `_run_pending_console_voice_action`, which stays on the screen --
  see that method's docstring for why).
- Every `ChatScreen` method whose body read or wrote `_console_dictation_
  state` (or one of its seven sibling attributes -- `_console_dictation_
  session`, `..._partial`, `..._timer`, `..._elapsed_timer`, `..._origin_
  session_id`, `_console_pending_voice_action`, `_console_dictation_late_
  discard_ack`) and touched nothing region-shaped: the 20 methods matching
  `*dictation*` on the old class.

`ChatScreen` keeps six one-line delegations, under their ORIGINAL private
names, because each is reached from outside this cluster:
`_request_console_dictation_start/_stop/_cancel` (the mic button's
`on_button_pressed` branch, plus the hands-free wiring's own capture-open/
close/force-send/stop calls), `_sync_console_dictation_availability`
(`on_mount`'s post-mount probe), and the two `@on`-decorated message
handlers `_handle_console_dictation_event` / `_handle_console_dictation_
buffer_limit` (Textual's own message dispatch requires the decorator to
live on the class that receives the message). Every other one of the 20
moves with no residue left on the screen.

The controller boundary is the state attribute, not the file: dictation's
own methods stay entangled with the hands-free loop (`_console_hands_free`
and friends) and with two not-yet-extracted composer/workspace attributes
(`_console_undo_histories`, `_console_visible_draft_session_id`). Rather
than reach through a bare screen handle ad hoc at each of those call
sites, this module's constructor binds a NAMED CALLABLE for each, under
the SAME name the original `ChatScreen` method or attribute used, exposed
back to the 20 method bodies below as a thin property. That is what lets
every one of those bodies be a byte-for-byte copy of the pre-extraction
source: no internal line needed to change, because every name a body
references still resolves -- either to this controller's own moved state,
or to one of these bound names.

Wave 1 shipped six of these bindings (`_console_hands_free` and its
`_console_hands_free_vad_degraded` sibling, `_enter_console_hands_free_
loop`, `_console_hands_free_force_immediate_send`, `_deliver_console_
hands_free_capture_ended`, and `_run_pending_console_voice_action`) as
live `@property`s reaching straight through `screen`, explicitly disclosed
as a temporary exception: hands-free had no controller of its own yet to
hand a named dependency to. It does now (`ConsoleHandsFreeController`,
`hands_free.py`, wave-2 console decomposition task 1). That exception is
over -- every one of those, plus `_console_realtime_adopt_transcript`
(same shape, same reason, even though its target -- the realtime engine --
stays screen-owned) and the two composer/workspace accessors above, is a
named keyword-only constructor callable now, wired by `ChatScreen.__init__`
as a late-binding lambda, matching this file's `composer_accessor`/`chat_
store_accessor`/`speak_status` from wave 1. See each parameter's own
docstring below for exactly which of these are read-only, write-only
(`set_hands_free_vad_degraded`, since no moved body here ever reads it),
or call-through.

Eleven `ChatScreen` methods are NOT part of this cluster but still read
`_console_dictation_state` (general voice-status/composer-sync/dispatch
code: `_speak_status`, `_console_read_last_response_back`, `_sync_console_
composer_action_state`, `on_button_pressed`, plus seven now living on
`ConsoleHandsFreeController` instead of `ChatScreen` -- `action_toggle_
console_hands_free`, `_teardown_console_hands_free_loop`, `_console_hands_
free_request_stop_and_send`, `_console_hands_free_open_capture`, `_console_
hands_free_close_capture`, `_repaint_console_hands_free_chip`, and
`_deliver_console_hands_free_capture_ended`). None of them are
dictation-shaped -- they own hands-free's FSM, the giant screen-level
button dispatcher, or general composer sync -- so none of them moved.
`ChatScreen` keeps a small set of read/write properties under the original
attribute names (see `ChatScreen`'s own "Dictation state (owned by
`ConsoleDictationController`)" section) so those eleven method bodies also
needed zero edits: the attribute now lives on `self._dictation`, but
`self._console_dictation_state` still reads and writes it transparently.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
import asyncio
import logging
import threading
from typing import Any, Literal, TYPE_CHECKING

from loguru import logger
from rich.markup import escape as escape_markup
from textual.message import Message

# Import-safe at module scope: `console_voice_input` reaches the optional
# speech stack only through `importlib.util.find_spec` and a function-body
# import inside `default_service_factory`, so nothing here drags
# `tldw_chatbook.Audio` (and with it faster-whisper and NeMo) into app
# start.
#
# The module itself is imported (not just the names below) so
# `_sync_console_dictation_availability` can call `console_voice_input.
# probe()` through the module's own namespace at call time -- the same
# target tests already monkeypatch (`_patch_availability` in
# `Tests/UI/test_console_dictation_streaming.py`) to make the controller's
# own internal `probe()` call deterministic. Binding `probe` as a bare
# name here instead would capture the unpatched function at import time
# and silently stop tracking that monkeypatch. Moved verbatim from
# `chat_screen.py`, which no longer references this import for itself.
from ...Chat import console_voice_input
from ...Chat.console_voice_input import (
    NO_CAPTURE_MESSAGE,
    NO_SPEECH_MESSAGE,
    STATE_LISTENING,
    TRANSCRIPTION_INCOMPLETE_REASON,
    TRANSCRIPTION_INCOMPLETE_REMEDY,
    VAD_UNAVAILABLE_MESSAGE,
    ConsoleVoiceInputController,
    VoiceCommand,
    VoiceDictationModelDefaulted,
    VoiceFailed,
    VoiceFinal,
    VoiceLocalSTTBusy,
    VoiceModelPreparing,
    VoiceModelWarmupFailed,
    VoicePartial,
    VoiceProviderOverridden,
    VoiceSegmentNoFinal,
    VoiceSegmentTranscribing,
    VoiceSpeechResumed,
    VoiceVadUnavailable,
    default_service_factory,
)
from ...Chat.console_glyphs import GLYPH_VOICE_WORKING
from ...STT.dispatch_coordinator import DICTATION_MAX_SECONDS, pcm_byte_limit
from ...Utils.persistent_diagnostics import persist_event
from ...Widgets.confirmation_dialog import ConfirmationDialog
from ...Widgets.Console import ConsoleComposerBar
from ...Widgets.glyph_fallback import resolve_glyph

if TYPE_CHECKING:
    from ..Screens.chat_screen import ChatScreen

logger = logger.bind(module="ChatScreen")

CONSOLE_DICTATION_MAX_SECONDS = DICTATION_MAX_SECONDS
#: `AudioRecordingService`'s own capture defaults, restated rather than
#: imported: `tldw_chatbook.Audio` pulls in the transcription stack at module
#: scope, and this module must stay importable without it (see
#: `Tests/UI/test_console_dictation_streaming.py`'s subprocess import guard).
CONSOLE_DICTATION_SAMPLE_RATE = 16_000
CONSOLE_DICTATION_CHANNELS = 1
CONSOLE_DICTATION_SAMPLE_WIDTH = 2
#: The PCM bound handed to the recorder. Without it `AudioRecordingService`
#: retains every chunk for the whole capture -- and so do its undrained
#: `audio_queue` and `LazyLiveDictationService.audio_buffer`, at ~32 KB/s each.
CONSOLE_DICTATION_MAX_BYTES = pcm_byte_limit(
    sample_rate=CONSOLE_DICTATION_SAMPLE_RATE,
    channels=CONSOLE_DICTATION_CHANNELS,
    sample_width=CONSOLE_DICTATION_SAMPLE_WIDTH,
)


class ConsoleDictationEvent(Message):
    """Carry a `console_voice_input` event onto the Console screen's thread.

    The controller emits from whichever thread the recognizer happens to be on.
    `post_message` is the only thread-safe route to the UI here: never
    `call_from_thread`, which blocks its caller, and the caller is the audio
    path.
    """

    def __init__(self, session: Any, event: Any) -> None:
        """Wrap one controller event.

        Args:
            session: The session that emitted it, so the screen can drop
                events from a session it has already discarded.
            event: The `VoicePartial` / `VoiceSegmentTranscribing` /
                `VoiceSpeechResumed` / `VoiceFinal` / `VoiceFailed` /
                `VoiceStateChanged` / `VoiceProviderOverridden` instance.
        """
        super().__init__()
        self.session = session
        self.event = event


class ConsoleDictationLimitSignal(Message):
    """Carry a recorder PCM-bound signal onto the Console screen's thread.

    `AudioRecordingService` invokes its `on_buffer_limit` callback from a
    freshly spawned notification thread, so the same rule as
    `ConsoleDictationEvent` applies: `post_message`, never `call_from_thread`
    (which blocks its caller until the UI thread gets round to it).

    Named to avoid `Message.handler_name`: Textual derives
    `on_<snake_case_class_name>` from this class and dispatches both that and
    its `_on_`-prefixed private twin. A `ConsoleDictationBufferLimit` would
    therefore have been delivered straight into
    `_on_console_dictation_buffer_limit` -- the recorder callback that *posts*
    this message -- with the message itself as the `session` argument, i.e. an
    unbounded message loop (8 GB of RSS in three minutes, measured).
    """

    def __init__(self, session: Any) -> None:
        """Wrap one buffer-limit signal.

        Args:
            session: The session whose recorder hit the bound, so the screen
                can drop a signal from a capture it has already torn down.
        """
        super().__init__()
        self.session = session


#: `VoiceCommand.name` -> the literal break text `ConsoleStreamingDictationSession`
#: appends to `_segments` in its place. These two are the only command names
#: this adapter acts on itself; the rest of `COMMAND_PHRASES` (`stop`, `send`,
#: `discard`, `read-that-back`, `new-session`) end the capture and are Task
#: 3's to route -- this adapter only keeps them out of `_segments` and counts
#: them via `commands_consumed`.
_INLINE_BREAK_COMMANDS: dict[str, str] = {
    "new-paragraph": "\n\n",
    "new-line": "\n",
}


#: Acknowledgements for the voice paths that decline to do what was asked.
#: Each is used verbatim for both the toast and the spoken ack, so the two
#: can never drift into telling the user two different things.
_VOICE_ACK_SESSION_CHANGED = "Session changed — not sent."
_VOICE_ACK_NOT_SENT = "Not sent."
_VOICE_ACK_TOO_LATE_TO_DISCARD = "Too late to discard — text inserted."
_VOICE_ACK_NOTHING_TO_INSERT = "Nothing to insert."


def _voice_command_chip_ack(name: str) -> str:
    """Return the short chip acknowledgement for a recognized voice command.

    A break command has no words to echo, so it acks with the pilcrow every
    editor uses for one. Everything else acks with its own name, de-kebabed
    (`read-that-back` -> "read that back") so the chip reads as the phrase the
    user actually said. Derived rather than tabulated: a future command name
    then gets a sensible ack without a second list to keep in step.

    Args:
        name: `VoiceCommand.name`, as `classify_segment` produced it.

    Returns:
        Chip-sized plain text. Never markup -- the chip writes through
        `Content` -- and never empty, so an ack is always visible.
    """
    if name in _INLINE_BREAK_COMMANDS:
        return "¶"
    return name.replace("-", " ")

def _join_segments(segments: list[str]) -> str:
    """Join transcript segments with single spaces, without padding breaks.

    A plain `" ".join(segments)` would sandwich an inline `new-paragraph`/
    `new-line` break entry between two spaces -- `"para. \\n\\n para"` --
    which reads as a blank line with a stray leading and trailing space
    rather than a clean paragraph break. A break entry is recognized as
    exactly `"\\n"` or `"\\n\\n"` (the only values
    `ConsoleStreamingDictationSession` ever appends for an inline command)
    and is concatenated directly, trimming any trailing space the previous
    segment left behind first.

    Args:
        segments: Finalized transcript text and inline-command break entries,
            in the order the recognizer produced them.

    Returns:
        The joined transcript: dictated segments separated by single spaces,
        breaks concatenated without surrounding padding.
    """
    out = ""
    for segment in segments:
        if segment in ("\n", "\n\n"):
            out = out.rstrip(" ") + segment
        elif out and not out.endswith((" ", "\n")):
            out += " " + segment
        else:
            out += segment
    return out

class ConsoleStreamingDictationSession:
    """Drive `ConsoleVoiceInputController` through the one-shot session port.

    The Console screen owns dictation as three blocking calls -- `start()`,
    `stop_and_transcribe()` and `discard()` -- each already run off the UI
    thread by `asyncio.to_thread`, with the visible button transitions applied
    around them. Keeping that port intact is what lets the streaming backend
    replace the one-shot recorder without changing a single observable
    transition:

    ============================  ==================  =========================
    Controller state              Button state        Applied by
    ============================  ==================  =========================
    ``preparing``                 ``starting``        before ``start()`` runs
    ``listening``                 ``recording``       when ``start()`` returns
    ``finishing``                 ``transcribing``    before ``stop_and_transcribe()``
    ``idle``                      ``idle``            when it returns
    ============================  ==================  =========================

    `spawn` is therefore inline: this object is *already* on a worker thread,
    and the controller's blocking halves must complete before the call it
    stands behind returns.

    Live events still flow the moment they happen -- partials and per-segment
    finals go straight to `on_event` -- but the finals are accumulated here so
    `stop_and_transcribe()` returns one transcript at the instant the
    controller reaches `idle`. That preserves the shipping insertion contract:
    the draft is written once, at the caret, and never mid-capture.

    A finalized segment that matched the spoken-command grammar arrives as a
    `VoiceCommand` instead of a `VoiceFinal` (see `classify_segment`). Two
    command names -- `new-paragraph` and `new-line` -- are this adapter's to
    act on: they become break entries in `_segments` (see `_join_segments`),
    never dictated text. Every other command name ends the capture and is
    Task 3's to route; this class only keeps it out of `_segments`, counts it
    in `commands_consumed`, and forwards it to `on_event` unchanged, exactly
    like any other event.
    """

    def __init__(
        self,
        *,
        on_event: Callable[[Any, Any], None],
        service_factory: Callable[..., Any] = default_service_factory,
        max_buffer_bytes: int | None = CONSOLE_DICTATION_MAX_BYTES,
    ) -> None:
        """Build a session over a fresh controller.

        Args:
            on_event: Called with `(session, event)` for every controller
                event, from whatever thread emitted it.
            service_factory: Builds the dictation service; injected by tests.
            max_buffer_bytes: Hard cap on the PCM the recorder retains for one
                capture, passed through to the dictation service. `None`
                leaves the recorder unbounded (the service default, which the
                non-Console dictation callers still use).
        """
        self._on_event = on_event
        self._lock = threading.Lock()
        self._segments: list[str] = []
        #: Finalized commands seen this capture -- inline (`new-paragraph`,
        #: `new-line`) and capture-ending alike. Read by `stop_and_transcribe`
        #: to tell a capture that was only ever spoken commands apart from a
        #: genuinely silent one, since both join down to an empty (or
        #: whitespace-only) transcript.
        self.commands_consumed: int = 0
        self._failure = ""
        self._in_blocking_call = False
        self._heard_recognizer_output = False
        # Bumped by every `start()`. This session object is reused across
        # captures (only a failure or an explicit cancel drops it -- see
        # `_notify_console_dictation_error`/`_request_console_dictation_cancel`
        # on the screen), so when `stop_and_transcribe()`'s join times out
        # (point 3 below) the orphaned processing thread's tail flush is
        # still bound to the SAME controller and the SAME `_handle_event`.
        # `_handle_event` compares the generation the event's callback closure
        # captured at wiring time (see `ConsoleVoiceInputController.start`'s
        # `capture_generation` parameter) against this counter's CURRENT
        # value and drops anything that no longer matches, before it can
        # mutate `_segments`/`commands_consumed` or reach the screen.
        self._capture_generation: int = 0
        self._service_factory = service_factory
        self._max_buffer_bytes = max_buffer_bytes
        self._on_buffer_limit: Callable[[], None] | None = None
        self._controller = ConsoleVoiceInputController(
            emit=self._handle_event,
            spawn=lambda thunk: thunk(),
            # Not `service_factory` directly: the controller builds the service
            # deep inside `start()`, long after the caller handed us its
            # buffer-limit callback, so the bound has to be attached here.
            service_factory=self._build_service,
        )

    def _build_service(self, **kwargs: Any) -> Any:
        """Build the dictation service with this session's PCM bound attached.

        Args:
            **kwargs: Provider, model and language keywords chosen by the
                controller. Passed through untouched.

        Returns:
            The dictation service the controller will drive.
        """
        if self._max_buffer_bytes is not None:
            kwargs.setdefault("max_buffer_bytes", self._max_buffer_bytes)
        if self._on_buffer_limit is not None:
            kwargs.setdefault("on_buffer_limit", self._on_buffer_limit)
        return self._service_factory(**kwargs)

    def _handle_event(self, event: Any, generation: int | None = None) -> None:
        """Record what the screen cannot see, then forward. Never raises.

        A raise here would land in the recognizer's callback -- or, for a
        `VoiceFailed`, inside the controller's own `_fail()`, whose raising-emit
        handling exists precisely so a plumbing error cannot bury the real
        cause. Neither is a place to propagate from.

        Args:
            event: The controller event being emitted.
            generation: The capture generation `ConsoleVoiceInputController`
                bound into this event's callback closure at wiring time (see
                `start()`'s `capture_generation` parameter and `_run_begin()`
                in `console_voice_input.py`). `None` for events that are
                always emitted synchronously within the current capture's own
                blocking call (state changes, advisory notices) and therefore
                need no check. A mismatch against `self._capture_generation`
                means an orphaned processing thread from a capture
                `stop_and_transcribe()` already gave up joining (see its
                docstring, point 3) has only now delivered its tail flush --
                after a LATER capture reused this same session object and
                bumped the generation. The event is dropped before it can
                mutate `_segments`/`commands_consumed` or reach the screen;
                this is what closes the race for all four variants a stale
                delivery can take: command, final, partial, and failed.
        """
        try:
            if generation is not None and generation != self._capture_generation:
                logger.debug(
                    "Dropping stale console dictation event (generation {}, "
                    "current generation {})",
                    generation,
                    self._capture_generation,
                )
                return
            forward = True
            if isinstance(event, VoiceFinal):
                text = event.text.strip()
                with self._lock:
                    # A final that strips to nothing still proves the
                    # recognizer ran and produced output; only its text is
                    # discarded. `stop_and_transcribe()` needs that distinction
                    # to pick between the two silent-capture messages.
                    self._heard_recognizer_output = True
                    if text:
                        self._segments.append(text)
            elif isinstance(event, VoiceCommand):
                # A command proves the recognizer ran, same as a `VoiceFinal`
                # above. `_INLINE_BREAK_COMMANDS` is this adapter's own to
                # act on -- its break text joins `_segments` in place of
                # dictated text. Every other command name (the
                # capture-ending ones `stop`/`send`/`discard`/
                # `read-that-back`/`new-session`) is deliberately left out of
                # `_segments` and just forwarded below, unchanged, for the
                # screen to route in Task 3.
                break_text = _INLINE_BREAK_COMMANDS.get(event.name)
                with self._lock:
                    self._heard_recognizer_output = True
                    if break_text is not None:
                        self._segments.append(break_text)
                    self.commands_consumed += 1
            elif isinstance(event, VoicePartial):
                if event.text.strip():
                    with self._lock:
                        self._heard_recognizer_output = True
            elif isinstance(event, VoiceFailed):
                with self._lock:
                    self._failure = (f"{event.reason} {event.remedy}").strip()
                    # A blocking call is in flight, so it is about to raise
                    # this failure and the screen's existing error path will
                    # report it. Forwarding it as well would notify the user
                    # about one failure twice. What does need forwarding is
                    # the other kind: the recognizer dying mid-capture, with
                    # nothing blocked on it to carry the news.
                    forward = not self._in_blocking_call
            if forward:
                self._on_event(self, event)
        except Exception:  # noqa: BLE001 - the audio path must never see this
            logger.opt(exception=True).warning(
                "Console dictation event could not be delivered"
            )

    def _take_failure(self) -> str:
        with self._lock:
            failure, self._failure = self._failure, ""
        return failure

    @contextmanager
    def _blocking_call(self) -> Iterator[None]:
        """Mark the window in which a failure will be raised, not forwarded."""
        with self._lock:
            self._in_blocking_call = True
        try:
            yield
        finally:
            with self._lock:
                self._in_blocking_call = False

    def start(self, *, on_buffer_limit: Callable[[], None] | None = None) -> None:
        """Open the microphone, blocking until it is live or has failed.

        Args:
            on_buffer_limit: Invoked once, from a recorder notification thread,
                if the capture reaches `max_buffer_bytes`. Wired through to
                `AudioRecordingService`, which stops taking audio at the bound.
                Streaming does *not* make the PCM go away: the recorder's own
                `audio_buffer`, its undrained `audio_queue`, and
                `LazyLiveDictationService.audio_buffer` all grow for the whole
                capture at ~32 KB/s each, and the screen's wall-clock timer is
                the only other thing bounding them.

        Raises:
            RuntimeError: The controller refused or could not start capture.
        """
        self._on_buffer_limit = on_buffer_limit
        with self._lock:
            self._segments.clear()
            self.commands_consumed = 0
            self._failure = ""
            self._heard_recognizer_output = False
            self._capture_generation += 1
            generation = self._capture_generation
        with self._blocking_call():
            self._controller.start(capture_generation=generation)
        failure = self._take_failure()
        if failure:
            raise RuntimeError(failure)
        if self._controller.state != STATE_LISTENING:
            # `start()` was ignored -- the controller was abandoned, or a
            # previous capture never returned it to idle.
            raise RuntimeError("Microphone dictation could not be started.")

    def stop_and_transcribe(self) -> str:
        """Close the microphone and return every segment finalized so far.

        Blocks until the controller reaches `idle`, so the screen inserts
        exactly once, with the whole transcript, at that moment.

        Never returns an empty transcript for an ordinary capture, matching
        the one-shot backend this replaced (`Audio/console_dictation.py`): it
        raised rather than hand back nothing, and the screen's insertion has
        no empty case for dictated text -- an empty transcript still pads to
        a stray space at the caret, silently, and gets persisted to the
        session draft. The one deliberate exception is a capture made of
        nothing but spoken commands (`self.commands_consumed > 0` -- e.g.
        "Console, new paragraph." alone, or "Console, stop." with nothing
        dictated first): the segments still join down to `""` or pure
        whitespace, but that is not a silent microphone, so this returns
        `""` rather than raising it as one. The caller must treat that empty
        return as "nothing to insert," not as the empty case above.

        An empty transcript with no commands consumed has three genuinely
        different causes, and this reports each as itself rather than
        blaming the microphone for all three:

        1. The recorder delivered no bytes -- a real capture or permission
           problem. Keeps the one-shot backend's wording verbatim.
        2. Bytes arrived but nothing was recognized -- also that backend's
           wording, verbatim.
        3. The service's processing thread was still transcribing when its
           join expired, so audio was dropped unread. Nothing here is a
           statement about the microphone, which worked fine.

        The recorder's byte count comes from the service
        (`CaptureOutcome.captured_bytes`), not from guessing at the transcript.
        When a service does not report it -- test fakes, older services -- the
        recognizer-output flag is the fallback, exactly as before.

        Returns:
            The accumulated segments, joined by `_join_segments` (so an
            inline command's break lands unpadded). Empty only when the
            capture consisted solely of spoken commands; an ordinary
            dictated capture is never empty.

        Raises:
            RuntimeError: The controller failed while finishing, or the
                capture was genuinely silent (`self.commands_consumed == 0`
                and nothing was transcribed, or the transcription never
                completed).
        """
        with self._blocking_call():
            self._controller.stop()
        failure = self._take_failure()
        if failure:
            raise RuntimeError(failure)
        with self._lock:
            transcript = _join_segments(self._segments)
            heard = self._heard_recognizer_output
            commands_consumed = self.commands_consumed
        if transcript.strip():
            return transcript
        if commands_consumed > 0:
            return ""
        outcome = self._controller.last_capture_outcome
        if not outcome.transcription_complete:
            raise RuntimeError(
                f"{TRANSCRIPTION_INCOMPLETE_REASON} {TRANSCRIPTION_INCOMPLETE_REMEDY}"
            )
        heard = heard or bool(outcome.captured_bytes)
        raise RuntimeError(NO_SPEECH_MESSAGE if heard else NO_CAPTURE_MESSAGE)

    @property
    def retry_available(self) -> bool:
        """Whether one bounded faster-whisper replay is retained."""

        return self._controller.retry_available

    def clear_retry(self) -> None:
        """Release any retained retry PCM without replaying it."""

        self._controller.clear_retry()

    def retry_with_faster_whisper(self) -> str:
        """Replay logical segments, classify them, and return the full capture."""

        with self._lock:
            generation = self._capture_generation
        logical_texts = self._controller.retry_segments_with_faster_whisper()
        for text in logical_texts:
            self._handle_event(
                console_voice_input.classify_segment(text),
                generation,
            )
        with self._lock:
            return _join_segments(self._segments)

    def discard(self) -> None:
        """Release the microphone without the blocking join.

        Terminal teardown only (unmount, or a failure the screen has already
        surfaced): `abandon()` is one-way for this instance, and the screen
        drops the session on both of those paths.
        """
        self._controller.abandon()
        with self._lock:
            self._segments.clear()
            self.commands_consumed = 0
            self._failure = ""


class ConsoleDictationController:
    """Owns the Console shell's one-shot mic-button dictation lifecycle.

    Wave 1's proof of the controller collaborator kind (see the module
    docstring). Holds every attribute the pre-extraction `ChatScreen`
    named `_console_dictation_*`, plus `_console_pending_voice_action`,
    and every method whose body touched only those and the screen's own
    framework services. `ChatScreen` constructs exactly one of these, in
    `__init__`, and keeps a `self._dictation` reference plus six one-line
    delegations (see the module docstring) for the entry points reached
    from outside this cluster.
    """

    def __init__(
        self,
        screen: "ChatScreen",
        *,
        app_instance: Any,
        composer_accessor: Callable[[], ConsoleComposerBar | None],
        chat_store_accessor: Callable[[], Any],
        speak_status: Callable[[str], None],
        hands_free_session_accessor: Callable[[], Any],
        set_hands_free_vad_degraded: Callable[[bool], None],
        enter_hands_free_loop: Callable[..., None],
        hands_free_force_immediate_send: Callable[[], None],
        deliver_hands_free_capture_ended: Callable[[Any, bool], Any],
        realtime_session_accessor: Callable[[], Any],
        realtime_adopt_transcript: Callable[[str], bool],
        run_pending_voice_action: Callable[[str | None], Any],
        undo_histories_accessor: Callable[[], dict[str, Any]],
        visible_draft_session_id_accessor: Callable[[], str | None],
        dictation_service_factory: Callable[..., Any] = default_service_factory,
    ) -> None:
        """Build the controller and bind everything its moved bodies need.

        Every one of the 20 method bodies below is a byte-for-byte copy of
        the pre-extraction `ChatScreen` method: no internal line was
        edited to retarget a call or an attribute. That is possible
        because this constructor binds every name those bodies reference
        that is not this controller's own state, under the SAME name the
        original method used.

        Two kinds of binding, one rule each (a wave-1 third kind, "reach
        through `screen` because the target has no controller of its own
        yet," is retired as of wave 2 -- see the module docstring):

        1. **Framework services** (`run_worker`, `post_message`,
           `set_timer`, `set_interval`, `is_mounted`) live-read from
           `screen` via `@property` on every access -- never snapshotted.
           A value captured once at construction (a bound method is just
           a fixed reference to a function plus the `self` it closes
           over) goes stale the instant a test replaces the attribute on
           the SCREEN INSTANCE afterward, confirmed for `set_timer`/
           `set_interval` by this cluster's own test suite.
        2. **Everything else dictation genuinely depends on that is not
           its own state** is a NAMED constructor dependency, matching the
           design spec's rule that "a controller's dependencies are its
           signature": discoverable by reading the constructor, not by
           reading every `@property` on the class. Each is a callable the
           CALLER constructs to close over the screen's own attribute
           lookup at CALL time, not at construction time -- see
           `ChatScreen.__init__`'s `lambda: self._console_composer_or_
           none()` (not `composer_accessor=self._console_composer_or_
           none`, which would freeze the bound method the same way
           `run_worker` used to) -- so the same staleness guarantee kind 1
           gets from a property, kind 2 gets from the lambda's own late
           attribute lookup. The controller's own properties below are
           thin wrappers around these stored callables, kept under the
           ORIGINAL attribute/method names so every one of the 20 moved
           method bodies still reads/calls `self._console_composer_or_
           none()`, `self._console_hands_free`, etc. unchanged. Most
           return the callable itself, for bodies that already called it
           with `()`; `_console_undo_histories`/`_console_visible_draft_
           session_id` were bare-attribute reads in the pre-move source,
           so those two properties CALL the stored accessor internally and
           return the value instead.

        `app_instance` is the one plain-attribute exception to both rules
        above: see its own line below for why.

        Args:
            screen: The Console screen. Used ONLY for the framework
                services in binding kind 1 above. None of this is
                `query_one` traffic -- dictation owns no DOM of its own,
                so there is no region boundary for it to cross.
            app_instance: For `notify()`, posting TTS events, and the
                once-per-app-run notification latches the moved event
                handler stores on it (`_console_dictation_override_
                notified` and its two siblings) -- unchanged from how the
                pre-extraction methods used `self.app_instance`. Snapshotted
                as a plain attribute, not a property, and correctly so:
                `app_instance` does not change identity over the
                controller's life the way a screen METHOD can be replaced
                on the instance -- the pre-extraction methods already read
                it as a plain attribute (`self.app_instance.notify(...)`),
                never as a call, so there is no late-binding hazard here
                to guard against.
            composer_accessor: A general screen helper (33 call sites
                across the whole screen, not dictation-specific) --
                `ChatScreen._console_composer_or_none`. Passed as a
                late-binding lambda (see above), never the bound method
                directly.
            chat_store_accessor: Same rationale as `composer_accessor` --
                `ChatScreen._ensure_console_chat_store`, 65 call sites
                screen-wide.
            speak_status: `ChatScreen._speak_status`, shared by two
                non-dictation voice paths too (hands-free's queued-send
                ack, "read that back"), so it stays screen-owned;
                dictation only calls it.
            hands_free_session_accessor: Reads `ConsoleHandsFreeController.
                _console_hands_free`, the pipeline loop's live session (or
                None). Wave 1's disclosed, temporary exception, closed out
                now that hands-free has its own controller (wave-2 console
                decomposition, task 1) -- see the module docstring.
            set_hands_free_vad_degraded: Writes `ConsoleHandsFreeController.
                _console_hands_free_vad_degraded`. Setter-only: only
                `_handle_console_dictation_event`'s `VoiceVadUnavailable`
                branch touches this attribute, and it only ever assigns to
                it, never reads it.
            enter_hands_free_loop: `ConsoleHandsFreeController._enter_
                console_hands_free_loop`, the engine fork -- reached by a
                spoken "Console, hands free." mid-capture.
            hands_free_force_immediate_send: `ConsoleHandsFreeController.
                _console_hands_free_force_immediate_send`, reached by a
                spoken "send" mid-loop.
            deliver_hands_free_capture_ended: `ConsoleHandsFreeController.
                _deliver_console_hands_free_capture_ended`, scheduled (not
                awaited) by `_handle_console_dictation_limit` when a
                limit-triggered stop finds nothing captured.
            realtime_adopt_transcript: `ChatScreen._console_realtime_adopt_
                transcript` -- the realtime engine's own entry point. Same
                shape and staleness reason as the five hands-free callables
                above, but its target stays screen-owned (not extracted
                this wave): the realtime loop (V4) owns whether a capture's
                transcript becomes its first spoken turn instead of
                composer text.
            run_pending_voice_action: `ChatScreen._run_pending_console_
                voice_action` -- general screen-orchestration (chat store,
                send button, tab creation, TTS read-back), well beyond a
                controller's "handful of well-known ids," so it stays
                screen-owned; only `_stop_console_dictation`'s tail calls
                it, unconditionally (a no-op when nothing was queued).
                `ConsoleHandsFreeController` has an identically-shaped,
                identically-named dependency pointed at the SAME screen
                method.
            realtime_session_accessor: Reads `ChatScreen._console_
                realtime` -- the V4 realtime loop's live session, or None.
                The realtime engine stays screen-owned (not extracted), so
                this is the same disclosed shape as `realtime_adopt_
                transcript` above and is named exactly as `ConsoleHandsFree
                Controller`'s identical dependency is. Needed by
                `_handle_console_dictation_button` (wave-4 task 2): a
                running realtime session supersedes the mic button's
                one-shot toggle, and falling through would open a SECOND
                recorder alongside the live 24 kHz tap.
            undo_histories_accessor: Reads `ChatScreen._console_undo_
                histories` -- per-session composer undo history, screen-
                owned (not extracted this wave). Returns the actual dict
                (not a copy): `_insert_console_dictation` mutates it in
                place via `.pop(...)`.
            visible_draft_session_id_accessor: Reads `ChatScreen._console_
                visible_draft_session_id` -- the session id the mounted
                composer's draft currently reflects, screen-owned.
        """
        self._screen = screen
        self.app_instance = app_instance
        self._composer_accessor = composer_accessor
        self._chat_store_accessor = chat_store_accessor
        self._speak_status_fn = speak_status
        self._hands_free_session_accessor = hands_free_session_accessor
        self._set_hands_free_vad_degraded_fn = set_hands_free_vad_degraded
        self._enter_hands_free_loop_fn = enter_hands_free_loop
        self._hands_free_force_immediate_send_fn = hands_free_force_immediate_send
        self._deliver_hands_free_capture_ended_fn = deliver_hands_free_capture_ended
        self._realtime_adopt_transcript_fn = realtime_adopt_transcript
        self._realtime_session_accessor = realtime_session_accessor
        self._run_pending_voice_action_fn = run_pending_voice_action
        self._undo_histories_accessor = undo_histories_accessor
        self._visible_draft_session_id_accessor = visible_draft_session_id_accessor
        self._dictation_service_factory = dictation_service_factory

        # Dictation's own state, moved verbatim from `ChatScreen.__init__`.
        self._console_dictation_session: Any | None = None
        self._console_dictation_state: Literal[
            "idle", "starting", "recording", "transcribing"
        ] = "idle"
        self._console_dictation_timer: Any | None = None
        self._console_dictation_elapsed_timer: Any | None = None
        self._console_dictation_origin_session_id: str | None = None
        self._console_dictation_partial = ""
        self._console_pending_voice_action: str | None = None
        self._console_dictation_late_discard_ack = False

    @property
    def is_mounted(self) -> bool:
        """Whether the Console screen is currently mounted.

        A live check: mount state changes over the controller's life, so
        this has to re-read `screen.is_mounted` on every access.
        """
        return self._screen.is_mounted

    @property
    def run_worker(self) -> Any:
        """`Screen.run_worker`, bound. See `__init__`'s docstring for why
        this is a property rather than a value snapshotted once."""
        return self._screen.run_worker

    @property
    def post_message(self) -> Any:
        """`Screen.post_message`, bound. See `__init__`'s docstring."""
        return self._screen.post_message

    @property
    def set_timer(self) -> Any:
        """`Screen.set_timer`, bound. See `__init__`'s docstring."""
        return self._screen.set_timer

    @property
    def set_interval(self) -> Any:
        """`Screen.set_interval`, bound. See `__init__`'s docstring."""
        return self._screen.set_interval

    @property
    def _console_composer_or_none(self) -> Any:
        """The injected `composer_accessor`. Kept under this name so the
        20 moved method bodies below still call `self._console_composer_
        or_none()` unchanged. See `__init__`'s docstring (binding kind 3)."""
        return self._composer_accessor

    @property
    def _ensure_console_chat_store(self) -> Any:
        """The injected `chat_store_accessor`. See `_console_composer_or_
        none`'s docstring immediately above."""
        return self._chat_store_accessor

    @property
    def _speak_status(self) -> Any:
        """The injected `speak_status`. See `_console_composer_or_none`'s
        docstring above."""
        return self._speak_status_fn

    @property
    def _enter_console_hands_free_loop(self) -> Any:
        """The injected `enter_hands_free_loop`. See `__init__`'s docstring."""
        return self._enter_hands_free_loop_fn

    @property
    def _console_hands_free_force_immediate_send(self) -> Any:
        """The injected `hands_free_force_immediate_send`. See `__init__`'s
        docstring."""
        return self._hands_free_force_immediate_send_fn

    @property
    def _deliver_console_hands_free_capture_ended(self) -> Any:
        """The injected `deliver_hands_free_capture_ended`. See `__init__`'s
        docstring."""
        return self._deliver_hands_free_capture_ended_fn

    @property
    def _console_realtime_adopt_transcript(self) -> Any:
        """The injected `realtime_adopt_transcript`. See `__init__`'s
        docstring: the realtime loop (V4) owns whether a capture's
        transcript becomes its first spoken turn instead of composer text.
        """
        return self._realtime_adopt_transcript_fn

    @property
    def _run_pending_console_voice_action(self) -> Any:
        """The injected `run_pending_voice_action`. See `__init__`'s
        docstring: `ChatScreen`'s own, not dictation's -- its body reaches
        the chat store, the send button, tab creation, and TTS read-back,
        well beyond a controller's "handful of well-known ids." Only
        `_stop_console_dictation`'s tail calls it, unconditionally (a
        no-op when nothing was queued).
        """
        return self._run_pending_voice_action_fn

    @property
    def _console_hands_free(self) -> Any:
        """Calls the injected `hands_free_session_accessor`. The pipeline
        loop's live session, or None. See `__init__`'s docstring."""
        return self._hands_free_session_accessor()

    @property
    def _console_realtime(self) -> Any:
        """Calls the injected `realtime_session_accessor`. The realtime
        (V4) loop's live session, or None. See `__init__`'s docstring."""
        return self._realtime_session_accessor()

    @property
    def _console_hands_free_vad_degraded(self) -> bool:
        """Write-only: see `__init__`'s docstring for
        `set_hands_free_vad_degraded`.

        Raises `RuntimeError`, deliberately NOT `AttributeError`: this is a
        property, and `hasattr()`/`getattr(obj, name, default)` swallow
        `AttributeError` specifically. A defensive
        `getattr(self._dictation, "_console_hands_free_vad_degraded", False)`
        would then read False forever regardless of the real flag, with no
        error ever surfacing. `RuntimeError` propagates instead.
        """
        raise RuntimeError(
            "_console_hands_free_vad_degraded is write-only on "
            "ConsoleDictationController"
        )

    @_console_hands_free_vad_degraded.setter
    def _console_hands_free_vad_degraded(self, value: bool) -> None:
        self._set_hands_free_vad_degraded_fn(value)

    @property
    def _console_undo_histories(self) -> Any:
        """Calls the injected `undo_histories_accessor`. See `__init__`'s
        docstring: per-session composer undo history, screen-owned (not
        extracted this wave)."""
        return self._undo_histories_accessor()

    @property
    def _console_visible_draft_session_id(self) -> Any:
        """Calls the injected `visible_draft_session_id_accessor`. See
        `__init__`'s docstring."""
        return self._visible_draft_session_id_accessor()

    async def teardown(self) -> None:
        """Release dictation's own resources during screen unmount.

        Moved out of `ChatScreen.on_unmount` (wave-1 console
        decomposition, task 5), where this was seven inline statements;
        `on_unmount` now calls this as one line. Abandon teardown, not a
        graceful exit -- matches the original block exactly: cancel both
        timers, drop the session and its origin/partial bookkeeping,
        force `idle`, then discard whatever session was live, off the UI
        thread.
        """
        self._cancel_console_dictation_timer()
        self._cancel_console_dictation_elapsed_timer()
        dictation_session = self._console_dictation_session
        self._console_dictation_session = None
        self._console_dictation_origin_session_id = None
        self._console_dictation_partial = ""
        self._console_dictation_state = "idle"
        if dictation_session is not None:
            await asyncio.to_thread(dictation_session.discard)

    def _set_console_dictation_state(
        self,
        state: Literal["idle", "starting", "recording", "transcribing"],
    ) -> None:
        """Set the one-shot dictation state and refresh its visible control."""
        self._console_dictation_state = state
        composer = self._console_composer_or_none()
        if composer is not None:
            composer.sync_dictation_state(state)


    def _sync_console_dictation_availability(self) -> None:
        """Refresh the mic button's tooltip from a fresh availability probe.

        Called once after mount, so the button's initial tooltip is accurate
        without waiting for a first press, and again at the top of every
        activation attempt (`_request_console_dictation_start`), so installing
        the missing extra or plugging in a microphone mid-run is picked up
        without a screen remount. `probe()` only calls
        `importlib.util.find_spec`, so it is cheap and safe to call
        repeatedly.

        This is the cosmetic half only -- it never blocks starting dictation.
        A real attempt still goes through `ConsoleVoiceInputController.start()`,
        which re-probes on its own and fails visibly
        (`_notify_console_dictation_error`) if unavailable. Gating here too
        would double-report the same failure, and -- because a genuinely
        Textual-`disabled` button can never be pressed again afterward
        (Click is never delivered to a disabled widget, so pressing could
        never recover it) -- this stays purely cosmetic by design; see
        `ConsoleComposerBar.sync_dictation_state`.

        A probe crash is swallowed and treated as available, so a bug in the
        probe cannot brick the button in a permanently unavailable-looking
        state with no way to recover.
        """
        composer = self._console_composer_or_none()
        if composer is None:
            return
        try:
            availability = console_voice_input.probe()
        except Exception:  # noqa: BLE001 - a probe crash must not disable the button
            logger.opt(exception=True).warning(
                "Console dictation availability probe crashed"
            )
            composer.set_dictation_availability(available=True)
            return
        # `remedy` alone is a complete, self-explanatory sentence for both
        # kinds (it already opens by restating `reason` -- see CAPTURE_REMEDY
        # / PROVIDER_REMEDY in console_voice_input.py); joining `reason` ahead
        # of it here would stutter ("No ... installed. No ... installed.
        # Install with: ..."). The `reason + remedy` join used for the
        # VoiceFailed toast is a one-off event, not idle copy shown on every
        # glance at the button, so it can afford the redundancy this can't.
        composer.set_dictation_availability(
            available=availability.ok,
            tooltip="" if availability.ok else availability.remedy,
        )


    def _cancel_console_dictation_timer(self) -> None:
        timer = self._console_dictation_timer
        self._console_dictation_timer = None
        if timer is not None:
            timer.stop()


    def _cancel_console_dictation_elapsed_timer(self) -> None:
        """Stop the chip's 1 s elapsed-time ticker if one is running."""
        timer = self._console_dictation_elapsed_timer
        self._console_dictation_elapsed_timer = None
        if timer is not None:
            timer.stop()


    def _tick_console_dictation_elapsed(self) -> None:
        """Advance the voice chip's elapsed-time display by one second."""
        composer = self._console_composer_or_none()
        if composer is not None:
            composer.tick_voice_elapsed()


    def _notify_console_dictation_error(self, exc: Exception) -> None:
        """Return dictation to idle and show its actionable failure."""
        self._cancel_console_dictation_timer()
        self._cancel_console_dictation_elapsed_timer()
        self._console_dictation_origin_session_id = None
        self._console_dictation_session = None
        self._console_dictation_partial = ""
        # A spoken "send"/"new session"/"read that back" queued its action
        # for after a successful stop -- this is every failure exit, so the
        # queued action must be dropped here rather than surviving to fire
        # on whatever capture succeeds next.
        self._console_pending_voice_action = None
        # Likewise a deferred late-discard ack: the failure reason below is
        # the better thing to say, and two spoken acks for one capture is
        # worse than either alone.
        self._console_dictation_late_discard_ack = False
        self._set_console_dictation_state("idle")
        reason = f"Dictation failed: {exc}"
        # Persist the failure: this branch's runtime log is admission-filtered
        # to `tldw_chatbook.diagnostics.*`, so a toast the user reads was the
        # ONLY record a dictation failure left anywhere -- four live-gate
        # rounds were spent reconstructing failures from paraphrased toasts
        # (2026-08-01). `exc_type` is the class name only; the message can
        # carry user paths or audio filenames and is deliberately not
        # persisted here (the loguru line below keeps it for a verbose run).
        try:
            persist_event(
                "dictation",
                "dictation_failed",
                level=logging.ERROR,
                exception_type=type(exc).__name__,
            )
        except Exception:  # noqa: BLE001 - diagnostics must never break dictation
            logger.opt(exception=True).debug("Could not persist dictation failure")
        logger.warning("Console dictation failed: {}", exc)
        self.app_instance.notify(reason, severity="error")
        self._speak_status(reason)


    def _emit_console_dictation_event(self, session: Any, event: Any) -> None:
        """Hand a controller event to the UI thread. Safe from any thread.

        Args:
            session: The dictation session that emitted the event.
            event: The `console_voice_input` event instance.
        """
        try:
            self.post_message(ConsoleDictationEvent(session, event))
        except Exception:  # noqa: BLE001 - the audio path must never see this
            logger.opt(exception=True).debug(
                "Console dictation event could not be posted"
            )

    def _handle_console_dictation_event(self, message: ConsoleDictationEvent) -> None:
        """Apply the streaming events the blocking session port cannot express.

        Button state stays owned by `_start_console_dictation` /
        `_stop_console_dictation`, which bracket the blocking calls: they are
        the only places that can order an `idle` transition *after* the
        transcript has been inserted. What only the event stream can deliver
        is a partial (chip-only) and a failure that arrives mid-capture, with
        no blocking call in flight to raise it.

        Args:
            message: The posted controller event.
        """
        if message.session is not self._console_dictation_session:
            # A session the screen has already discarded; its events are stale.
            return
        event = message.event
        if isinstance(event, VoicePartial):
            # Only while the microphone is live. A successful capture keeps its
            # session (only failures drop it), so the staleness check above
            # cannot catch the partial the recognizer flushes as
            # `stop_dictation()` joins -- that one drains after the state is
            # already `idle` and would leave a ghost in the chip.
            if self._console_dictation_state == "recording":
                self._console_dictation_partial = event.text
                composer = self._console_composer_or_none()
                if composer is not None:
                    composer.set_voice_partial(event.text)
            return
        if isinstance(event, VoiceSegmentTranscribing):
            # The silence gate closed a segment and its (potentially
            # seconds-long) transcription just started (`event.done` False)
            # or just ended (`event.done` True) -- otherwise zero signal
            # under the segment-at-silence architecture. Same staleness
            # guard as `VoicePartial` just above: only while THIS capture is
            # still actually recording. The indication reverts on whichever
            # comes first: this event's own `done=True` (review finding M1 --
            # a segment that transcribes to blank fires neither a final nor a
            # command, so this is sometimes the ONLY revert signal a capture
            # ever gets), `set_voice_partial` (called by both the
            # `VoiceFinal` and `VoiceCommand` branches below), or any
            # `sync_dictation_state` lifecycle transition.
            if self._console_dictation_state == "recording":
                composer = self._console_composer_or_none()
                if composer is not None:
                    composer.set_voice_segment_transcribing(not event.done)
            return
        if isinstance(event, VoiceSegmentNoFinal):
            # Qodo review (task-5 follow-up): positive proof no `VoiceFinal`
            # is coming for this segment -- fired right after this same
            # segment's own `VoiceSegmentTranscribing(done=True)` above, on
            # the blank/whitespace-only branch. Same `_console_dictation_
            # state == "recording"` same-capture guard as `VoiceFinal` below
            # (not a second source of truth): a stale signal from an already-
            # ended capture must not touch a later turn's resume latch.
            # Meaningless outside the hands-free loop -- see `HandsFree
            # Controller.on_segment_no_final`'s docstring for why this must
            # exist at all (a resume latch armed for a blank segment would
            # otherwise never be consumed, and would incorrectly swallow the
            # NEXT real segment's `VoiceFinal`/countdown).
            if self._console_dictation_state == "recording" and self._console_hands_free is not None:
                self._console_hands_free.controller.on_segment_no_final()
            return
        if isinstance(event, VoiceFinal):
            # The segment is committed; the partial that previewed it is spent.
            self._console_dictation_partial = ""
            composer = self._console_composer_or_none()
            if composer is not None:
                composer.set_voice_partial("")
            # Task 5: this is what arms the hands-free countdown -- same
            # `_console_dictation_state == "recording"` same-capture guard
            # as `VoicePartial` above (not a second source of truth): a
            # final that drains after THIS capture already ended (e.g. the
            # wall-clock/buffer limit beat the recognizer to it) must not
            # arm a countdown for a turn that is no longer live.
            if self._console_dictation_state == "recording" and self._console_hands_free is not None:
                self._console_hands_free.controller.on_voice_final()
            return
        if isinstance(event, VoiceCommand):
            # `new-paragraph`/`new-line` DO reach here too -- the adapter's
            # `forward` default is True for every `VoiceCommand`, inline or
            # capture-ending alike; `_INLINE_BREAK_COMMANDS` only keeps them
            # out of `_segments` (Task 2), it does not stop them being
            # forwarded. The `if`/`elif` chain below stays an explicit
            # allowlist rather than a dict lookup or a trailing `else` --
            # deliberately, so an inline name (or any future, not-yet-routed
            # command name) falls through as a no-op instead of a future
            # maintainer's catch-all accidentally acting on it.
            #
            # Review fix round 1 (Finding 1): a command draining after ITS
            # OWN capture already returned to `idle` is NOT caught by the
            # session-identity check above -- `_start_console_dictation`
            # reuses `self._console_dictation_session` whenever it is still
            # set, which it is after every ordinary successful stop (only a
            # failure or an explicit cancel nulls it). A late command from
            # capture 1 therefore carries the exact same session object as a
            # live capture 2, and would otherwise queue an action (or fire
            # stop/discard) against whichever capture is live NOW, not the
            # one that actually spoke it. `transcribing` stays admitted
            # alongside `recording`: a genuinely-current command that only
            # finalizes once ITS OWN capture's stop-and-transcribe is
            # already running (e.g. the wall timer beat the recognizer to
            # it) is not stale, and must still reach `_stop_console_dictation`'s
            # success tail.
            if self._console_dictation_state not in ("recording", "transcribing"):
                return
            # The command is itself a finalized segment, so its own utterance
            # ("console send") is the last thing sitting in the chip -- but
            # clearing it to empty leaves an inline command with no feedback
            # whatsoever, and the chip acknowledgement is the stated
            # mitigation for the accepted staccato false-fire (a fired
            # command has to be *visible* to be caught). Overwrite it with a
            # short ack instead: the next partial replaces it, and the chip
            # collapses at capture end. Written here, before dispatch below
            # changes the dictation state out from under `set_voice_partial`'s
            # own recording-only guard.
            ack = _voice_command_chip_ack(event.name)
            self._console_dictation_partial = ack
            composer = self._console_composer_or_none()
            if composer is not None:
                composer.set_voice_partial(ack)
            if event.name == "stop":
                if self._console_hands_free is not None:
                    # Hands-free's own exit: `on_exit_request()`'s `ExitLoop`
                    # handler stops the capture itself (via `CloseCapture`)
                    # AND tears the loop down -- a plain
                    # `_request_console_dictation_stop()` here would only
                    # do the first half, leaving the FSM believing it is
                    # still running.
                    self._console_hands_free.controller.on_exit_request()
                else:
                    self._request_console_dictation_stop()
            elif event.name == "discard":
                self._request_console_dictation_cancel()
                # Task-5 review I3: `_request_console_dictation_cancel()`
                # sets `_console_dictation_state` to `idle` SYNCHRONOUSLY
                # (only the actual device release is async), so by the time
                # `on_exit_request()` runs here `_console_hands_free_close_
                # capture`'s own `== "recording"` guard is already False --
                # a clean, single no-op close, not a second stop. Without
                # this the FSM stayed `listening` believing the mic was
                # still open while it was not, with nothing left to reopen
                # or exit it.
                if self._console_hands_free is not None:
                    self._console_hands_free.controller.on_exit_request()
            elif event.name == "hands-free":
                # Task 5: unlike the capture-ending commands below, this one
                # does NOT end the capture -- the still-open mic becomes the
                # loop's first turn (`capture_live=True`), matching the key
                # binding pressed mid-capture.
                self._enter_console_hands_free_loop(capture_live=True)
            elif event.name == "send" and self._console_hands_free is not None:
                # Task-5 review I3: spoken "send" mid-loop must drive the
                # SAME `RequestStopAndSend` semantics as a countdown expiry
                # -- `awaiting_reply` is entered and the reply is actually
                # spoken -- rather than the plain queue-and-stop below,
                # which ended the capture without telling the controller:
                # the FSM stayed `listening` (mic believed open, actually
                # closed) while the reply streamed and the tap silently
                # dropped every delta (`state != "awaiting_reply"`) -- the
                # user asked hands-free to send, it sent, and the loop went
                # quiet forever.
                controller = self._console_hands_free.controller
                if controller.state in ("listening", "countdown"):
                    self._console_hands_free_force_immediate_send()
                else:
                    # Task-5 review round 2, D3: reachable only in
                    # acoustic mode (the only mode with the mic open
                    # mid-reply) -- `_console_hands_free_force_immediate_
                    # send`'s own guard correctly refuses here (a send
                    # cannot happen while a reply is already outstanding;
                    # it would interleave two turns, which this FSM's
                    # single-outstanding-reply model cannot represent),
                    # but silently doing NOTHING -- not even ending the
                    # capture -- was the actual defect: the chip acked a
                    # command that then had no effect at all. End the
                    # capture (whatever was said still lands in the
                    # draft, same as `_request_console_dictation_stop`
                    # does for every other non-discard capture-ender
                    # here) and exit the loop cleanly -- the same honest
                    # choice already made for discard/new-session/
                    # read-that-back just below, for the identical reason
                    # (no FSM input exists for "another send arrived
                    # mid-reply").
                    self._request_console_dictation_stop()
                    controller.on_exit_request()
            elif event.name in ("send", "new-session", "read-that-back"):
                # Queued, not acted on immediately: `_stop_console_dictation`
                # runs it once the capture's own transcript has actually
                # landed (see `_console_pending_voice_action`'s docstring).
                self._console_pending_voice_action = event.name
                self._request_console_dictation_stop()
                # Task-5 review I3: `new-session`/`read-that-back` neither
                # continue nor reopen this turn (a new tab / reading back an
                # already-completed reply are not new conversational turns
                # for THIS loop) -- ending the loop cleanly here is the
                # honest outcome, matching `discard`'s fix immediately
                # above and for the identical reason (the capture just
                # ended out from under the FSM with nothing telling it).
                if self._console_hands_free is not None:
                    self._console_hands_free.controller.on_exit_request()
            return
        if isinstance(event, VoiceSpeechResumed):
            # Task 5: a mic-side fact (see the dataclass's own docstring),
            # forwarded like `VoicePartial` -- generation-gated by the same
            # staleness check at the top of this method, but otherwise
            # meaningless outside the hands-free loop. Gated on the SAME
            # `_console_dictation_state == "recording"` guard `VoicePartial`
            # uses just above (not a second source of truth): a resume that
            # drains after THIS capture already ended must not cancel a
            # countdown or barge in on a reply belonging to a later turn.
            if self._console_dictation_state != "recording":
                return
            if self._console_hands_free is not None:
                self._console_hands_free.controller.on_speech_resumed()
            return
        if isinstance(event, VoiceModelPreparing):
            # The speech model is loading, before the microphone opens. On a
            # fresh machine that is a multi-gigabyte download, and the button
            # sitting on "Mic…" with no explanation is indistinguishable from
            # a hang. Only meaningful while the screen is still starting: a
            # notice that drains late must not repaint a live capture's chip.
            if self._console_dictation_state != "starting":
                return
            composer = self._console_composer_or_none()
            if composer is not None:
                # The composer *holds* this, so an unrelated control-bar
                # refresh cannot wipe it mid-download.
                composer.set_voice_preparing_message(f"{resolve_glyph(GLYPH_VOICE_WORKING)} {event.message}")
            # The chip is 42 cells and one row; the full explanation would be
            # cut mid-sentence there, taking the duration warning with it. Send
            # it somewhere with room, once.
            if event.detail:
                self.app_instance.notify(event.detail, severity="information")
            return
        if isinstance(event, VoiceLocalSTTBusy):
            if self._console_dictation_state != "starting":
                return
            composer = self._console_composer_or_none()
            if composer is not None:
                composer.set_voice_preparing_message(event.message)
            return
        if isinstance(event, VoiceModelWarmupFailed):
            # Advisory, not fatal: the capture is going ahead. Making this
            # fatal would mean one transient error permanently disables
            # dictation, since the Console warms on every press.
            self.app_instance.notify(
                f"{event.reason} {event.remedy}".strip(), severity="warning"
            )
            return
        if isinstance(event, VoiceFailed):
            # Only ever a mid-capture failure: the session forwards a
            # `VoiceFailed` exactly when no blocking call is in flight to
            # raise it, so this cannot double-report a start/stop failure.
            # Handling it here -- ahead of the `VoiceStateChanged(idle)` the
            # controller emits next, which this method deliberately ignores --
            # is what cancels the wall timer and clears the origin session
            # before anything else can run.
            reason = f"{event.reason} {event.remedy}".strip()
            self._notify_console_dictation_error(RuntimeError(reason))
            return
        if isinstance(event, VoiceProviderOverridden):
            # The controller (`ConsoleVoiceInputController._override_announced`)
            # already latches this to once per controller instance, but a
            # fresh controller is built on every new dictation session (e.g.
            # after any failure discards the old one, or on a fresh screen
            # mount -- ChatScreen itself is rebuilt on every Console
            # navigation, never a persistent singleton). The user only needs
            # telling once per app run, not once per capture, so the flag
            # lives on `self.app_instance`, the one object that actually
            # persists for the app's whole run.
            if not getattr(
                self.app_instance, "_console_dictation_override_notified", False
            ):
                self.app_instance._console_dictation_override_notified = True
                # `event.configured` traces back to the user's own
                # `transcription.default_provider` TOML setting -- unvalidated
                # free text, not a value from a closed enum -- and
                # `App.notify` defaults to `markup=True`, so a provider name
                # containing `[...]` must be escaped or it is silently
                # swallowed as (invalid) Rich markup instead of shown.
                self.app_instance.notify(
                    f"Configured dictation provider "
                    f"'{escape_markup(event.configured)}' isn't available; "
                    f"using '{escape_markup(event.effective)}' instead.",
                    severity="warning",
                )
            return
        if isinstance(event, VoiceDictationModelDefaulted):
            # Same two-tier latch as `VoiceProviderOverridden` just above,
            # for the same reason -- see `VoiceDictationModelDefaulted`'s own
            # docstring. This is a deliberate latency policy, not a failure
            # (unlike the provider case), so it stays "information" rather
            # than "warning".
            if not getattr(
                self.app_instance,
                "_console_dictation_model_default_notified",
                False,
            ):
                self.app_instance._console_dictation_model_default_notified = True
                # `event.effective` is `DICTATION_FAST_MODEL_DEFAULT`, a
                # closed constant this code controls -- but `event.configured`
                # traces back to the user's own `transcription.default_model`
                # TOML setting, unvalidated free text, so both go through
                # `escape_markup` for the same reason the provider notice
                # above does.
                self.app_instance.notify(
                    f"Dictation uses the fast '{escape_markup(event.effective)}' "
                    f"model for low latency (configured model: "
                    f"'{escape_markup(event.configured)}') — set dictation.model "
                    f"to change.",
                    severity="information",
                )
            return
        if isinstance(event, VoiceVadUnavailable):
            # Same two-tier latch as `VoiceProviderOverridden` just above,
            # and for the same reason: the controller's own
            # `_vad_unavailable_announced` only covers this one controller
            # instance, and a fresh one is built on every new dictation
            # session. The user only needs telling once per app run. The
            # controller already logged this (see
            # `_maybe_report_vad_unavailable`), so only the toast lives here.
            #
            # Task 5 (VAD-degraded honesty): recorded for this screen's
            # whole life, independent of the once-per-run toast latch above
            # -- `_enter_console_hands_free_loop` reads this to warn, every
            # time the loop starts in degraded mode, that its silence-based
            # auto-send (`VoiceSpeechResumed`/mid-capture `VoiceFinal`
            # never fire without webrtcvad -- see this event's own
            # docstring) will not work; only a manual mic press, spoken
            # "stop", or Esc/ctrl+shift+h will ever end a turn.
            self._console_hands_free_vad_degraded = True
            if not getattr(
                self.app_instance, "_console_dictation_vad_unavailable_notified", False
            ):
                self.app_instance._console_dictation_vad_unavailable_notified = True
                # Not spoken (`spoken_feedback` never applies here): the
                # microphone is open by the time this fires, and speaking
                # over an open mic is exactly what that setting exists to
                # avoid everywhere else in this file.
                self.app_instance.notify(VAD_UNAVAILABLE_MESSAGE, severity="warning")
            return


    def _on_console_dictation_buffer_limit(self, session: Any) -> None:
        """Marshal a recorder-thread memory-limit signal onto the UI thread.

        `post_message`, not `call_from_thread`: this runs on the recorder's
        notification thread, and `call_from_thread` blocks its caller until the
        UI thread services it -- the same rule `ConsoleDictationEvent` exists
        to enforce for the recognizer's callbacks.

        Args:
            session: The dictation session whose recorder hit its PCM bound.
        """
        try:
            self.post_message(ConsoleDictationLimitSignal(session))
        except Exception:  # noqa: BLE001 - the audio path must never see this
            logger.opt(exception=True).debug(
                "Console dictation buffer limit could not be posted"
            )

    def _handle_console_dictation_buffer_limit(
        self, message: ConsoleDictationLimitSignal
    ) -> None:
        """Stop the capture whose recorder ran out of its PCM budget.

        Args:
            message: The posted buffer-limit signal.
        """
        if message.session is not self._console_dictation_session:
            # A capture the screen has already torn down; its recorder's late
            # signal must not stop whatever is recording now.
            return
        self._handle_console_dictation_limit()


    def _handle_console_dictation_limit(self) -> None:
        """Stop and transcribe when the wall-clock or memory bound is reached."""
        if self._console_dictation_state != "recording":
            return
        self.app_instance.notify(
            "Limit reached — press Mic to continue.",
            severity="warning",
        )
        if self._console_hands_free is not None:
            # A bounded ending is never a conversational turn boundary. Exit
            # the loop through its existing close-capture path so retained
            # text follows ordinary caret insertion without auto-send or the
            # loop's historical one-time reopen. A later physical Mic press
            # starts a fresh capture explicitly.
            self._console_hands_free.controller.on_exit_request()
            return
        self._request_console_dictation_stop()

    def _create_console_dictation_session(self) -> Any:
        """Build a streaming dictation session bound to this screen.

        Constructing the controller costs nothing: the optional speech stack is
        only imported when `start()` actually reaches the service factory.
        """
        return ConsoleStreamingDictationSession(
            on_event=self._emit_console_dictation_event,
            service_factory=self._dictation_service_factory,
            max_buffer_bytes=pcm_byte_limit(
                sample_rate=CONSOLE_DICTATION_SAMPLE_RATE,
                channels=CONSOLE_DICTATION_CHANNELS,
                sample_width=CONSOLE_DICTATION_SAMPLE_WIDTH,
            ),
        )


    async def _start_console_dictation(self) -> None:
        """Open the microphone for the capture the user just asked for.

        `session` is captured up front and re-checked after the await, exactly
        as `_stop_console_dictation` does: this one await covers the speech-model
        load (minutes on a fresh machine) *and* the capture opening, and two
        different things can null the screen's session inside it -- a deliberate
        cancel (`_request_console_dictation_cancel`) and a mid-capture
        `VoiceFailed` draining through `_notify_console_dictation_error`. Both
        have already told the user and cleaned up, so whichever side loses that
        race must stay silent; announcing "recording" and arming the timers
        afterwards would leave a ticking chip and a `Rec ●` button over a
        capture that is already dead, and the next press would surface an
        internal string ("Microphone dictation is not recording.").
        """
        session = (
            self._console_dictation_session or self._create_console_dictation_session()
        )
        self._console_dictation_session = session
        try:
            await asyncio.to_thread(
                session.start,
                on_buffer_limit=partial(
                    self._on_console_dictation_buffer_limit, session
                ),
            )
        except Exception as exc:
            if self._console_dictation_session is session:
                self._notify_console_dictation_error(exc)
            else:
                logger.debug(
                    "Console dictation start skipped; the attempt was cancelled"
                )
            return
        if not self.is_mounted or self._console_dictation_session is not session:
            # Cancelled, failed or unmounted while the model was loading: the
            # capture may have opened a moment ago, so release it rather than
            # leave a live microphone behind an idle button.
            await asyncio.to_thread(session.discard)
            return
        self._set_console_dictation_state("recording")
        self._console_dictation_timer = self.set_timer(
            DICTATION_MAX_SECONDS,
            self._handle_console_dictation_limit,
        )
        self._console_dictation_elapsed_timer = self.set_interval(
            1.0, self._tick_console_dictation_elapsed
        )

    @staticmethod
    def _dictation_insertion(
        draft: str,
        cursor: int,
        transcript: str,
    ) -> str:
        """Return transcript text padded only where adjacent text needs spacing.

        Trims only `" "` and `"\\t"` from the ends, not a full `.strip()`: the
        recognizer never produces a leading/trailing newline on its own, so
        an ordinary dictated transcript behaves identically either way -- but
        an inline `new-paragraph`/`new-line` command at the very start or end
        of a capture (`_join_segments`) does produce one, and a full strip
        used to discard it silently, along with everything after it looking
        like the command was never spoken.

        For the same reason, the caret-context padding below never adds a
        space next to a leading or trailing newline: the break is already
        the separator the padding exists to provide.

        Args:
            draft: The composer's current text.
            cursor: The insertion point, an offset into `draft`.
            transcript: The capture's transcript, as returned by
                `ConsoleStreamingDictationSession.stop_and_transcribe`.

        Returns:
            `transcript` trimmed of edge spaces/tabs (edge newlines kept),
            padded with a single space on whichever side abuts non-space
            text in `draft` and does not itself start or end with a newline.
        """
        cursor = max(0, min(cursor, len(draft)))
        insertion = transcript.strip(" \t")
        if not insertion.startswith("\n"):
            if cursor and not draft[cursor - 1].isspace():
                insertion = " " + insertion
        if not insertion.endswith("\n"):
            if cursor < len(draft) and not draft[cursor].isspace():
                insertion += " "
        return insertion


    def _insert_console_dictation(
        self,
        *,
        origin_session_id: str | None,
        transcript: str,
    ) -> None:
        """Insert into the originating draft without sending a message."""
        if not origin_session_id:
            return
        store = self._ensure_console_chat_store()
        composer = self._console_composer_or_none()
        if (
            composer is not None
            and store.active_session_id == origin_session_id
            and self._console_visible_draft_session_id == origin_session_id
        ):
            insertion = self._dictation_insertion(
                composer.draft_text(),
                composer.cursor_index,
                transcript,
            )
            composer.insert_text(insertion)
            store.set_session_draft(origin_session_id, composer.draft_text())
            return

        try:
            draft = store.session_draft(origin_session_id)
            insertion = self._dictation_insertion(draft, len(draft), transcript)
            store.set_session_draft(origin_session_id, draft + insertion)
            # TASK-1281 review F5: this session's composer isn't the live
            # one, so nothing records this mutation into its banked undo/
            # redo history -- correct on its own (a background session's
            # history must not be touched by a mutation its own composer
            # never saw), but it leaves that banked history stale relative
            # to the store draft it will be re-paired with on switch-in.
            # Left alone, the banked top entry predates BOTH this dictated
            # text and whatever was in the draft before it, so a single
            # Ctrl+Z after switching back in would destroy both in one
            # step (reproduced: history top = pre-"hello", store draft =
            # "hello dictated words", one undo -> ""). Dropping the banked
            # history instead makes the dictated text simply not undoable
            # via history (consistent with "history records composer
            # mutations, not store writes") rather than undoable in a way
            # that silently corrupts the draft.
            self._console_undo_histories.pop(origin_session_id, None)
        except KeyError:
            self.app_instance.notify(
                "Dictation finished, but its original Console session is gone.",
                severity="warning",
            )


    async def _stop_console_dictation(self, session: Any) -> None:
        """Finish the capture this stop was requested for, and only that one.

        The session is captured on the UI thread by
        `_request_console_dictation_stop` rather than read here, and re-checked
        after every await: a mid-capture `VoiceFailed` can drain at any point
        in between, and it tears the capture down and tells the user itself.
        Whichever side loses that race must stay silent, or one failure becomes
        two toasts -- the second one either a duplicate or an internal string
        ("Microphone dictation is not recording.") that means nothing to a user.

        Args:
            session: The dictation session that was live when the user (or the
                wall timer) asked to stop.
        """
        origin_session_id = self._console_dictation_origin_session_id
        if session is None:
            self._notify_console_dictation_error(
                RuntimeError("Microphone dictation is not recording.")
            )
            return
        if self._console_dictation_session is not session:
            logger.debug("Console dictation stop skipped; the capture was torn down")
            return
        try:
            transcript = await asyncio.to_thread(session.stop_and_transcribe)
        except Exception as exc:
            if not session.retry_available:
                await asyncio.to_thread(session.discard)
                if self._console_dictation_session is session:
                    self._notify_console_dictation_error(exc)
                return
            try:
                confirmed = await self.run_worker(
                    self.app_instance.push_screen_wait(
                        ConfirmationDialog(
                            title="Parakeet transcription failed",
                            message=(
                                "Parakeet failed. Retry this audio with "
                                "faster-whisper?"
                            ),
                            confirm_label="Retry",
                            cancel_label="Keep draft",
                        )
                    ),
                    exclusive=False,
                    exit_on_error=False,
                ).wait()
            except asyncio.CancelledError:
                self._finish_failed_console_dictation(session)
                raise
            except Exception:  # noqa: BLE001 - modal teardown is best effort
                logger.opt(exception=True).debug(
                    "Console dictation retry prompt did not complete"
                )
                self._finish_failed_console_dictation(session)
                return
            if not confirmed:
                self._finish_failed_console_dictation(session)
                return
            try:
                transcript = await asyncio.to_thread(
                    session.retry_with_faster_whisper
                )
            except Exception as retry_exc:
                owns_session = self._console_dictation_session is session
                self._finish_failed_console_dictation(session)
                if owns_session and self.is_mounted:
                    self._notify_console_dictation_error(retry_exc)
                return
        await self._finish_successful_console_dictation(
            session,
            origin_session_id=origin_session_id,
            transcript=transcript,
        )

    def _finish_failed_console_dictation(self, session: Any) -> None:
        """Clear one failed/rejected retry without mutating the draft."""

        try:
            session.clear_retry()
        except Exception:  # noqa: BLE001 - cleanup must always reach idle
            logger.opt(exception=True).debug(
                "Console dictation retry state could not be cleared"
            )
        if self._console_dictation_session is not session:
            return
        self._cancel_console_dictation_timer()
        self._cancel_console_dictation_elapsed_timer()
        self._console_dictation_session = None
        self._console_dictation_origin_session_id = None
        self._console_dictation_partial = ""
        self._console_pending_voice_action = None
        self._console_dictation_late_discard_ack = False
        self._set_console_dictation_state("idle")

    async def _finish_successful_console_dictation(
        self,
        session: Any,
        *,
        origin_session_id: str | None,
        transcript: str,
    ) -> None:
        """Apply the single insertion/idle/action tail for success or retry."""

        session.clear_retry()
        if not self.is_mounted:
            return
        if self._console_dictation_session is not session:
            return
        # A command-only capture (Task 2's `commands_consumed` early-return
        # in `stop_and_transcribe`) returns "" here rather than raising --
        # that is not a silent-microphone failure, but it is also nothing to
        # insert. `_dictation_insertion` would otherwise pad it to a stray
        # space at the caret and persist that to the draft.
        # V4 task 5, rule 10: a realtime loop entered while this capture was
        # open ADOPTS its transcript as the loop's first spoken turn. It is
        # consumed there instead of being inserted here -- the words were
        # spoken as a turn, not typed as a draft, and a leftover copy in the
        # composer would be sent a second time by the next Enter.
        if transcript and not self._console_realtime_adopt_transcript(transcript):
            self._insert_console_dictation(
                origin_session_id=origin_session_id,
                transcript=transcript,
            )
        self._console_dictation_origin_session_id = None
        self._console_dictation_partial = ""
        self._set_console_dictation_state("idle")
        late_discard, self._console_dictation_late_discard_ack = (
            self._console_dictation_late_discard_ack,
            False,
        )
        if not transcript:
            # A command-only capture ("Console, new paragraph." alone, or
            # "Console, stop." with nothing dictated first). Not an error --
            # `stop_and_transcribe` returns "" for it deliberately -- but the
            # user's break IS silently dropped here, and saying nothing at all
            # is indistinguishable from a capture that did land.
            self.app_instance.notify(
                _VOICE_ACK_NOTHING_TO_INSERT, severity="information"
            )
        if late_discard:
            # A spoken "discard" that arrived too late to abort anything. It
            # explains the outcome better than "Capture ended." does, so it
            # takes that slot rather than adding a second spoken ack.
            self._speak_status(_VOICE_ACK_TOO_LATE_TO_DISCARD)
        elif self._console_pending_voice_action is None:
            # A plain stop -- no capture-ending command queued anything
            # further. The command acks below speak in its place, so this
            # and they never double up on the same capture.
            self._speak_status(
                "Capture ended." if transcript else _VOICE_ACK_NOTHING_TO_INSERT
            )
        await self._run_pending_console_voice_action(origin_session_id)

    def _request_console_dictation_stop(self) -> None:
        if self._console_dictation_state != "recording":
            return
        # Read on the UI thread, atomically with the state change, so the
        # worker finishes the capture the user stopped rather than whatever is
        # in the field by the time it first ticks.
        session = self._console_dictation_session
        self._cancel_console_dictation_timer()
        self._cancel_console_dictation_elapsed_timer()
        self._set_console_dictation_state("transcribing")
        self.run_worker(
            self._stop_console_dictation(session),
            exclusive=True,
            group="console-dictation-stop",
            exit_on_error=False,
        )


    async def _discard_console_dictation_session(self, session: Any) -> None:
        """Release a cancelled capture's microphone off the UI thread.

        `discard()` is documented as non-blocking (`abandon()` never joins), but
        it still reaches the audio backend to close the stream: measured at
        1.51 s of frozen UI when called inline. Every other call site already
        goes through `asyncio.to_thread`; this is the one that did not.

        Args:
            session: The session to abandon. Failures are logged, never raised
                -- cancelling must not produce an error the user did not cause.
        """
        try:
            await asyncio.to_thread(session.discard)
        except Exception:  # noqa: BLE001 - cancelling must never raise
            logger.opt(exception=True).debug(
                "Console dictation could not be cancelled cleanly"
            )
        # Spoken only now, not by the caller: the release above holds the
        # microphone for up to ~1.5 s after the UI is already back at `idle`,
        # and `_speak_status`'s state check cannot see that -- so acking from
        # the caller talks straight into a still-open mic. Its check is still
        # what refuses if the user has opened a NEW capture in the meantime.
        self._speak_status("Discarded.")


    def _request_console_dictation_cancel(self) -> None:
        """Abandon a capture that is `starting` or `recording`, without waiting.

        The `starting` phase now covers a speech-model load, which is a
        multi-gigabyte download on a fresh machine. `abandon()` returns
        immediately (it never joins), and the load itself is on a daemon
        thread, so this returns the UI to idle at once and lets the process
        exit even if the download is still running.

        Dropping the session first is what makes `_start_console_dictation`'s
        re-checks fire, so the cancelled attempt cannot also raise a failure
        toast on its way out. The release itself is handed to a worker: the UI
        is already back at idle by then, so nothing the user can see is waiting
        on the audio backend letting go of the device.

        Task 3: also the target for a spoken "Console, discard." mid-capture --
        the body below has never depended on which of the two states it is
        torn down from (both stop the same two timers and hand the same
        session to the same worker), so `recording` needed no separate path,
        only this guard admitting it. The manual mic button still never
        reaches this method for `recording` (it routes there to
        `_request_console_dictation_stop`, which inserts); only the spoken
        command does.

        Review fix round 1 (Finding 1): a spoken "discard" can also arrive
        while `transcribing` -- the `VoiceCommand` branch admits that state
        so a genuinely-current trailing command is not mistaken for a stale
        one. This method cannot itself abort an in-flight stop-and-transcribe
        (nothing here is cancelable at that point, so the guard below still
        refuses), but the clear immediately below still applies regardless:
        "Console, send." then "Console, discard." inside that same window
        must drop the queued `send` even though the capture finishes normally.
        """
        # Unconditional, ahead of the guard below, for exactly that reason.
        self._console_pending_voice_action = None
        if self._console_dictation_state not in ("starting", "recording"):
            if self._console_dictation_state == "transcribing":
                # A spoken "discard" the guard above cannot honor: the capture
                # is already being transcribed and will insert. Silently
                # doing nothing here reads as the command having been missed,
                # when in fact it was heard and refused -- and the user is
                # about to watch text appear that they just asked to throw
                # away. The spoken half waits for `idle` (see the field's
                # docstring); the toast does not.
                self._console_dictation_late_discard_ack = True
                self.app_instance.notify(
                    _VOICE_ACK_TOO_LATE_TO_DISCARD, severity="warning"
                )
            return
        session = self._console_dictation_session
        self._cancel_console_dictation_timer()
        self._cancel_console_dictation_elapsed_timer()
        self._console_dictation_session = None
        self._console_dictation_origin_session_id = None
        self._console_dictation_partial = ""
        self._set_console_dictation_state("idle")
        if session is not None:
            # The worker speaks "Discarded." once the microphone is actually
            # released; see `_discard_console_dictation_session`.
            self.run_worker(
                self._discard_console_dictation_session(session),
                group="console-dictation-cancel",
                exit_on_error=False,
            )
        self.app_instance.notify("Dictation cancelled.", severity="information")
        if session is None:
            # Nothing to release, so nothing to wait for.
            self._speak_status("Discarded.")


    def _request_console_dictation_start(self) -> None:
        if self._console_dictation_state != "idle":
            return
        # Re-probe on every activation attempt (TASK-15): refreshes the mic
        # tooltip so an extra installed or a microphone plugged in mid-run is
        # reflected without a remount. Cosmetic only -- see
        # `_sync_console_dictation_availability`'s docstring for why this
        # never blocks the attempt below.
        self._sync_console_dictation_availability()
        store = self._ensure_console_chat_store()
        self._console_dictation_origin_session_id = store.active_session_id
        self._console_dictation_partial = ""
        # Defensive (review fix round 1, Finding 1): the `VoiceCommand`
        # branch's own state guard is what actually stops a stale command
        # from queuing here, but a fresh capture starting with nothing
        # pending is the correct state regardless of how one might have
        # leaked in, so it is reset again on this end too.
        self._console_pending_voice_action = None
        self._console_dictation_late_discard_ack = False
        self._set_console_dictation_state("starting")
        # Unconditional -- not gated on the spoken-feedback toggle: a status
        # ack or a "read that back" reply can be playing even with feedback
        # off, and the single-slot TTS player only stops the PREVIOUS clip
        # when a NEW one starts -- opening a microphone plays nothing, so it
        # would never stop this on its own. Posted here, before the worker
        # below ever reaches the recorder, so an in-flight clip cannot bleed
        # into the recognizer and get transcribed into this capture's draft.
        from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
            TTSPlaybackEvent,
        )

        self.app_instance.post_message(TTSPlaybackEvent(action="stop"))
        self.run_worker(
            self._start_console_dictation(),
            exclusive=True,
            group="console-dictation-start",
            exit_on_error=False,
        )

    def _handle_console_dictation_button(self) -> None:
        """Route a press of the composer's mic button (`#console-dictation`).

        Moved verbatim out of `ChatScreen.on_button_pressed`'s
        `console-dictation` branch (wave-4 console decomposition, task 2),
        which was the fifth-largest of that method's 19 branches. Every
        line below is the pre-move body unchanged -- including the two
        loop-supersede guards, whose ordering (pipeline hands-free first,
        realtime second) is behaviour, not style: a session can only be
        installed in one of the two at a time, but the guards are the only
        thing standing between a mic press and a second recorder opening
        on top of a live one.

        The screen's branch is now `event.stop()` plus a call to this.
        `event` itself never crosses the boundary: the mic button carries
        no per-press payload, so there is nothing for a controller to read
        off it, and keeping Textual's event object on the screen is what
        lets this method be reached directly by a test or a key binding.
        """
        if self._console_hands_free is not None:
            # Task 5: mic press exits the hands-free loop from any
            # state, exactly like Esc/spoken "stop" -- superseding the
            # ordinary one-shot toggle below for as long as the loop is
            # running.
            self._console_hands_free.controller.on_exit_request()
            return
        if self._console_realtime is not None:
            # V4 task 5 (final review C1): the SAME rule for the
            # realtime engine, and it matters more here. Falling
            # through to the dictation toggle below would open a
            # second `AudioRecordingService` (at 16 kHz, alongside the
            # tap's 24 kHz stream), load the entire STT stack the
            # realtime engine exists to avoid, and arm the V2 spoken-
            # command classifier mid-session -- all while the realtime
            # session kept running and billing. The docs promise this
            # button exits the loop; it exits the loop.
            self._console_realtime.controller.on_exit_request()
            return
        if self._console_dictation_state == "idle":
            self._request_console_dictation_start()
        elif self._console_dictation_state == "starting":
            # A first-run model load runs for minutes; this is the only
            # in-app way out of it. "transcribing" has no cancel -- the
            # capture is already recorded and worth finishing.
            self._request_console_dictation_cancel()
        elif self._console_dictation_state == "recording":
            self._request_console_dictation_stop()
        return
