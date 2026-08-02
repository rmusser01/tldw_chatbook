"""Headless finite-state machine for the Console hands-free conversation
loop: speak -> it sends -> the reply is spoken -> speak again.

Deliberately free of Textual, wall-clock, and direct audio/TTS imports:
every input is a plain method call (`enter`, `on_voice_final`,
`on_composer_key`, ...), every output is a frozen-dataclass intent handed to
an injected `emit` callable, and the only notion of time is `tick(now)`,
driven by a caller-supplied float. That split is what makes the countdown
and the resume-vs-expiry race below deterministically unit-testable without
a running app, a real clock, or a real dictation service (see the
hands-free-loop design doc,
`Docs/superpowers/specs/2026-08-02-hands-free-loop-design.md`).

## States

`idle -> listening -> countdown -> awaiting_reply -> speaking -> listening
-> ...`; `exit` (an `ExitLoop` intent, landing back in `idle`) is reachable
from every state via `on_exit_request()` or `on_voice_command("stop")`.

## Countdown timing (`tick`)

`on_voice_final()` arms the countdown but cannot itself record an anchor
time -- it takes no `now` argument. Instead it leaves `_armed_at` as
`None` ("armed, pending an anchor"), and the *first* subsequent `tick(now)`
call adopts that `now` as the anchor. Every following `tick(now)` compares
`now - _armed_at` against `send_delay_seconds`: while positive remaining
time is left, it emits `CountdownTick(remaining)` (monotonically
decreasing as `now` advances); once remaining time reaches zero or below,
the countdown expires into the existing V2 stop-and-send flow via
`RequestStopAndSend` -- there is no second send path.

## Capture open/close bookkeeping

`_capture_open` tracks whether the mic is presently live. `enter()` sets it
from `capture_live` (opening the mic itself, via `OpenCapture`, only if the
loop is entered from idle). The mic stays open through `listening` and
`countdown` (a countdown must remain cancellable by resumed speech). It
closes on a successful send (`CloseCapture`) *unless* `acoustic_barge_in`
is enabled, in which case it deliberately stays open through
`awaiting_reply`/`speaking` so an acoustic interruption needs no reopen.
Every path back to `listening` reopens the mic (`OpenCapture`) only if it
is not already open, so acoustic-mode barge-in never double-opens a mic
that was never closed.

## Reopen-once-then-exit (service-side capture limits)

`on_capture_ended(had_segments, limit_hit)` covers the recorder's own
60s wall-clock cutoff / buffer cap ending a capture out from under the
loop. With finalized segments pending, this is treated exactly like
countdown expiry (`RequestStopAndSend`). With nothing captured, the loop
reopens for one more turn (`OpenCapture`) rather than exiting outright --
but only once in a row: a *second* consecutive empty-limit ending exits
the loop (`ExitLoop`) rather than looping forever in a silent room. A
successful send in between resets the reopen-once flag, so it is
consecutive empty endings specifically that exit, not a lifetime cap.

## Degraded (no-webrtcvad) mode

Without `webrtcvad` available, the dictation service never emits
`VoiceSpeechResumed` at all (see `Audio/dictation_service_lazy.py`'s
silence-gate detection, which the resume signal depends on) -- so
countdown-cancel-by-voice (`on_speech_resumed()` while `countdown`) and
acoustic barge-in (`on_speech_resumed()` while `speaking` with
`acoustic_barge_in=True`) are both silently inert in that mode: the ticks
simply keep counting down / the reply simply keeps speaking, because the
event that would cancel them never arrives. This controller needs no
special case for that -- keypress cancel (`on_composer_key()`) and spoken
"stop" (`on_voice_command("stop")`) work identically either way, since
they do not depend on VAD. It is purely a documentation obligation: this
docstring states the caveat, and `ModeChanged` (see below) is worded so it
never promises voice-cancel as a universal guarantee that degraded mode
would falsify.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union

HandsFreeState = Literal["idle", "listening", "countdown", "awaiting_reply", "speaking"]

_VALID_STATES: tuple[HandsFreeState, ...] = (
    "idle", "listening", "countdown", "awaiting_reply", "speaking",
)


# ---------------------------------------------------------------------------
# Intents (all frozen, emitted via the injected `emit` callable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestStopAndSend:
    """Countdown expired (or a service-side capture limit ended the
    capture with segments pending): drive V2's existing stop-and-send flow
    (`_console_pending_voice_action = "send"`). There is no second send
    path -- this is the only intent that triggers a send."""


@dataclass(frozen=True)
class SilenceSpeech:
    """Stop in-flight reply audio (a barge-in). Reply GENERATION is never
    cancelled by this -- only the audio output. The screen's job is to
    call the sequencer's `flush()` and stop the active sink/player."""


@dataclass(frozen=True)
class SuppressReplySpeech:
    """Suppress this reply's speech entirely before it starts (a keypress
    arrived while still `awaiting_reply`) or recover from a failed
    generation -- either way the sequencer must never start speaking this
    reply. Generation itself continues silently into the transcript."""


@dataclass(frozen=True)
class OpenCapture:
    """Open (or reopen) the mic for a fresh turn."""


@dataclass(frozen=True)
class CloseCapture:
    """Close the mic (mic/speaker exclusion while a reply is generated or
    spoken), emitted only when the mic is not already closed."""


@dataclass(frozen=True)
class CountdownTick:
    """A countdown heartbeat while `countdown`; `remaining` (seconds) is
    monotonically decreasing across consecutive `tick()` calls until either
    cancellation or expiry."""

    remaining: float


@dataclass(frozen=True)
class ModeChanged:
    """Announces the controller's current state (the `_transition()`
    chokepoint's only side effect besides updating `state` itself). Purely
    descriptive of the state label -- it makes no claim about which inputs
    can move the loop out of that state, so it stays accurate regardless of
    which cancellation paths happen to be live or inert in a given capture
    mode (see the module/class docstrings)."""

    state: HandsFreeState


@dataclass(frozen=True)
class ExitLoop:
    """Tear the hands-free loop down to today's idle Console behavior.
    Reachable from every state."""


HandsFreeIntent = Union[
    RequestStopAndSend,
    SilenceSpeech,
    SuppressReplySpeech,
    OpenCapture,
    CloseCapture,
    CountdownTick,
    ModeChanged,
    ExitLoop,
]


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class HandsFreeController:
    """Pure headless FSM driving the Console hands-free conversation loop.
    See the module docstring for the full state/timing/capture-bookkeeping
    design, including the degraded (no-webrtcvad) mode caveat: without
    webrtcvad, VoiceSpeechResumed never arrives, so countdown-cancel-by-
    voice and acoustic barge-in are silently inert there -- keypress cancel
    still works regardless.

    Args:
        emit: Called with exactly one `HandsFreeIntent` at a time as the
            controller reacts to inputs.
        send_delay_seconds: Countdown duration (seconds) from a finalized
            segment to an automatic send.
        acoustic_barge_in: When True, `on_speech_resumed()` also barges in
            during `speaking` (not just a composer keypress), and the mic
            is kept open through `awaiting_reply`/`speaking` instead of
            being closed on send.
    """

    def __init__(
        self,
        emit: Callable[[HandsFreeIntent], None],
        send_delay_seconds: float = 1.5,
        acoustic_barge_in: bool = False,
    ) -> None:
        self._emit = emit
        self._send_delay_seconds = send_delay_seconds
        self._acoustic_barge_in = acoustic_barge_in

        self._state: HandsFreeState = "idle"
        self._armed_at: Optional[float] = None
        self._resume_latched: bool = False
        self._capture_open: bool = False
        self._capture_limit_reopened: bool = False
        self._reply_finished: bool = False
        self._sequencer_drained: bool = False

    # -- public state ---------------------------------------------------

    @property
    def state(self) -> HandsFreeState:
        return self._state

    # -- transition chokepoint -------------------------------------------

    def _transition(self, new_state: HandsFreeState) -> None:
        """The SOLE place `self._state` is assigned outside `__init__`.
        Always emits `ModeChanged`, even for a nominal self-transition --
        callers that do not conceptually "enter" a new state (e.g. the
        resume-latch consuming an already-cancelled `on_voice_final` while
        staying `listening`) simply do not call this."""
        self._state = new_state
        self._emit(ModeChanged(new_state))

    def _exit(self) -> None:
        self._emit(ExitLoop())
        self._transition("idle")

    def _reopen_capture_if_closed(self) -> None:
        if not self._capture_open:
            self._emit(OpenCapture())
            self._capture_open = True

    def _begin_awaiting_reply(self) -> None:
        """Shared by countdown expiry and `on_capture_ended`'s
        with-segments path -- both are "a send is happening now"."""
        self._emit(RequestStopAndSend())
        if self._capture_open and not self._acoustic_barge_in:
            self._emit(CloseCapture())
            self._capture_open = False
        # A successful send resets the reopen-once flag: only *consecutive*
        # empty-limit endings exit the loop, not a lifetime cap.
        self._capture_limit_reopened = False
        self._reply_finished = False
        self._sequencer_drained = False
        self._armed_at = None
        self._transition("awaiting_reply")

    def _maybe_complete_reply(self) -> None:
        """Gate on BOTH `on_reply_finished()` and `on_sequencer_drained()`
        having fired (arrival order does not matter -- whichever lands
        second triggers this). Gating on `state in (awaiting_reply,
        speaking)` is also what makes carried finding #2 a one-line no-op:
        a late `on_sequencer_drained()` for an already-barged-in reply
        arrives while `state == "listening"`, fails this check, and does
        nothing -- in particular it never emits a second `OpenCapture`."""
        if self._state not in ("awaiting_reply", "speaking"):
            return
        if not (self._reply_finished and self._sequencer_drained):
            return
        self._reply_finished = False
        self._sequencer_drained = False
        self._reopen_capture_if_closed()
        self._transition("listening")

    # -- public inputs ----------------------------------------------------

    def enter(self, capture_live: bool) -> None:
        """Enter the hands-free loop. `capture_live=True` keeps an
        already-open capture as the first turn; `capture_live=False` opens
        one (emits `OpenCapture`)."""
        self._armed_at = None
        self._resume_latched = False
        self._capture_limit_reopened = False
        self._reply_finished = False
        self._sequencer_drained = False
        self._capture_open = capture_live
        if not capture_live:
            self._emit(OpenCapture())
            self._capture_open = True
        self._transition("listening")

    def tick(self, now: float) -> None:
        """Injected-clock heartbeat. A no-op outside `countdown`. The
        first call after arming adopts `now` as the anchor (`on_voice_final`
        cannot record one itself -- it has no `now` parameter)."""
        if self._state != "countdown":
            return
        if self._armed_at is None:
            self._armed_at = now
        remaining = self._send_delay_seconds - (now - self._armed_at)
        if remaining <= 0:
            self._begin_awaiting_reply()
            return
        self._emit(CountdownTick(remaining))

    def on_voice_final(self) -> None:
        """A segment was finalized. Only meaningful while `listening`."""
        if self._state != "listening":
            return
        if self._resume_latched:
            # Carried finding #1: the silence gate zeroes its timestamp
            # BEFORE the seconds-long transcription runs, so a resume for
            # the NEXT utterance can arrive before this (delayed) final for
            # the previous one. Treat as already-cancelled: stay listening,
            # do not arm a countdown, consume the latch (one-shot).
            self._resume_latched = False
            return
        self._armed_at = None
        self._transition("countdown")

    def on_speech_resumed(self) -> None:
        """Silence-to-speech transition (`VoiceSpeechResumed`). Effect
        depends on state; a no-op in `idle`/`awaiting_reply`, and in
        `speaking` unless `acoustic_barge_in` is enabled. See the class
        docstring for the degraded (no-webrtcvad) mode caveat: this input
        simply never arrives there, which is what makes voice-cancel inert
        in that mode without any special-casing here."""
        if self._state == "listening":
            # Latch for a following on_voice_final -- see carried finding
            # #1 above.
            self._resume_latched = True
            return
        if self._state == "countdown":
            self._armed_at = None
            self._transition("listening")
            return
        if self._state == "speaking" and self._acoustic_barge_in:
            self._emit(SilenceSpeech())
            self._reopen_capture_if_closed()
            self._transition("listening")
            return
        # awaiting_reply, idle, or speaking-without-opt-in: ignored.

    def on_voice_command(self, name: str) -> None:
        """V2 spoken commands keep working mid-loop; only "stop" is this
        controller's business (exit from any state). Every other command
        name is out of scope here -- the screen dispatches it normally."""
        if name == "stop":
            self._exit()

    def on_capture_ended(self, had_segments: bool, limit_hit: bool) -> None:
        """The recorder's own service-side limit (60s wall-clock cutoff or
        buffer cap) ended the capture out from under the loop. Meaningful
        only when `limit_hit` is True and the loop is `listening` or
        `countdown` -- a non-limit ending is out of this method's contract
        (normal endings arrive via `on_voice_final`/countdown expiry
        instead) and is silently ignored."""
        if not limit_hit:
            return
        if self._state not in ("listening", "countdown"):
            return
        self._capture_open = False  # the capture already ended
        if had_segments:
            self._begin_awaiting_reply()
            return
        if not self._capture_limit_reopened:
            # Reopen once for a fresh turn rather than exiting outright.
            self._capture_limit_reopened = True
            self._reopen_capture_if_closed()
            self._transition("listening")
            return
        # Second CONSECUTIVE empty-limit ending: no infinite reopen churn.
        self._exit()

    def on_composer_key(self) -> None:
        """A composer keypress. Cancels a countdown, suppresses an
        as-yet-unspoken reply, or barges in on a speaking one; a no-op
        while `listening`/`idle` (ordinary typing -- the controller never
        swallows keys outside the states that need them)."""
        if self._state == "countdown":
            self._armed_at = None
            self._transition("listening")
            return
        if self._state == "awaiting_reply":
            self._emit(SuppressReplySpeech())
            self._reopen_capture_if_closed()
            self._transition("listening")
            return
        if self._state == "speaking":
            self._emit(SilenceSpeech())
            self._reopen_capture_if_closed()
            self._transition("listening")
            return

    def on_exit_request(self) -> None:
        """Esc / mic press: exit from any state."""
        self._exit()

    def on_reply_started(self) -> None:
        """Reply generation has begun streaming. Currently a bookkeeping
        hook with no FSM effect of its own -- the state-changing signal is
        `on_first_utterance()`, once the sequencer actually queues
        speakable text. Kept as a distinct input for reply-lifecycle
        symmetry (`started` / `first_utterance` / `finished` / `failed`)
        and potential future instrumentation."""
        return

    def on_first_utterance(self) -> None:
        """The sequencer queued its first speakable sentence."""
        if self._state != "awaiting_reply":
            return
        self._transition("speaking")

    def on_reply_finished(self) -> None:
        """Reply text generation is complete. Combined with
        `on_sequencer_drained()` (either order) this returns to
        `listening`; a reply with zero speakable sentences reaches
        `listening` this way without ever visiting `speaking`."""
        self._reply_finished = True
        self._maybe_complete_reply()

    def on_sequencer_drained(self) -> None:
        """The sentence sequencer's queue is empty and nothing is in
        flight. See `_maybe_complete_reply` for why a late arrival of this
        while already `listening` (carried finding #2 -- a barged-in
        reply's own `reply_completed()` still drains) is a safe no-op."""
        self._sequencer_drained = True
        self._maybe_complete_reply()

    def on_reply_failed(self) -> None:
        """Reply generation failed. The loop never traps on this -- the
        existing error toast is the screen's business, not this
        controller's; here we only suppress this reply's speech and
        recover to `listening`."""
        if self._state not in ("awaiting_reply", "speaking"):
            return
        self._emit(SuppressReplySpeech())
        self._reply_finished = False
        self._sequencer_drained = False
        self._reopen_capture_if_closed()
        self._transition("listening")
