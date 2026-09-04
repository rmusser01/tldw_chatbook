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
non-increasing as `now` advances, and always clamped to
`[0, send_delay_seconds]`); once remaining time reaches zero or below, the
countdown expires into the existing V2 stop-and-send flow via
`RequestStopAndSend` -- there is no second send path. **The injected clock
is expected to be monotonic**; a backwards step is defensively handled by
re-anchoring to the earliest `now` seen so far (so elapsed time can never
go negative) and by clamping every emitted value to the last one emitted,
but this is a safety net, not a feature to rely on -- see task-3-review.md
F6.

## `awaiting_reply` watchdog (`tick`, again)

`awaiting_reply` is also `tick()`-driven: `_awaiting_armed_at` anchors the
same way `_armed_at` does (first `tick()` call after
`_begin_awaiting_reply()` adopts `now`). **The watchdog guards only the
send -> `on_reply_started()` gap**, not generation in progress
(task-3-review.md N1): if `AWAITING_REPLY_DEADLINE_SECONDS` elapses with
`on_reply_started()` never having arrived at all, the send is presumed to
have silently refused (V2's send path has refusal branches -- an
unmounted send button, a session that changed mid-flight -- that emit no
reply signal whatsoever back to this controller). `on_reply_started()`
itself DISARMS the watchdog outright the moment it arrives (it does not
merely re-anchor it) -- it is positive proof the send did not silently
refuse, and more than `AWAITING_REPLY_DEADLINE_SECONDS` from there to the
first *speakable* sentence is routine in this app (a cold local-model
load, a reasoning model's long non-speakable thinking block, or a reply
that opens with a fenced code block, which the sentence sequencer skips
entirely by design and so legitimately yields no utterance for its whole
duration). Once disarmed, only `on_reply_failed()` or the sequencer's own
signals are the failure detector for that reply -- never this wall clock.

On genuine expiry (never disarmed), the honest recovery is to stop
waiting for a reply that may never come and let the user speak again:
`SuppressReplySpeech` (this reply must never be allowed to speak, exactly
like `on_reply_failed()`), `OpenCapture` (if needed), and
`_transition("listening")`. Because the abandoned reply may still show up
late, a subsequent `on_reply_started()` for it re-emits
`SuppressReplySpeech` again (idempotently) rather than being a silent
no-op -- in the real wiring, `on_reply_started` is exactly the point that
would otherwise call the sentence sequencer's own `begin_reply()`, which
RESETS the sequencer's suppression latch, so without this the late reply
could still speak into the reopened mic despite the earlier suppression
(task-3-review.md N2). This is a last-resort backstop for the narrow
send-to-started window, not the primary error path -- `on_reply_failed()`,
driven by the wiring layer's own error handling, is expected to fire long
before this in the vast majority of failures.

## Capture open/close bookkeeping

`_capture_open` tracks whether the mic is presently live (exposed
read-only as `capture_open` for callers/tests). `enter()` sets it from
`capture_live` when entering fresh from `idle` (opening the mic itself,
via `OpenCapture`, only if not already live); re-entering while already
running trusts this controller's own bookkeeping over a possibly-stale
`capture_live` argument instead (see "Re-entry" below). The mic stays open
through `listening` and `countdown` (a countdown must remain cancellable
by resumed speech).

**The send always stops the capture, in BOTH capture modes** -- V2's
stop-and-send flow only ever runs from the recorder's own stop-success
tail, so there is no way for a capture to survive a send regardless of
`acoustic_barge_in`. `_begin_awaiting_reply()` therefore closes the mic
unconditionally. Acoustic mode's actual difference is *when* the mic
reopens: `on_reply_started()` reopens it immediately (mid-reply is
precisely the acoustic mode's point -- the user may interrupt by
speaking), whereas default mode leaves it closed until the reply fully
drains (`_maybe_complete_reply`, the V2 mic/speaker exclusion rule). Every
reopen goes through `_reopen_capture_if_closed()`, which only opens if not
already open, so acoustic-mode's early reopen never double-opens when
drained-completion runs its own reopen check afterward.

## Reopen-once-then-exit (service-side capture limits)

`on_capture_ended(had_segments, limit_hit)` covers the recorder's own
60s wall-clock cutoff / buffer cap ending a capture out from under the
loop. It always corrects `_capture_open = False` first (the capture
genuinely ended, regardless of FSM state) -- see task-3-review.md F2. In
`listening`/`countdown`: with finalized segments pending, this is treated
exactly like countdown expiry (`RequestStopAndSend`); with nothing
captured, the loop reopens for one more turn (`OpenCapture`) rather than
exiting outright, but only once in a row -- a *second* consecutive
empty-limit ending exits the loop (`ExitLoop`) rather than looping forever
in a silent room. A successful send in between resets the reopen-once
flag, so it is consecutive empty endings specifically that exit, not a
lifetime cap. In `awaiting_reply`/`speaking` (a reply is outstanding, most
often the acoustic-mode mic hitting its own limit mid-reply): no send is
issued regardless of `had_segments` -- a send mid-reply would interleave
two turns, which this FSM's single-outstanding-reply model cannot
represent -- but in acoustic mode the mic is still reopened (per the same
`on_reply_started` rule above) so the user is not left deaf mid-turn; in
default mode there is nothing further to do (the mic was already meant to
be closed there). This mid-reply reopen is routed through the SAME
consecutive-empty-limit ceiling as the `listening`/`countdown` branch
(task-3-review.md N3) -- `speaking` has no watchdog of its own (by
design; keypress barge-in stays available there), so this ceiling is the
only bound on it. A limit-hit ending WITH segments mid-reply resets the
ceiling (it is not a "silent room" ending, even though no send is issued
for it); a SECOND consecutive empty-limit ending exits the loop, exactly
as it would in `listening`/`countdown`.

## Re-entry (`enter()` called while not `idle`)

Calling `enter()` again while the loop is already running (state !=
`idle`) re-confirms `listening`: it resets the same per-loop bookkeeping
as a fresh entry, but does NOT trust the `capture_live` argument for the
open/close decision (a stale caller-supplied snapshot could otherwise
double-open an already-open mic -- task-3-review.md F7/P4b) -- it defers
to `_capture_open` via `_reopen_capture_if_closed()` instead. If reply
audio was in flight (`state == "speaking"`), it is silenced first
(`SilenceSpeech`) -- there is no AEC, so leaving it audible while
`listening` believes the mic is live would have the recognizer transcribe
the assistant's own voice.

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
from typing import TYPE_CHECKING, Callable, Literal, Optional, Union

if TYPE_CHECKING:
    # Only for the `ModeChanged.state` annotation below -- `from __future__
    # import annotations` (above) means this is never evaluated at import
    # time, so it carries no runtime dependency on the V4 module and no
    # circular-import risk even though `console_realtime_loop.py` imports
    # `ModeChanged`/`ExitLoop` from *this* module.
    from tldw_chatbook.Chat.console_realtime_loop import RealtimeLoopState

HandsFreeState = Literal["idle", "listening", "countdown", "awaiting_reply", "speaking"]

_VALID_STATES: tuple[HandsFreeState, ...] = (
    "idle",
    "listening",
    "countdown",
    "awaiting_reply",
    "speaking",
)

#: How long `awaiting_reply` may wait, via `tick()`, before presuming the
#: send silently refused and recovering to `listening` on its own (see the
#: module docstring's "`awaiting_reply` watchdog" section). A last-resort
#: backstop, not the primary error path -- `on_reply_failed()` is expected
#: to fire long before this in the vast majority of failures.
AWAITING_REPLY_DEADLINE_SECONDS: float = 30.0


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
    always clamped to `[0, send_delay_seconds]` and monotonically
    non-increasing across consecutive `tick()` calls until either
    cancellation or expiry -- strictly decreasing under a monotonic clock,
    never larger than the previous value even under a backwards clock
    step (see `HandsFreeController`'s "Countdown timing" docstring
    section)."""

    remaining: float


@dataclass(frozen=True)
class ModeChanged:
    """Announces the controller's current state (the `_transition()`
    chokepoint's only side effect besides updating `state` itself). Purely
    descriptive of the state label -- it makes no claim about which inputs
    can move the loop out of that state, so it stays accurate regardless of
    which cancellation paths happen to be live or inert in a given capture
    mode (see the module/class docstrings).

    `state` also accepts a V4 `RealtimeLoopState` label: this intent type is
    shared verbatim between `HandsFreeController` (V3, this module) and
    `RealtimeLoopController` (V4, `console_realtime_loop.py`) -- same
    vocabulary, different internal machine. `reason` is a V4-only,
    additive field (default `None`); `HandsFreeController` never sets it."""

    state: "HandsFreeState | RealtimeLoopState"
    reason: Optional[str] = None


@dataclass(frozen=True)
class ExitLoop:
    """Tear the hands-free loop down to today's idle Console behavior.
    Reachable from every state.

    `reason` is a V4-only, additive field (default `None`) explaining why
    `RealtimeLoopController` (`console_realtime_loop.py`) exited (e.g.
    `"idle-timeout"`, `"connect-failed"`, `"connection-lost"`);
    `HandsFreeController` (V3, this module) never sets it."""

    reason: Optional[str] = None


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
            is reopened as soon as reply generation starts
            (`on_reply_started()`) rather than only once the reply fully
            drains -- the send still closes the mic in both modes.
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
        self._last_countdown_remaining: Optional[float] = None
        self._awaiting_armed_at: Optional[float] = None
        self._awaiting_watchdog_disarmed: bool = False
        self._reply_abandoned_by_watchdog: bool = False
        self._resume_latched: bool = False
        self._capture_open: bool = False
        self._capture_limit_reopened: bool = False
        self._reply_finished: bool = False
        self._sequencer_drained: bool = False

    # -- public state ---------------------------------------------------

    @property
    def state(self) -> HandsFreeState:
        return self._state

    @property
    def capture_open(self) -> bool:
        """This controller's own belief about whether the mic is
        presently live. Read-only; useful for tests and diagnostics --
        the emitted `OpenCapture`/`CloseCapture` intents are what the
        wiring layer should actually act on."""
        return self._capture_open

    # -- transition chokepoint -------------------------------------------

    def _transition(self, new_state: HandsFreeState) -> None:
        """The SOLE place `self._state` is assigned outside `__init__`.
        Always emits `ModeChanged`, even for a nominal self-transition --
        callers that do not conceptually "enter" a new state (e.g. the
        resume-latch consuming an already-cancelled `on_voice_final` while
        staying `listening`) simply do not call this. Rejects any target
        outside `_VALID_STATES` (a programming error, not a runtime
        condition -- see task-3-review.md F10). Also the one place that
        clears the resume latch on leaving `listening` (see
        `_clear_resume_latch`'s docstring for the other clear sites)."""
        assert new_state in _VALID_STATES, f"invalid HandsFreeState: {new_state!r}"
        if self._state == "listening" and new_state != "listening":
            self._resume_latched = False
        self._state = new_state
        self._emit(ModeChanged(new_state))

    def _exit(self) -> None:
        self._emit(ExitLoop())
        self._transition("idle")

    def _reopen_capture_if_closed(self) -> None:
        if not self._capture_open:
            self._emit(OpenCapture())
            self._capture_open = True

    def _cancel_countdown(self) -> None:
        """Reset countdown-arming state (both the anchor and the
        monotonicity tracker) without necessarily transitioning -- shared
        by every place a countdown is cancelled or newly armed."""
        self._armed_at = None
        self._last_countdown_remaining = None

    def _clear_resume_latch(self) -> None:
        """Bound the resume latch's lifetime (task-3-review.md F3): besides
        one-shot consumption in `on_voice_final` and `_transition`'s clear
        on leaving `listening`, a turn boundary (`on_capture_ended`) also
        clears it explicitly -- a latch may only cancel a final within the
        SAME capture, never across turns."""
        self._resume_latched = False

    def _begin_awaiting_reply(self) -> None:
        """Shared by countdown expiry and `on_capture_ended`'s
        with-segments path -- both are "a send is happening now". The send
        stops the capture in BOTH capture modes (task-3-review.md F1): V2's
        stop-and-send flow only ever runs from the recorder's own stop
        success tail, so there is no way for a capture to survive it."""
        self._emit(RequestStopAndSend())
        if self._capture_open:
            self._emit(CloseCapture())
            self._capture_open = False
        # A successful send resets the reopen-once flag: only *consecutive*
        # empty-limit endings exit the loop, not a lifetime cap.
        self._capture_limit_reopened = False
        self._reply_finished = False
        self._sequencer_drained = False
        self._cancel_countdown()
        self._awaiting_armed_at = None
        self._awaiting_watchdog_disarmed = False
        self._reply_abandoned_by_watchdog = False
        self._clear_resume_latch()
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
        """Enter the hands-free loop. On a genuine entry from `idle`,
        `capture_live=True` keeps an already-open capture as the first
        turn and `capture_live=False` opens one (`OpenCapture`). Calling
        `enter()` again while already running (state != `idle`) instead
        re-confirms `listening` from this controller's own `capture_open`
        bookkeeping (never trusting a possibly-stale `capture_live`
        argument -- see the module docstring's "Re-entry" section and
        task-3-review.md F7), and silences any reply audio still in
        flight first.

        Args:
            capture_live: Whether the microphone is already open at the
                moment of a genuine entry from `idle`. Ignored on a
                re-entry (state != `idle`) -- see the docstring above for
                why a stale caller-supplied snapshot cannot be trusted
                there.
        """
        was_speaking = self._state == "speaking"
        from_idle = self._state == "idle"
        self._cancel_countdown()
        self._awaiting_armed_at = None
        self._awaiting_watchdog_disarmed = False
        self._reply_abandoned_by_watchdog = False
        self._clear_resume_latch()
        self._capture_limit_reopened = False
        self._reply_finished = False
        self._sequencer_drained = False
        if was_speaking:
            self._emit(SilenceSpeech())
        if from_idle:
            self._capture_open = capture_live
            if not capture_live:
                self._emit(OpenCapture())
                self._capture_open = True
        else:
            self._reopen_capture_if_closed()
        self._transition("listening")

    def tick(self, now: float) -> None:
        """Injected-clock heartbeat. A no-op outside `countdown` and
        `awaiting_reply`. See the module docstring's "Countdown timing" and
        "`awaiting_reply` watchdog" sections for what each does.

        Args:
            now: The caller's own monotonic clock reading. Expected to be
                non-decreasing across calls; see the module docstring's
                "Countdown timing" section for the (defensive-only)
                handling of a backwards step.
        """
        if self._state == "countdown":
            self._tick_countdown(now)
            return
        if self._state == "awaiting_reply":
            self._tick_awaiting_reply(now)
            return

    def _tick_countdown(self, now: float) -> None:
        """The first call after arming adopts `now` as the anchor
        (`on_voice_final` cannot record one itself -- it has no `now`
        parameter). A backwards clock step re-anchors to the earliest
        `now` seen so far, and every emitted value is additionally clamped
        to `[0, send_delay_seconds]` and to be no larger than the last
        value emitted -- see task-3-review.md F6."""
        if self._armed_at is None:
            self._armed_at = now
        else:
            self._armed_at = min(self._armed_at, now)
        remaining = self._send_delay_seconds - (now - self._armed_at)
        remaining = max(0.0, min(remaining, self._send_delay_seconds))
        if self._last_countdown_remaining is not None:
            remaining = min(remaining, self._last_countdown_remaining)
        if remaining <= 0:
            self._begin_awaiting_reply()
            return
        self._last_countdown_remaining = remaining
        self._emit(CountdownTick(remaining))

    def _tick_awaiting_reply(self, now: float) -> None:
        """See the module docstring's "`awaiting_reply` watchdog" section:
        a send that silently refuses (`on_reply_started()` never arrives
        at all) would otherwise hang this state forever. Mirrors
        `_tick_countdown`'s anchoring: the first call after entering
        `awaiting_reply` adopts `now` as the anchor (elapsed is always 0
        relative to a same-call anchor, so this never expires on its own
        first call, exactly like the countdown's arming tick). A no-op
        once `on_reply_started()` has disarmed it (task-3-review.md N1) --
        this watchdog guards only the send -> `on_reply_started()` gap,
        never generation in progress. On genuine expiry, suppresses this
        reply's speech (task-3-review.md N2) exactly like
        `on_reply_failed()` would, and records the abandonment so a LATE
        `on_reply_started()` for this same reply can re-affirm it."""
        if self._awaiting_watchdog_disarmed:
            return
        if self._awaiting_armed_at is None:
            self._awaiting_armed_at = now
        if now - self._awaiting_armed_at >= AWAITING_REPLY_DEADLINE_SECONDS:
            self._reply_abandoned_by_watchdog = True
            self._emit(SuppressReplySpeech())
            self._reopen_capture_if_closed()
            self._transition("listening")

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
        self._cancel_countdown()
        self._transition("countdown")

    def on_segment_no_final(self) -> None:
        """A segment finished transcribing to nothing, so no `on_voice_
        final()` will ever arrive for it (see `Audio/dictation_service_
        lazy.py`'s `_transcribe_segment_audio`: a blank/whitespace-only
        result -- routine for room noise or a too-short VAD sliver --
        fires neither a partial nor a final).

        Qodo review (task-5 follow-up): a resume latched via
        `on_speech_resumed()` (carried finding #1, see `on_voice_final`
        above) is normally consumed by the very next `on_voice_final()`.
        But `_notify_segment_transcribing(done=True)` -- the ONLY other
        unconditional signal this blank outcome ever produces -- fires
        BEFORE a text segment's own final would (finalization runs in
        `_processing_loop`, after `_transcribe_segment_audio` returns), so
        clearing the latch there instead would re-admit a stale final the
        latch exists to drop. Without a dedicated blank-segment signal, a
        latch armed while THIS segment was (unknowingly) about to produce
        nothing would sit armed indefinitely and incorrectly swallow the
        NEXT REAL segment's `on_voice_final()`, silently dropping a whole
        turn's countdown.

        One-shot, exactly like `on_voice_final`'s own consumption: only
        clears a latch armed for the segment that just ended, never a
        later one. Meaningful only while `listening` (mirroring
        `on_voice_final`'s own state gate) -- a no-op everywhere else,
        including when nothing is latched."""
        if self._state != "listening":
            return
        self._resume_latched = False

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
            self._cancel_countdown()
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
        name is out of scope here -- the screen dispatches it normally.

        Args:
            name: The recognized command name (e.g. `"stop"`,
                `"new-paragraph"`) -- one of `console_voice_input.py`'s
                `COMMAND_PHRASES` values.
        """
        if name == "stop":
            self._exit()

    def on_capture_ended(self, had_segments: bool, limit_hit: bool) -> None:
        """The recorder's own service-side limit (60s wall-clock cutoff or
        buffer cap) ended the capture out from under the loop. Meaningful
        only when `limit_hit` is True; a non-limit ending is out of this
        method's contract (normal endings arrive via `on_voice_final`/
        countdown expiry instead) and is silently ignored.

        Always corrects `_capture_open = False` first, in EVERY state --
        the capture genuinely ended regardless of what the FSM was doing
        (task-3-review.md F2). In `listening`/`countdown` this is a turn
        boundary: with segments pending, treated exactly like countdown
        expiry; with nothing captured, reopen once rather than exit
        outright (see the module docstring's reopen-once section). In
        `awaiting_reply`/`speaking` (a reply is outstanding -- typically
        the acoustic-mode mic hitting its own limit mid-reply) no send is
        issued regardless of `had_segments`, since a send mid-reply would
        interleave two turns; acoustic mode still reopens the mic (per
        `on_reply_started`'s rule) so the user is not left deaf mid-turn,
        default mode has nothing further to do. This mid-reply reopen is
        routed through the SAME consecutive-empty-limit ceiling as
        `listening`/`countdown` (task-3-review.md N3) -- `speaking` has no
        watchdog of its own, so this ceiling is the only bound on it: an
        ending WITH segments resets the ceiling (it is not a "silent room"
        ending, even without a send for it), a SECOND consecutive
        empty-limit ending exits the loop exactly as it would elsewhere.

        Args:
            had_segments: Whether the capture had already-finalized
                segments pending when the limit hit.
            limit_hit: Whether this ending was the recorder's own
                service-side limit, as opposed to a normal ending (which
                does not call this method at all). False is a documented
                no-op, not a distinct code path.
        """
        if not limit_hit:
            return
        self._capture_open = False  # the capture already ended, any state
        if self._state in ("awaiting_reply", "speaking"):
            if not self._acoustic_barge_in:
                return  # default mode: nothing to reopen, no ceiling to track
            if had_segments:
                self._capture_limit_reopened = (
                    False  # a real capture; reset the ceiling
                )
                self._reopen_capture_if_closed()
                return
            if not self._capture_limit_reopened:
                self._capture_limit_reopened = True
                self._reopen_capture_if_closed()
                return
            self._exit()
            return
        if self._state not in ("listening", "countdown"):
            return  # idle: nothing further to do
        self._clear_resume_latch()  # turn boundary either way below
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
            self._cancel_countdown()
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
        """Reply generation has begun streaming. Never itself changes
        `state` -- that is `on_first_utterance()`'s job, once the
        sequencer actually queues speakable text.

        While `awaiting_reply`: DISARMS the `awaiting_reply` watchdog
        outright (task-3-review.md N1 -- positive proof the send did not
        silently refuse; see the module docstring's "`awaiting_reply`
        watchdog" section for why this must disarm rather than merely
        re-anchor). In acoustic mode this is also exactly when the mic
        reopens (`_begin_awaiting_reply()` just closed it, unconditionally,
        for the send): mid-reply is the acoustic mode's whole point, since
        the user may interrupt by speaking (task-3-review.md F1). A no-op
        for the reopen in default mode (the mic stays closed until the
        reply drains).

        Outside `awaiting_reply`: if the watchdog already abandoned THIS
        reply (`_reply_abandoned_by_watchdog`), this is a LATE arrival for
        a reply the loop gave up on -- re-emits `SuppressReplySpeech`
        (idempotently) rather than doing nothing, since in the real wiring
        this input is exactly the point that would otherwise call the
        sentence sequencer's own `begin_reply()`, which resets its
        suppression latch and would let the late reply speak after all
        (task-3-review.md N2). A true no-op in every other case (e.g. a
        stray call in `idle`/`listening` with nothing outstanding at all)."""
        if self._state == "awaiting_reply":
            self._awaiting_watchdog_disarmed = True
            if self._acoustic_barge_in:
                self._reopen_capture_if_closed()
            return
        if self._reply_abandoned_by_watchdog:
            self._emit(SuppressReplySpeech())

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
