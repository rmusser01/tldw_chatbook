"""Headless finite-state machine for the Console V4 realtime hands-free
voice loop: the OpenAI Realtime session stays connected for the whole
conversation, and this controller tracks its lifecycle (connect -> live ->
turn -> reply -> live -> ...) and the barge-in / reconnect / idle-ceiling
rules layered on top of it.

Deliberately free of Textual, wall-clock, and direct
audio/session/WebSocket imports -- exactly like its V3 sibling,
`HandsFreeController` (`Chat/console_hands_free.py`): every input is a
plain method call (`enter`, `on_session_ready`, `on_turn_committed`, ...),
every output is one of the SAME frozen-dataclass intents V3 emits
(`ModeChanged`, `ExitLoop`, `SilenceSpeech`, imported from
`console_hands_free.py` rather than redefined here -- see that module's
`ModeChanged`/`ExitLoop` docstrings for the V4-only, additive `reason`
field), handed to an injected `emit` callable, and the only notion of time
is `tick(now)`, driven by a caller-supplied float. Task 5's wiring enacts
`mic_gated` by syncing the mic tap on every `ModeChanged` and translates a
`SilenceSpeech` intent into sink-abort + `cancel_response(played_ms)` on the
realtime session -- this controller never touches audio itself, only emits
intents (see `.superpowers/sdd/2026-08-04-realtime-voice-engine/
task-4-brief.md`).

## States

`idle -> connecting -> live -> thinking -> speaking -> live -> ...`, with
`reconnecting` layered in when the transport drops mid-loop. `exit` (an
`ExitLoop` intent, landing back in `idle`) is reachable from every state via
`on_exit_request()`, and also arrives with a specific `reason` from
`on_connect_failed()`, a second consecutive `on_transport_closed(error=True)`
within one loop entry, or `tick()`'s idle ceiling.

## `mic_gated`

Exposed read-only for the wiring to sync the mic tap against on every
`ModeChanged`. In default (keyboard-only) barge-in mode, the mic is gated
(should not stream to the provider) exactly while a reply is outstanding
(`thinking` or `speaking`) -- mirroring V3's mic/speaker exclusion rule.
In acoustic barge-in mode the mic is never gated: it stays hot for the
whole live loop so `on_speech_started` (server-side VAD) can interrupt a
reply the same way a composer keypress can in either mode.

## Barge-in

Both barge-in kinds -- a composer keypress (`on_keypress`, either mode) and
server-side VAD (`on_speech_started`, acoustic mode only) -- share the same
effect while `thinking` or `speaking`: emit `SilenceSpeech` (stop the
audio output; generation itself is untouched, exactly like V3's own
`SilenceSpeech` contract) then transition back to `live`. Task 5's wiring
is what turns that `SilenceSpeech` into an actual `cancel_response
(played_ms)` call against the realtime session. Outside those two states
both inputs are a no-op -- there is no reply in flight to interrupt.

## Reconnect-once-then-exit

`on_transport_closed(error=True)` covers an unexpected transport drop (as
opposed to `error=False`, which is this controller's own deliberate close
and is always a no-op). The FIRST such drop within one loop entry
transitions to `reconnecting` and emits `ModeChanged("reconnecting",
reason="reconnecting")`, giving the wiring a chance to open a fresh
session and re-seed context. A SECOND drop within that SAME loop entry --
regardless of whether a reconnect succeeded in between and the loop was
briefly back to `live`/`thinking`/`speaking` -- gives up outright:
`ExitLoop(reason="connection-lost")`. The once-flag is scoped to one loop
entry: only a fresh `enter()` (a brand new loop, after a full exit) resets
it, exactly matching the design doc's "no infinite retry" rule.
`on_connect_failed()` shares this SAME give-up exit when it arrives while
`reconnecting` (Task 5's wiring reuses one connect code path for both the
first connect and every reconnect attempt) -- the reconnect-once allowance
is already spent by definition whenever state is `reconnecting`, so a
failed reconnect attempt is exactly as terminal as a second transport
drop would be; without this, a failed reconnect attempt would strand the
loop in `reconnecting` forever, since `tick()`'s idle ceiling only ever
evaluates `live`.

## Idle ceiling (`tick`)

`tick(now)` is a no-op outside `live` -- in particular it can never fire
while `thinking` or `speaking`, however long a reply takes, because a long
reply is not activity-starved (see the spec's idle-ceiling rule: "never
fires while a reply is active"). `_last_activity` anchors the window the
same way V3's `_armed_at` anchors its countdown: `enter()` and
`on_session_ready()` take no `now` argument, so they mark the anchor
pending (`None`) and the FIRST subsequent `tick(now)` call adopts that
`now` as the anchor rather than firing immediately. `on_turn_committed(now)`
and `on_reply_done(now)` DO receive `now` directly and re-anchor to it
immediately -- both are genuine activity. A barge-in returning to `live`
(`_barge_in_if_reply_outstanding`, driven by `on_keypress`/
`on_speech_started`) also marks the anchor pending rather than leaving the
pre-reply anchor in place -- a barge-in IS a reply-audio end, which the
spec's idle definition counts as activity, and stamping nothing here would
otherwise let the very next `tick()` fire `idle-timeout` against a
long-stale anchor moments after the user's own keypress proved the session
attended (review F1). `on_speech_started` while ALREADY `live` (no reply
outstanding, so no barge-in either) marks the anchor pending too, in BOTH
barge-in modes (task-2361, review M3) -- a user who starts an utterance
just before the ceiling but has not yet reached `on_turn_committed` (the
turn is still being transcribed server-side) must not be cut off
mid-sentence; a speaker is a speaker regardless of whether
`acoustic_barge_in` is enabled, since that flag only ever governs
INTERRUPTING an outstanding reply, never idle bookkeeping. Once
`now - _last_activity >= idle_timeout_seconds` while still `live`,
`tick()` emits `ExitLoop(reason="idle-timeout")` -- an unattended session
must not bill indefinitely; a genuinely silent session (no turn commit,
no reply, no speech_started at all) still exits exactly as before.
"""

from __future__ import annotations

from typing import Callable, Literal

from loguru import logger

from tldw_chatbook.Chat.console_hands_free import ExitLoop, ModeChanged, SilenceSpeech

RealtimeLoopState = Literal[
    "idle", "connecting", "live", "thinking", "speaking", "reconnecting"
]

_VALID_STATES: tuple[RealtimeLoopState, ...] = (
    "idle",
    "connecting",
    "live",
    "thinking",
    "speaking",
    "reconnecting",
)

#: States in which a reply is outstanding -- barge-in (`on_keypress`,
#: acoustic-mode `on_speech_started`) is only meaningful here, and default
#: mode's `mic_gated` is True exactly here.
_REPLY_OUTSTANDING_STATES: tuple[RealtimeLoopState, ...] = ("thinking", "speaking")

#: States a transport drop can meaningfully originate from -- `idle` (never
#: entered) and `connecting` (the FIRST connect attempt; see
#: `on_connect_failed` instead) are out of `on_transport_closed`'s contract.
_TRANSPORT_ACTIVE_STATES: tuple[RealtimeLoopState, ...] = (
    "live",
    "thinking",
    "speaking",
    "reconnecting",
)


class RealtimeLoopController:
    """Pure headless FSM driving the Console V4 realtime hands-free loop.
    See the module docstring for the full state/timing/reconnect design.

    Args:
        emit: Called with exactly one intent (`ModeChanged`, `ExitLoop`, or
            `SilenceSpeech`, all imported from `console_hands_free.py`) at a
            time as the controller reacts to inputs.
        acoustic_barge_in: When True, `on_speech_started()` (server-side
            VAD) also barges in during `thinking`/`speaking` (not just a
            composer keypress via `on_keypress()`), and `mic_gated` is
            always False -- the mic stays hot for the whole live loop
            rather than being gated while a reply is outstanding.
        idle_timeout_seconds: How long `tick()` may observe no activity
            while `live` before giving up on an unattended session
            (`ExitLoop(reason="idle-timeout")`). Never applies outside
            `live` -- see the module docstring's "Idle ceiling" section.
    """

    def __init__(
        self,
        emit: Callable[[object], None],
        *,
        acoustic_barge_in: bool,
        idle_timeout_seconds: float,
    ) -> None:
        self._emit = emit
        self._acoustic_barge_in = acoustic_barge_in
        self._idle_timeout_seconds = idle_timeout_seconds

        self._state: RealtimeLoopState = "idle"
        self._last_activity: float | None = None
        self._reconnect_attempted: bool = False

    # -- public state -----------------------------------------------------

    @property
    def state(self) -> RealtimeLoopState:
        """The controller's current state label."""
        return self._state

    @property
    def mic_gated(self) -> bool:
        """Whether the wiring should keep the mic tap gated right now.

        Returns:
            Always False in acoustic barge-in mode (the mic stays hot for
            the whole live loop). In default mode, True exactly while a
            reply is outstanding (`thinking` or `speaking`) -- mirroring
            V3's mic/speaker exclusion rule -- and False otherwise.
        """
        if self._acoustic_barge_in:
            return False
        return self._state in _REPLY_OUTSTANDING_STATES

    # -- transition chokepoint ---------------------------------------------

    def _transition(
        self, new_state: RealtimeLoopState, *, reason: str | None = None
    ) -> None:
        """The SOLE place `self._state` is assigned outside `__init__`.
        Always emits `ModeChanged`, mirroring V3's `_transition()`
        chokepoint (`console_hands_free.py`). Rejects any target outside
        `_VALID_STATES` (a programming error, not a runtime condition).

        Args:
            new_state: The state to transition into.
            reason: Optional reason to attach to the emitted `ModeChanged`
                (currently only `"reconnecting"`, for the first transport
                drop within a loop entry -- see the module docstring).
        """
        assert new_state in _VALID_STATES, f"invalid RealtimeLoopState: {new_state!r}"
        self._state = new_state
        self._emit(ModeChanged(new_state, reason=reason))

    def _exit(self, reason: str | None = None) -> None:
        """Emit `ExitLoop(reason)` then transition to `idle` (which itself
        emits a trailing `ModeChanged("idle")`, exactly like V3's own
        `_exit()`) -- the sole path back to `idle` from any other state.

        Args:
            reason: Optional reason to attach to the emitted `ExitLoop`
                (e.g. `"connect-failed"`, `"connection-lost"`,
                `"idle-timeout"`); None for a plain user-initiated exit.
        """
        self._emit(ExitLoop(reason=reason))
        self._transition("idle")

    # -- public inputs ------------------------------------------------------

    def enter(self) -> None:
        """Enter the realtime loop from `idle`, moving to `connecting`.
        Idempotent: calling this again while the loop is already running
        (state != `idle`) is a no-op -- there is no re-entry semantics to
        reconcile here (unlike V3's countdown/capture bookkeeping), so a
        stray duplicate call must not re-arm anything mid-loop.

        Resets the per-loop-entry reconnect-once bookkeeping (see the
        module docstring's "Reconnect-once-then-exit" section) and marks
        the idle-ceiling anchor pending -- see "Idle ceiling" -- so a fresh
        loop always starts with a clean slate.
        """
        if self._state != "idle":
            return
        self._reconnect_attempted = False
        self._last_activity = None
        self._transition("connecting")

    def on_session_ready(self) -> None:
        """The realtime session is connected and ready. Moves `connecting`
        or `reconnecting` to `live`; a no-op in every other state (e.g. a
        stray duplicate arrival while already `live`).

        Marks the idle-ceiling anchor pending (no `now` argument here,
        mirroring V3's `on_voice_final` arming pattern) -- the first
        subsequent `tick(now)` call adopts its `now` as the anchor rather
        than firing immediately.
        """
        if self._state not in ("connecting", "reconnecting"):
            return
        self._last_activity = None
        self._transition("live")

    def on_connect_failed(self) -> None:
        """The realtime session's connection attempt failed outright.

        Meaningful during BOTH the first connect (`connecting`) and a
        reconnect attempt (`reconnecting` -- Task 5's wiring shares the
        same connect code path for both, so this callback WILL arrive here
        too): a `connecting` failure exits with `reason="connect-failed"`;
        a `reconnecting` failure routes to the SAME give-up exit a second
        `on_transport_closed(error=True)` would (`reason=
        "connection-lost"`) -- the reconnect-once allowance is already
        spent by definition whenever this state is `reconnecting`, since
        `on_transport_closed`'s own first-failure path is what put it
        there (review F2: a `reconnecting`-only no-op left the loop
        permanently stranded there forever, since `tick()` only ever
        evaluates the idle ceiling while `live`).

        A no-op in every other state (e.g. a stray arrival with no connect
        attempt outstanding at all, `live` or `speaking`) -- logged at
        debug so a wiring bug that fires this unexpectedly is still
        observable without ever changing behavior.
        """
        if self._state == "connecting":
            self._exit(reason="connect-failed")
            return
        if self._state == "reconnecting":
            self._exit(reason="connection-lost")
            return
        logger.debug(
            f"RealtimeLoopController.on_connect_failed: ignored, no connect "
            f"attempt outstanding: op=on_connect_failed state={self._state!r}"
        )

    def on_turn_committed(self, now: float) -> None:
        """The user's input turn was committed server-side. Moves `live` to
        `thinking`; a no-op outside `live`.

        Args:
            now: The caller's monotonic clock reading at the moment of
                commit -- genuine activity, so this re-anchors the
                idle-ceiling window immediately (see the module docstring's
                "Idle ceiling" section).
        """
        if self._state != "live":
            return
        self._last_activity = now
        self._transition("thinking")

    def on_reply_started(self) -> None:
        """Reply generation has begun streaming. Never itself changes
        `state` in V4 -- `on_first_audio()` is what moves `thinking` to
        `speaking`, once audio actually starts arriving -- and disarms
        nothing (V4 has no `awaiting_reply` watchdog; the reply-lifecycle
        semantics that make this safe are Task 2's: `on_reply_done` never
        fires for a cancelled response, so there is nothing here for this
        signal to guard against). A pure no-op, kept only to mirror the
        session's own callback shape 1:1.
        """
        return

    def on_first_audio(self) -> None:
        """The first chunk of reply audio arrived. Moves `thinking` to
        `speaking`; a no-op outside `thinking` (e.g. a stray duplicate
        arrival while already `speaking`)."""
        if self._state != "thinking":
            return
        self._transition("speaking")

    def on_reply_done(self, now: float) -> None:
        """The assistant's current reply has fully completed (per Task 2's
        semantics, this never fires for a response the client itself
        cancelled -- see the module docstring's `on_reply_started`
        section -- so every arrival here is a genuine end-of-reply). Moves
        `thinking` or `speaking` back to `live`; a no-op otherwise.

        Args:
            now: The caller's monotonic clock reading at reply completion
                -- genuine activity, so this re-anchors the idle-ceiling
                window immediately.
        """
        if self._state not in _REPLY_OUTSTANDING_STATES:
            return
        self._last_activity = now
        self._transition("live")

    def on_speech_started(self) -> None:
        """Server-side VAD detected the user starting to speak.

        Two independent things happen here, in order:

        1. Idle-anchor refresh (task-2361, V4 final review M3): while
           `live`, this marks the idle-ceiling anchor pending (the same
           idiom `enter()`/`on_session_ready()` use) REGARDLESS of
           `acoustic_barge_in` -- a speaker mid-utterance is never
           activity-starved just because the loop is in default (keyboard-
           only) barge-in mode. Without this, a user who started talking
           just before the ceiling but had not yet reached `on_turn_
           committed` could be ejected by `tick()` with "idle for N
           minutes" mid-sentence, even though they were plainly attending.
           Scoped to `live` only: while `thinking`/`speaking` (reachable
           here only in acoustic mode, since that is the only mode where
           the mic stays hot then), the barge-in path below already
           refreshes the anchor as part of returning to `live` -- see
           `_barge_in_if_reply_outstanding`.
        2. Barge-in (see the module docstring's "Barge-in" section): only
           when `acoustic_barge_in` is enabled AND a reply is outstanding
           (`thinking` or `speaking`); a no-op in default mode (keyboard-
           only barge-in) and a no-op with nothing to interrupt. This half
           is UNCHANGED by the anchor refresh above.
        """
        if self._state == "live":
            self._last_activity = None
        if not self._acoustic_barge_in:
            return
        self._barge_in_if_reply_outstanding()

    def on_keypress(self) -> None:
        """A composer keypress arrived. Barges in (see the module
        docstring's "Barge-in" section) in EITHER barge-in mode, whenever a
        reply is outstanding (`thinking` or `speaking`) -- V3's
        keyboard-first, speaker-safe-by-default discipline always keeps
        this path live. A no-op otherwise (ordinary typing while `live`,
        `connecting`, etc.)."""
        self._barge_in_if_reply_outstanding()

    def _barge_in_if_reply_outstanding(self) -> None:
        """Shared by `on_speech_started` (once already gated on acoustic
        mode) and `on_keypress` (ungated): emit `SilenceSpeech` -- stop the
        audio output; generation itself is untouched -- then return to
        `live`. A no-op if no reply is outstanding.

        Review F1: a barge-in IS a reply-audio end (`SilenceSpeech` stops
        the audio right there), which the idle-ceiling spec counts as
        activity -- but neither `_transition` nor `live` itself stamps a
        `now` here (this method takes none), so leaving the OLD, possibly
        long-stale activity anchor in place made the very next `tick()`
        measure elapsed time from before the reply even started, firing
        `idle-timeout` moments after the user's own keypress proved the
        session attended. Fixed with this file's own established idiom --
        `enter()`/`on_session_ready()` already mark the anchor pending
        (`None`) rather than stamping a `now` they don't have -- so the
        NEXT `tick(now)` adopts a fresh anchor instead of an ExitLoop."""
        if self._state not in _REPLY_OUTSTANDING_STATES:
            return
        self._emit(SilenceSpeech())
        self._last_activity = None
        self._transition("live")

    def on_transport_closed(self, *, error: bool) -> None:
        """The realtime session's transport closed. See the module
        docstring's "Reconnect-once-then-exit" section for the full policy.

        Args:
            error: True for an unexpected drop (network failure, provider
                closing the socket); False for this controller's own
                deliberate close (e.g. as part of `_exit()`'s own teardown),
                which is always a no-op here -- it is not a failure to
                recover from.
        """
        if not error:
            return
        if self._state not in _TRANSPORT_ACTIVE_STATES:
            return
        if self._reconnect_attempted:
            self._exit(reason="connection-lost")
            return
        self._reconnect_attempted = True
        self._transition("reconnecting", reason="reconnecting")

    def on_exit_request(self) -> None:
        """Esc / mic press / explicit user exit: exit from any state,
        including `idle` itself (a harmless redundant `ExitLoop`) --
        mirrors V3's `on_exit_request` exactly."""
        self._exit()

    def tick(self, now: float) -> None:
        """Injected-clock heartbeat driving the idle ceiling. A no-op
        outside `live` -- see the module docstring's "Idle ceiling"
        section for the full anchoring rules (including why `enter()` and
        `on_session_ready()`, having no `now` argument, mark the anchor
        pending rather than setting it directly).

        Args:
            now: The caller's own monotonic clock reading. Expected to be
                non-decreasing across calls.
        """
        if self._state != "live":
            return
        if self._last_activity is None:
            self._last_activity = now
        elapsed = now - self._last_activity
        if elapsed >= self._idle_timeout_seconds:
            self._exit(reason="idle-timeout")
