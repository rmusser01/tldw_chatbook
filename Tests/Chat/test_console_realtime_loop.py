"""Tests for `tldw_chatbook.Chat.console_realtime_loop.RealtimeLoopController`.

Like its V3 sibling (`HandsFreeController`, see `Tests/Chat/
test_console_hands_free.py`), this controller is a pure, headless finite-
state machine: no Textual, no wall-clock, no direct audio/session imports.
Every input is a plain method call, every output is an injected
`emit(intent)` call, and time only ever enters through `tick(now)` with a
caller-supplied float -- see task-4-brief.md
(`.superpowers/sdd/2026-08-04-realtime-voice-engine/task-4-brief.md`) for
the state table and the exact rules each test below pins.

Task 5's wiring enacts `mic_gated` by syncing the mic tap on every
`ModeChanged`, and translates `SilenceSpeech` into sink-abort +
`cancel_response(played_ms)` -- this suite only proves the FSM emits the
right intents in the right order, never touches audio itself.
"""

from tldw_chatbook.Chat.console_hands_free import ExitLoop, ModeChanged, SilenceSpeech
from tldw_chatbook.Chat.console_realtime_loop import RealtimeLoopController


def _make(acoustic=False, idle=300.0):
    intents = []
    c = RealtimeLoopController(
        intents.append, acoustic_barge_in=acoustic, idle_timeout_seconds=idle
    )
    return c, intents


# ---------------------------------------------------------------------------
# Task-4 brief, Step 1 -- named tests
# ---------------------------------------------------------------------------


def test_enter_connect_ready_reaches_live_with_mode_intents():
    c, ev = _make()
    c.enter()
    assert c.state == "connecting"
    assert any(isinstance(e, ModeChanged) and e.state == "connecting" for e in ev)
    ev.clear()
    c.on_session_ready()
    assert c.state == "live"
    assert any(isinstance(e, ModeChanged) and e.state == "live" for e in ev)


def test_connect_failed_exits_with_reason():
    c, ev = _make()
    c.enter()
    ev.clear()
    c.on_connect_failed()
    assert c.state == "idle"
    exits = [e for e in ev if isinstance(e, ExitLoop)]
    assert len(exits) == 1
    assert exits[0].reason == "connect-failed"


def test_turn_committed_thinking_first_audio_speaking_done_live():
    c, ev = _make()
    c.enter()
    c.on_session_ready()
    ev.clear()
    c.on_turn_committed(now=1.0)
    assert c.state == "thinking"
    assert any(isinstance(e, ModeChanged) and e.state == "thinking" for e in ev)

    ev.clear()
    c.on_first_audio()
    assert c.state == "speaking"
    assert any(isinstance(e, ModeChanged) and e.state == "speaking" for e in ev)

    ev.clear()
    c.on_reply_done(now=2.0)
    assert c.state == "live"
    assert any(isinstance(e, ModeChanged) and e.state == "live" for e in ev)


def test_keypress_mid_speaking_emits_silence_then_live():
    c, ev = _make()
    c.enter()
    c.on_session_ready()
    c.on_turn_committed(now=0.0)
    c.on_first_audio()
    assert c.state == "speaking"

    ev.clear()
    c.on_keypress()
    assert any(isinstance(e, SilenceSpeech) for e in ev)
    assert any(isinstance(e, ModeChanged) and e.state == "live" for e in ev)
    assert c.state == "live"
    # SilenceSpeech must precede the ModeChanged that follows it.
    silence_idx = next(i for i, e in enumerate(ev) if isinstance(e, SilenceSpeech))
    live_idx = next(
        i for i, e in enumerate(ev) if isinstance(e, ModeChanged) and e.state == "live"
    )
    assert silence_idx < live_idx


def test_keypress_while_live_is_a_noop():
    c, ev = _make()
    c.enter()
    c.on_session_ready()
    ev.clear()
    c.on_keypress()
    assert ev == []
    assert c.state == "live"


def test_speech_started_barges_only_in_acoustic_mode():
    c, ev = _make(acoustic=False)
    c.enter()
    c.on_session_ready()
    c.on_turn_committed(now=0.0)
    c.on_first_audio()
    ev.clear()
    c.on_speech_started()
    assert ev == []
    assert c.state == "speaking"

    c2, ev2 = _make(acoustic=True)
    c2.enter()
    c2.on_session_ready()
    c2.on_turn_committed(now=0.0)
    c2.on_first_audio()
    ev2.clear()
    c2.on_speech_started()
    assert any(isinstance(e, SilenceSpeech) for e in ev2)
    assert any(isinstance(e, ModeChanged) and e.state == "live" for e in ev2)
    assert c2.state == "live"


def test_mic_gated_true_during_reply_default_mode():
    c, ev = _make(acoustic=False)
    c.enter()
    c.on_session_ready()
    assert c.mic_gated is False
    c.on_turn_committed(now=0.0)
    assert c.mic_gated is True
    c.on_first_audio()
    assert c.mic_gated is True
    c.on_reply_done(now=1.0)
    assert c.mic_gated is False


def test_mic_gated_always_false_acoustic_mode():
    c, ev = _make(acoustic=True)
    c.enter()
    c.on_session_ready()
    assert c.mic_gated is False
    c.on_turn_committed(now=0.0)
    assert c.mic_gated is False
    c.on_first_audio()
    assert c.mic_gated is False
    c.on_reply_done(now=1.0)
    assert c.mic_gated is False


def test_transport_error_reconnects_once_then_exits_with_reason():
    c, ev = _make()
    c.enter()
    c.on_session_ready()
    assert c.state == "live"

    ev.clear()
    c.on_transport_closed(error=True)
    assert c.state == "reconnecting"
    modes = [e for e in ev if isinstance(e, ModeChanged)]
    assert modes and modes[-1].state == "reconnecting"
    assert modes[-1].reason == "reconnecting"
    assert not any(isinstance(e, ExitLoop) for e in ev)

    ev.clear()
    c.on_transport_closed(error=True)  # second time in the same loop entry
    assert c.state == "idle"
    exits = [e for e in ev if isinstance(e, ExitLoop)]
    assert exits and exits[-1].reason == "connection-lost"


def test_idle_ceiling_fires_only_in_live_and_resets_on_activity():
    c, ev = _make(idle=10.0)
    c.enter()
    c.on_session_ready()
    c.tick(now=0.0)  # anchors last_activity at 0.0
    c.tick(now=5.0)  # elapsed 5 < 10
    assert c.state == "live"

    c.on_turn_committed(now=5.0)  # resets activity to 5.0, leaves "live"
    c.on_reply_done(now=8.0)  # resets activity to 8.0, back to "live"

    c.tick(now=9.0)  # elapsed since 8.0 == 1 < 10
    assert c.state == "live"

    ev.clear()
    c.tick(now=18.0)  # elapsed since 8.0 == 10 >= 10
    assert c.state == "idle"
    assert any(isinstance(e, ExitLoop) and e.reason == "idle-timeout" for e in ev)


def test_idle_ceiling_never_fires_mid_reply_even_past_deadline():
    c, ev = _make(idle=5.0)
    c.enter()
    c.on_session_ready()
    c.tick(now=0.0)

    c.on_turn_committed(now=0.0)
    assert c.state == "thinking"
    c.tick(now=1000.0)  # way past the deadline, but not "live"
    assert c.state == "thinking"
    assert not any(isinstance(e, ExitLoop) for e in ev)

    c.on_first_audio()
    assert c.state == "speaking"
    c.tick(now=2000.0)
    assert c.state == "speaking"
    assert not any(isinstance(e, ExitLoop) for e in ev)

    c.on_reply_done(now=2000.0)
    assert c.state == "live"
    c.tick(now=2000.0 + 5.0 - 0.1)  # just under the deadline from the reset anchor
    assert c.state == "live"


def test_exit_reachable_from_every_state():
    for build in (
        _build_idle,
        _build_connecting,
        _build_live,
        _build_thinking,
        _build_speaking,
        _build_reconnecting,
    ):
        c, ev = build()
        c.on_exit_request()
        assert any(isinstance(e, ExitLoop) for e in ev), c.state
        assert c.state == "idle"


def test_intents_after_exit_are_dropped():
    c, ev = _make()
    c.enter()
    c.on_session_ready()
    c.on_exit_request()
    assert c.state == "idle"

    ev.clear()
    c.on_turn_committed(now=0.0)
    c.on_first_audio()
    c.on_reply_done(now=1.0)
    c.on_speech_started()
    c.on_keypress()
    c.on_session_ready()
    c.on_connect_failed()
    c.on_transport_closed(error=True)
    c.tick(now=1_000_000.0)
    assert ev == []
    assert c.state == "idle"


# ---------------------------------------------------------------------------
# Build helpers for the exit-reachability matrix
# ---------------------------------------------------------------------------


def _build_idle():
    return _make()


def _build_connecting():
    c, ev = _make()
    c.enter()
    return c, ev


def _build_live():
    c, ev = _make()
    c.enter()
    c.on_session_ready()
    return c, ev


def _build_thinking():
    c, ev = _build_live()
    c.on_turn_committed(now=0.0)
    return c, ev


def _build_speaking():
    c, ev = _build_thinking()
    c.on_first_audio()
    return c, ev


def _build_reconnecting():
    c, ev = _build_live()
    c.on_transport_closed(error=True)
    return c, ev


# ---------------------------------------------------------------------------
# Additional coverage (mirrors the V3 suite's own extra transition-matrix
# section) -- one rule per test, per the brief.
# ---------------------------------------------------------------------------


def test_barge_in_also_works_while_thinking_before_first_audio():
    c, ev = _make(acoustic=True)
    c.enter()
    c.on_session_ready()
    c.on_turn_committed(now=0.0)
    assert c.state == "thinking"
    ev.clear()
    c.on_speech_started()
    assert any(isinstance(e, SilenceSpeech) for e in ev)
    assert c.state == "live"

    c2, ev2 = _make()
    c2.enter()
    c2.on_session_ready()
    c2.on_turn_committed(now=0.0)
    assert c2.state == "thinking"
    ev2.clear()
    c2.on_keypress()
    assert any(isinstance(e, SilenceSpeech) for e in ev2)
    assert c2.state == "live"


def test_reconnect_then_second_close_still_exits_even_after_reaching_live_again():
    c, ev = _make()
    c.enter()
    c.on_session_ready()
    c.on_transport_closed(error=True)  # 1st: -> reconnecting
    assert c.state == "reconnecting"
    c.on_session_ready()  # reconnect succeeded -> live
    assert c.state == "live"

    ev.clear()
    c.on_transport_closed(error=True)  # 2nd within the SAME loop entry
    assert c.state == "idle"
    assert any(isinstance(e, ExitLoop) and e.reason == "connection-lost" for e in ev)


def test_fresh_enter_after_exit_resets_reconnect_once_flag():
    c, ev = _make()
    c.enter()
    c.on_session_ready()
    c.on_transport_closed(error=True)  # 1st
    c.on_transport_closed(error=True)  # 2nd: exits
    assert c.state == "idle"

    ev.clear()
    c.enter()
    c.on_session_ready()
    assert c.state == "live"

    ev.clear()
    c.on_transport_closed(error=True)  # a brand new loop entry: this is the 1st again
    assert c.state == "reconnecting"
    assert not any(isinstance(e, ExitLoop) for e in ev)


def test_transport_closed_without_error_is_a_noop():
    c, ev = _make()
    c.enter()
    c.on_session_ready()
    ev.clear()
    c.on_transport_closed(error=False)
    assert ev == []
    assert c.state == "live"


def test_transport_closed_is_ignored_while_idle_or_connecting():
    c, ev = _make()
    c.on_transport_closed(error=True)  # idle: never entered
    assert ev == []
    assert c.state == "idle"

    c.enter()
    ev.clear()
    c.on_transport_closed(error=True)  # connecting: on_connect_failed is the path here
    assert ev == []
    assert c.state == "connecting"


def test_on_session_ready_is_a_noop_outside_connecting_or_reconnecting():
    c, ev = _make()
    c.enter()
    c.on_session_ready()
    assert c.state == "live"
    ev.clear()
    c.on_session_ready()  # already live: nothing to do
    assert ev == []
    assert c.state == "live"


def test_on_first_audio_is_a_noop_outside_thinking():
    c, ev = _make()
    c.enter()
    c.on_session_ready()
    ev.clear()
    c.on_first_audio()  # live, not thinking
    assert ev == []
    assert c.state == "live"


def test_on_reply_done_is_a_noop_outside_thinking_or_speaking():
    c, ev = _make()
    c.enter()
    c.on_session_ready()
    ev.clear()
    c.on_reply_done(now=1.0)
    assert ev == []
    assert c.state == "live"


def test_on_turn_committed_is_a_noop_outside_live():
    c, ev = _make()
    c.enter()  # connecting, not live
    ev.clear()
    c.on_turn_committed(now=1.0)
    assert ev == []
    assert c.state == "connecting"


def test_on_reply_started_is_a_pure_noop():
    c, ev = _make()
    c.enter()
    c.on_session_ready()
    c.on_turn_committed(now=0.0)
    ev.clear()
    c.on_reply_started()
    assert ev == []
    assert c.state == "thinking"


def test_enter_while_already_running_is_a_noop():
    c, ev = _make()
    c.enter()
    c.on_session_ready()
    ev.clear()
    c.enter()  # already live: not idle, so this must not re-arm the loop
    assert ev == []
    assert c.state == "live"


def test_events_before_enter_are_ignored():
    c, ev = _make()
    assert c.state == "idle"
    c.on_session_ready()
    c.on_connect_failed()
    c.on_turn_committed(now=0.0)
    c.on_first_audio()
    c.on_reply_done(now=1.0)
    c.on_speech_started()
    c.on_keypress()
    c.on_transport_closed(error=True)
    c.tick(now=1.0)
    assert c.state == "idle"
    assert ev == []


def test_mode_changed_full_cycle_payloads():
    c, ev = _make()
    c.enter()
    c.on_session_ready()
    c.on_turn_committed(now=0.0)
    c.on_first_audio()
    c.on_reply_done(now=1.0)
    c.on_exit_request()
    modes = [e.state for e in ev if isinstance(e, ModeChanged)]
    assert modes == [
        "connecting",
        "live",
        "thinking",
        "speaking",
        "live",
        "idle",
    ]


def test_persona_buddy_voice_adapter_consumes_the_real_fsm_cycle():
    """The headless FSM's emitted states are the adapter's trusted input."""
    from tldw_chatbook.Persona_Buddy.console_adapter import PersonaBuddyConsoleAdapter
    from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController

    buddy = PersonaBuddyController()
    adapter = PersonaBuddyConsoleAdapter(buddy)

    def emit(intent):
        if isinstance(intent, ModeChanged):
            adapter.voice_state("session-a", 7, intent.state)
        elif isinstance(intent, ExitLoop):
            adapter.release_voice("session-a", 7)

    controller = RealtimeLoopController(
        emit, acoustic_barge_in=False, idle_timeout_seconds=10.0
    )
    controller.enter()
    assert buddy.snapshot().state == "offline"
    controller.on_session_ready()
    assert buddy.snapshot().state == "listening"
    controller.on_turn_committed(now=0.0)
    assert buddy.snapshot().state == "thinking"
    controller.on_first_audio()
    assert buddy.snapshot().state == "speaking"
    controller.on_exit_request()
    assert buddy.snapshot().state == "idle"


def test_exit_request_reason_defaults_to_none():
    c, ev = _make()
    c.enter()
    c.on_session_ready()
    ev.clear()
    c.on_exit_request()
    exits = [e for e in ev if isinstance(e, ExitLoop)]
    assert exits[0].reason is None


# ---------------------------------------------------------------------------
# Review fix wave (task-4-review): F1 High, F2 Medium-High, F3 Low
# ---------------------------------------------------------------------------


def test_barge_in_mid_reply_refreshes_activity_so_idle_ceiling_does_not_fire_immediately():
    """F1 (review, HIGH): reproduces the reviewer's live finding --
    on_keypress() mid-reply at t=361 (long after a short idle_timeout would
    have elapsed while the reply was outstanding, which tick() correctly
    ignores per `test_idle_ceiling_never_fires_mid_reply_even_past_
    deadline`), then the FIRST tick(now) after the barge-in's return to
    `live` fired ExitLoop(reason="idle-timeout") half a second later --
    punishing the user for the exact keypress that proves the session is
    attended. The spec's idle definition counts "reply-audio end" as
    activity, and a barge-in IS a reply-audio end (SilenceSpeech stops the
    audio right there). Fixed with the file's own established idiom:
    `enter()`/`on_session_ready()` already mark the idle-ceiling anchor
    pending (None) rather than stamping a `now` they don't have --
    `_barge_in_if_reply_outstanding` now does the same before returning to
    `live`, so the NEXT tick(now) adopts it fresh instead of measuring
    against a stale pre-reply anchor. Mutation evidence: reverting the
    pending-anchor reset makes this fail exactly as the reviewer
    reproduced it (immediate exit on the tick right after the barge-in)."""
    c, ev = _make(idle=10.0)
    c.enter()
    c.on_session_ready()
    c.tick(now=0.0)  # anchors last_activity at 0.0
    c.on_turn_committed(now=0.0)
    c.on_first_audio()
    assert c.state == "speaking"

    # The reply has been outstanding well past what idle_timeout_seconds
    # would allow -- tick() correctly never evaluates it mid-reply.
    ev.clear()
    c.on_keypress()  # barge-in at t~361, analogous to the review's repro
    assert c.state == "live"
    assert any(isinstance(e, SilenceSpeech) for e in ev)

    ev.clear()
    c.tick(now=361.5)  # first tick after the barge-in: must NOT exit
    assert c.state == "live"
    assert not any(isinstance(e, ExitLoop) for e in ev)

    # But genuine silence for a full fresh idle window afterward still exits.
    ev.clear()
    c.tick(now=361.5 + 10.0)
    assert c.state == "idle"
    assert any(isinstance(e, ExitLoop) and e.reason == "idle-timeout" for e in ev)


def test_connect_failed_while_reconnecting_exits_with_connection_lost():
    """F2 (review, MEDIUM-HIGH): reproduces the reviewer's permanent-strand
    finding -- a failed RECONNECT attempt (Task 5's wiring shares the same
    connect code path for first-connect and reconnect, so on_connect_failed
    WILL arrive while `reconnecting`) used to be silently swallowed
    (on_connect_failed only acted on `connecting`), leaving the loop stuck
    in `reconnecting` forever: no ExitLoop, and tick() only ever evaluates
    `live`, so nothing could ever rescue it. The reconnect-once allowance
    is already spent by definition whenever this state is `reconnecting`
    (on_transport_closed's own first-failure path is what put it there), so
    a failed reconnect attempt routes to the SAME give-up exit a second
    on_transport_closed(error=True) would: ExitLoop(reason=
    "connection-lost")."""
    c, ev = _make()
    c.enter()
    c.on_session_ready()
    c.on_transport_closed(error=True)  # 1st failure -> reconnecting
    assert c.state == "reconnecting"

    ev.clear()
    c.on_connect_failed()  # the reconnect attempt itself failed to connect
    assert c.state == "idle"
    exits = [e for e in ev if isinstance(e, ExitLoop)]
    assert exits and exits[-1].reason == "connection-lost"


def test_connect_failed_is_a_noop_while_live_or_speaking():
    """F2 (continued): pins the OTHER half of the contract so it cannot
    silently flip either way later -- a stray on_connect_failed() arriving
    while there is no connect attempt outstanding at all (`live` or
    `speaking`) must still be a pure no-op (dropped, no state change)."""
    c, ev = _make()
    c.enter()
    c.on_session_ready()
    ev.clear()
    c.on_connect_failed()
    assert ev == []
    assert c.state == "live"

    c.on_turn_committed(now=0.0)
    c.on_first_audio()
    assert c.state == "speaking"
    ev.clear()
    c.on_connect_failed()
    assert ev == []
    assert c.state == "speaking"


def test_keypress_barges_in_identically_in_acoustic_mode():
    """F3 (review, LOW): keyboard barge-in must work identically whether or
    not acoustic_barge_in is enabled -- V3's keyboard-first,
    speaker-safe-by-default discipline is not conditional on the mode."""
    c, ev = _make(acoustic=True)
    c.enter()
    c.on_session_ready()
    c.on_turn_committed(now=0.0)
    c.on_first_audio()
    assert c.state == "speaking"

    ev.clear()
    c.on_keypress()
    assert any(isinstance(e, SilenceSpeech) for e in ev)
    assert any(isinstance(e, ModeChanged) and e.state == "live" for e in ev)
    assert c.state == "live"


# ---------------------------------------------------------------------------
# task-2361: on_speech_started while `live` refreshes the idle-ceiling
# anchor, in EITHER barge-in mode, so a user who starts speaking just
# before the deadline (speech_started, no commit yet) is never cut off
# mid-utterance by the cost guard (V4 final review M3).
# ---------------------------------------------------------------------------


def test_speech_started_while_live_refreshes_idle_anchor_both_modes():
    """Direction 1: a speaker is a speaker regardless of barge-in policy.
    Default mode's on_speech_started used to early-return the instant it
    saw `acoustic_barge_in is False` -- correct for barge-in (there is
    nothing to interrupt while `live`), but that same early return also
    skipped the idle-anchor refresh, so a user who started an utterance
    just before the ceiling could still be ejected mid-sentence with
    "idle for N minutes" the moment tick() next ran, because on_turn_
    committed (the only other activity signal) had not fired yet. Fixed
    by refreshing the anchor whenever on_speech_started arrives while
    `live`, BEFORE the mode gate that governs barge-in only."""
    for acoustic in (False, True):
        c, ev = _make(acoustic=acoustic, idle=10.0)
        c.enter()
        c.on_session_ready()
        c.tick(now=0.0)  # anchors last_activity at 0.0
        c.tick(now=9.9)  # just under the deadline
        assert c.state == "live", f"acoustic={acoustic}"

        c.on_speech_started()  # user starts talking right before the ceiling
        # No reply is outstanding while `live`, so this must never barge in
        # (barge-in semantics are unchanged by this fix).
        assert c.state == "live", f"acoustic={acoustic}"

        ev.clear()
        # Without the refresh this elapses 19.8s from the t=0.0 anchor,
        # comfortably past the 10s ceiling, and would fire idle-timeout.
        c.tick(now=19.8)
        assert c.state == "live", f"acoustic={acoustic}"
        assert not any(isinstance(e, ExitLoop) for e in ev), f"acoustic={acoustic}"


def test_genuinely_silent_session_still_exits_at_idle_ceiling():
    """Direction 2: the refresh must not defeat the ceiling outright -- a
    session that never sees on_speech_started at all (genuinely silent,
    or abandoned) still exits once the deadline elapses, exactly as
    before this fix."""
    c, ev = _make(idle=10.0)
    c.enter()
    c.on_session_ready()
    c.tick(now=0.0)  # anchors last_activity at 0.0

    ev.clear()
    c.tick(now=10.0)  # elapsed 10 >= 10, no activity of any kind in between
    assert c.state == "idle"
    assert any(isinstance(e, ExitLoop) and e.reason == "idle-timeout" for e in ev)


def test_speech_started_does_not_refresh_anchor_outside_live():
    """The refresh is scoped to `live` only -- on_speech_started arriving
    while `thinking`/`speaking` (acoustic mode; the only mode where the
    mic is hot there) must not touch `_last_activity` directly. That case
    is already handled by the existing barge-in path
    (`_barge_in_if_reply_outstanding`, which marks the anchor pending as
    part of returning to `live` -- see `test_barge_in_mid_reply_
    refreshes_activity_so_idle_ceiling_does_not_fire_immediately`), so
    this test only pins that the new `live`-scoped refresh does not
    double up or otherwise change that already-covered path's outcome."""
    c, ev = _make(acoustic=True, idle=10.0)
    c.enter()
    c.on_session_ready()
    c.on_turn_committed(now=0.0)
    c.on_first_audio()
    assert c.state == "speaking"

    ev.clear()
    c.on_speech_started()  # acoustic-mode barge-in, unrelated to this fix
    assert any(isinstance(e, SilenceSpeech) for e in ev)
    assert c.state == "live"


# ---------------------------------------------------------------------------
# Import lightness. Runs in fresh subprocesses -- importing anything else in
# this module (or an earlier-collected test) may already have pulled the
# heavy modules into `sys.modules`, which would make an in-process check
# meaningless. Mirrors `Tests/LLM_Calls/test_realtime_protocol.py`'s shape.
# ---------------------------------------------------------------------------


def test_console_realtime_loop_import_stays_headless():
    """This FSM is documented "free of Textual, wall-clock, and direct
    audio/session/WebSocket imports" (module docstring) -- that is what
    makes it unit-testable without an app, and what keeps it importable
    from the screen at module scope without dragging the realtime stack
    into every Console mount (final review M8).

    `textual` is deliberately NOT in the absence set: the baseline
    (`import tldw_chatbook.Chat`) already pulls it in through the package's
    own `__init__`, so asserting its absence here would assert something
    about a package this module does not own. `websockets`, `numpy` and
    `sounddevice` -- the realtime transport and the audio stack -- are
    exactly what this module must never reach, and none of them are in the
    baseline.
    """
    import subprocess
    import sys

    baseline_probe = (
        "import time\n"
        "t0 = time.monotonic()\n"
        "import tldw_chatbook.Chat\n"
        "print(time.monotonic() - t0)\n"
    )
    loop_probe = (
        "import sys, time\n"
        "import tldw_chatbook.Chat\n"
        "t0 = time.monotonic()\n"
        "import tldw_chatbook.Chat.console_realtime_loop\n"
        "elapsed = time.monotonic() - t0\n"
        "for name in ('websockets', 'numpy', 'sounddevice'):\n"
        "    assert name not in sys.modules, name + ' imported by "
        "console_realtime_loop'\n"
        "print(elapsed)\n"
    )

    baseline = subprocess.run(
        [sys.executable, "-c", baseline_probe],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr

    result = subprocess.run(
        [sys.executable, "-c", loop_probe],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    incremental_seconds = float(result.stdout.strip().splitlines()[-1])
    assert incremental_seconds < 0.2, (
        f"console_realtime_loop added {incremental_seconds:.3f}s on top of "
        "an already-imported tldw_chatbook.Chat -- check for a heavy import"
    )
