"""Tests for `tldw_chatbook.Chat.console_hands_free.HandsFreeController`.

The controller is a pure, headless finite-state machine (no Textual, no
wall-clock, no direct audio/TTS imports): every input is a plain method
call, every output is an injected `emit(intent)` call, and time only ever
enters through `tick(now)` with a caller-supplied float. That split is what
makes the countdown/resume race conditions below deterministically testable
without a running app, a real clock, or a real dictation service (see the
hands-free-loop design doc,
`Docs/superpowers/specs/2026-08-02-hands-free-loop-design.md`, and
task-3-brief.md's "Carried findings" section for the three review-mandated
behaviors pinned explicitly below).

The task-3 brief's Step 1 tests through `test_reply_drained_reopens_capture`
are reproduced verbatim (including its semicolon-joined statement style,
each suppressed with `# noqa: E702` rather than reformatted, to keep the
brief's exact code intact); every other test below is this task's own,
written one statement per line.
"""

from tldw_chatbook.Chat.console_hands_free import (
    CloseCapture,
    CountdownTick,
    ExitLoop,
    HandsFreeController,
    ModeChanged,
    OpenCapture,
    RequestStopAndSend,
    SilenceSpeech,
    SuppressReplySpeech,
)


def mk(**kw):
    events = []
    c = HandsFreeController(emit=events.append, send_delay_seconds=1.5, **kw)
    return c, events


# ---------------------------------------------------------------------------
# Verbatim load-bearing tests (task-3 brief, Step 1)
# ---------------------------------------------------------------------------


def test_voice_final_arms_countdown_and_expiry_sends():
    c, ev = mk(); c.enter(capture_live=True)  # noqa: E702
    c.on_voice_final(); assert c.state == "countdown"  # noqa: E702
    c.tick(now=0.0); c.tick(now=1.6)  # noqa: E702
    assert any(isinstance(e, RequestStopAndSend) for e in ev)
    assert c.state == "awaiting_reply"


def test_speech_resumed_cancels_countdown():
    c, ev = mk(); c.enter(capture_live=True)  # noqa: E702
    c.on_voice_final(); c.tick(now=0.0)  # noqa: E702
    c.on_speech_resumed()
    assert c.state == "listening"
    c.tick(now=5.0)
    assert not any(isinstance(e, RequestStopAndSend) for e in ev)


def test_resume_vs_expiry_race_arrival_order_wins():
    c, ev = mk(); c.enter(capture_live=True)  # noqa: E702
    c.on_voice_final(); c.tick(now=0.0); c.tick(now=1.6)   # expiry first  # noqa: E702
    c.on_speech_resumed()                                   # late resume: rides next turn
    assert c.state == "awaiting_reply"


def test_keypress_in_speaking_barges_in():
    c, ev = mk(); c.enter(capture_live=True)  # noqa: E702
    c.on_voice_final(); c.tick(0.0); c.tick(1.6)  # noqa: E702
    c.on_reply_started(); c.on_first_utterance(); assert c.state == "speaking"  # noqa: E702
    c.on_composer_key()
    assert any(isinstance(e, SilenceSpeech) for e in ev)
    assert any(isinstance(e, OpenCapture) for e in ev)
    assert c.state == "listening"


def test_keypress_in_awaiting_suppresses_speech():
    c, ev = mk(); c.enter(capture_live=True)  # noqa: E702
    c.on_voice_final(); c.tick(0.0); c.tick(1.6)  # noqa: E702
    c.on_composer_key()
    assert any(isinstance(e, SuppressReplySpeech) for e in ev)
    assert c.state == "listening"


def test_limit_hit_with_segments_sends_without_segments_reopens_once_then_exits():
    c, ev = mk(); c.enter(capture_live=True)  # noqa: E702
    c.on_capture_ended(had_segments=True, limit_hit=True)
    assert any(isinstance(e, RequestStopAndSend) for e in ev)
    c2, ev2 = mk(); c2.enter(capture_live=True)  # noqa: E702
    c2.on_capture_ended(had_segments=False, limit_hit=True)
    assert any(isinstance(e, OpenCapture) for e in ev2)
    c2.on_capture_ended(had_segments=False, limit_hit=True)
    assert any(isinstance(e, ExitLoop) for e in ev2)


def test_reply_drained_reopens_capture():
    c, ev = mk(); c.enter(capture_live=True)  # noqa: E702
    c.on_voice_final(); c.tick(0.0); c.tick(1.6)  # noqa: E702
    c.on_reply_started(); c.on_first_utterance()  # noqa: E702
    c.on_reply_finished(); c.on_sequencer_drained()  # noqa: E702
    assert c.state == "listening"
    assert any(isinstance(e, OpenCapture) for e in ev)


# -- exit reachable from every state (parametrized over five builders) ------


def build_idle():
    events = []
    c = HandsFreeController(emit=events.append, send_delay_seconds=1.5)
    return c, events


def build_listening():
    c, ev = mk()
    c.enter(capture_live=True)
    return c, ev


def build_countdown():
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_voice_final()
    return c, ev


def build_awaiting():
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=1.6)
    return c, ev


def build_speaking():
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=1.6)
    c.on_reply_started()
    c.on_first_utterance()
    return c, ev


def test_exit_reachable_from_every_state():
    for build in (build_idle, build_listening, build_countdown,
                  build_awaiting, build_speaking):
        c, ev = build()
        c.on_exit_request()
        assert any(isinstance(e, ExitLoop) for e in ev), c.state


# ---------------------------------------------------------------------------
# "Plus:" list (task-3 brief, Step 1)
# ---------------------------------------------------------------------------


def test_acoustic_barge_in_silences_and_returns_to_listening_when_opted_in():
    c, ev = mk(acoustic_barge_in=True)
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(0.0)
    c.tick(1.6)
    c.on_reply_started()
    c.on_first_utterance()
    assert c.state == "speaking"
    c.on_speech_resumed()
    assert any(isinstance(e, SilenceSpeech) for e in ev)
    assert c.state == "listening"


def test_speech_resumed_in_speaking_is_ignored_without_acoustic_opt_in():
    c, ev = mk()  # acoustic_barge_in defaults False
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(0.0)
    c.tick(1.6)
    c.on_reply_started()
    c.on_first_utterance()
    assert c.state == "speaking"
    c.on_speech_resumed()
    assert not any(isinstance(e, SilenceSpeech) for e in ev)
    assert c.state == "speaking"


def test_voice_command_stop_exits():
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_voice_command("stop")
    assert any(isinstance(e, ExitLoop) for e in ev)
    assert c.state == "idle"


def test_voice_command_other_names_do_not_exit():
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_voice_command("hands free")
    assert not any(isinstance(e, ExitLoop) for e in ev)
    assert c.state == "listening"


def test_countdown_ticks_emit_monotonically_decreasing_remaining():
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=0.4)
    c.tick(now=0.9)
    ticks = [e.remaining for e in ev if isinstance(e, CountdownTick)]
    assert len(ticks) == 3
    assert ticks == sorted(ticks, reverse=True)
    assert all(r > 0 for r in ticks)


def test_reopen_once_flag_resets_after_a_successful_send():
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_capture_ended(had_segments=False, limit_hit=True)  # reopen #1
    assert any(isinstance(e, OpenCapture) for e in ev)
    assert not any(isinstance(e, ExitLoop) for e in ev)

    # A genuine successful send-and-reply cycle happens in between.
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=1.6)
    assert c.state == "awaiting_reply"
    c.on_reply_started()
    c.on_first_utterance()
    c.on_reply_finished()
    c.on_sequencer_drained()
    assert c.state == "listening"

    ev.clear()
    # Limit-hit-with-no-segments again: the reopen-once flag must have reset,
    # so this is a fresh reopen, not the second-consecutive exit.
    c.on_capture_ended(had_segments=False, limit_hit=True)
    assert any(isinstance(e, OpenCapture) for e in ev)
    assert not any(isinstance(e, ExitLoop) for e in ev)


def test_reply_failed_suppresses_speech_reopens_capture_and_returns_to_listening():
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=1.6)
    assert c.state == "awaiting_reply"
    c.on_reply_failed()
    assert any(isinstance(e, SuppressReplySpeech) for e in ev)
    assert any(isinstance(e, OpenCapture) for e in ev)
    assert c.state == "listening"


def test_reply_failed_mid_speech_also_recovers_to_listening():
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=1.6)
    c.on_reply_started()
    c.on_first_utterance()
    assert c.state == "speaking"
    c.on_reply_failed()
    assert any(isinstance(e, SuppressReplySpeech) for e in ev)
    assert c.state == "listening"


# ---------------------------------------------------------------------------
# Carried findings (task-3 brief, "Carried findings binding this task")
# ---------------------------------------------------------------------------


def test_resume_latched_in_listening_cancels_the_immediately_following_voice_final():
    """Carried finding #1: the silence gate zeroes its timestamp BEFORE the
    seconds-long transcription runs, so `VoiceSpeechResumed` for the NEXT
    utterance's speech can arrive before the `VoiceFinal` for the utterance
    that is still being transcribed. A resume seen while `listening` must be
    latched and consume the very next `on_voice_final` as already-cancelled
    (stay in `listening`, no countdown armed) rather than arming a countdown
    while the user is already mid-sentence again."""
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_speech_resumed()   # arrives before the VoiceFinal it must cancel
    c.on_voice_final()
    assert c.state == "listening"
    c.tick(now=0.0)
    c.tick(now=5.0)
    assert not any(isinstance(e, RequestStopAndSend) for e in ev)
    assert not any(isinstance(e, CountdownTick) for e in ev)


def test_resume_latch_is_consumed_only_once():
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_speech_resumed()
    c.on_voice_final()   # consumes the latch, stays listening
    assert c.state == "listening"
    c.on_voice_final()   # a genuine final this time: latch already spent
    assert c.state == "countdown"


def test_sequencer_drained_arriving_in_listening_is_a_noop():
    """Carried finding #2: a barged-in reply still fires `reply_completed()`
    on the sequencer, so `on_sequencer_drained` can arrive after the
    controller has already returned to `listening` (via the keypress
    barge-in path) -- and the reply's text stream finishing independently
    delivers `on_reply_finished` too, so BOTH completion flags end up true
    while already `listening`. Must be a no-op — in particular it must NOT
    emit a second `OpenCapture`."""
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(0.0)
    c.tick(1.6)
    c.on_reply_started()
    c.on_first_utterance()
    assert c.state == "speaking"
    c.on_composer_key()  # barge-in: SilenceSpeech + OpenCapture -> listening
    assert c.state == "listening"

    ev.clear()
    c.on_reply_finished()     # the reply's text stream still completes later
    c.on_sequencer_drained()  # ...and the sequencer's own reply_completed()
    assert c.state == "listening"
    assert ev == []


def test_class_docstring_documents_degraded_mode_voice_cancel_caveat():
    """Carried finding #3: degraded (no-webrtcvad) mode never emits
    `VoiceSpeechResumed`, so countdown-cancel-by-voice is inert there
    (keypress cancel still works; the FSM itself needs no special case) --
    the controller's docstring must say so explicitly rather than let the
    behavior go undocumented."""
    doc = (HandsFreeController.__doc__ or "").lower()
    assert "webrtcvad" in doc
    assert "keypress" in doc


def test_mode_changed_docstring_does_not_universally_promise_voice_cancel():
    """Carried finding #3 (continued): `ModeChanged` payloads must not
    promise voice-cancel universally -- its docstring stays purely
    descriptive of the state label and makes no cancellation claim, since
    that claim would be false in degraded (no-webrtcvad) mode."""
    doc = ModeChanged.__doc__ or ""
    assert "voice" not in doc.lower()


# ---------------------------------------------------------------------------
# Additional transition-matrix coverage
# ---------------------------------------------------------------------------


def test_composer_key_in_countdown_cancels():
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.on_composer_key()
    assert c.state == "listening"
    c.tick(now=5.0)
    assert not any(isinstance(e, RequestStopAndSend) for e in ev)


def test_capture_ended_without_limit_hit_is_ignored():
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_capture_ended(had_segments=True, limit_hit=False)
    assert c.state == "listening"
    assert not any(isinstance(e, RequestStopAndSend) for e in ev)
    assert not any(isinstance(e, OpenCapture) for e in ev)


def test_events_before_enter_are_ignored():
    c, ev = mk()
    assert c.state == "idle"
    c.on_voice_final()
    c.on_speech_resumed()
    c.on_composer_key()
    c.on_reply_finished()
    c.on_sequencer_drained()
    c.tick(now=1.0)
    assert c.state == "idle"
    assert ev == []


def test_enter_from_idle_capture_opens_it():
    c, ev = mk()
    c.enter(capture_live=False)
    assert any(isinstance(e, OpenCapture) for e in ev)
    assert c.state == "listening"


def test_enter_from_live_capture_does_not_reopen_it():
    c, ev = mk()
    c.enter(capture_live=True)
    assert not any(isinstance(e, OpenCapture) for e in ev)
    assert c.state == "listening"


def test_expiry_closes_capture_unless_acoustic_opt_in():
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=1.6)
    assert any(isinstance(e, CloseCapture) for e in ev)

    c2, ev2 = mk(acoustic_barge_in=True)
    c2.enter(capture_live=True)
    c2.on_voice_final()
    c2.tick(now=0.0)
    c2.tick(now=1.6)
    assert not any(isinstance(e, CloseCapture) for e in ev2)


def test_zero_sentence_reply_short_circuits_to_listening():
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=1.6)
    assert c.state == "awaiting_reply"
    # A reply that produces no speakable sentences never reaches "speaking".
    c.on_reply_started()
    c.on_reply_finished()
    c.on_sequencer_drained()
    assert c.state == "listening"
    assert any(isinstance(e, OpenCapture) for e in ev)


def test_mode_changed_emitted_on_every_transition():
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_voice_final()
    modes = [e.state for e in ev if isinstance(e, ModeChanged)]
    assert modes == ["listening", "countdown"]
