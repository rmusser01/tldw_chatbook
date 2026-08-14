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

from tldw_chatbook.Chat.console_auto_speak import (
    AutoSpeakContext,
    AutoSpeakDisposition,
    decide_auto_speak,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_hands_free import (
    AWAITING_REPLY_DEADLINE_SECONDS,
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
from tldw_chatbook.Chat.console_speech_preferences import ConsoleSpeechPreferences

DESTINATION = "sha256:" + "a" * 64


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
    # F4 (review): this reopen was unpinned -- deleting it passed 29/29.
    # The mic was closed on send; a keypress-suppressed reply must reopen
    # it or the loop lands in `listening` with a permanently closed mic.
    assert any(isinstance(e, OpenCapture) for e in ev)
    assert c.capture_open is True


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


def test_segment_no_final_consumes_a_latch_so_the_next_real_final_arms_countdown():
    """Qodo review (task-5 follow-up): the silence gate zeroes its
    timestamp BEFORE the (possibly seconds-long) transcription runs, so a
    resume for the NEXT segment's speech can arrive while THIS segment is
    still transcribing -- carried finding #1's whole premise. But when
    that in-flight segment turns out blank, no `on_voice_final()` ever
    arrives to consume the latch (a blank/whitespace-only result fires no
    final at all -- see `Audio/dictation_service_lazy.py`'s
    `_transcribe_segment_audio`). Without `on_segment_no_final()`, that
    latch would sit armed indefinitely and incorrectly swallow the NEXT
    REAL segment's `on_voice_final()`, silently dropping a whole turn's
    countdown -- this fails today (before the fix) because the swallowed
    final never arms a countdown at all."""
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_speech_resumed()    # latches, before the blank segment's own final
    c.on_segment_no_final()  # THIS segment transcribed to nothing
    c.on_voice_final()       # a genuine final, for the NEXT segment
    assert c.state == "countdown"


def test_segment_no_final_is_a_noop_outside_listening():
    """Mirrors `on_voice_final`'s own state gate: meaningless anywhere but
    `listening`, and must never touch state elsewhere."""
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=1.6)
    assert c.state == "awaiting_reply"
    c.on_segment_no_final()  # must not raise or change state
    assert c.state == "awaiting_reply"


def test_segment_no_final_without_a_latched_resume_is_a_noop():
    """No latch armed at all: consuming nothing is still safe, and a
    genuine final right after still arms the countdown normally -- the
    text-segment path (a real final consuming its own latch) is untouched
    by this method, pinned separately by `test_resume_latch_is_consumed_
    only_once` above."""
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_segment_no_final()  # nothing latched; must not raise
    assert c.state == "listening"
    c.on_voice_final()
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


def test_expiry_always_closes_capture_regardless_of_acoustic_mode():
    """F1 (review): the send stops the capture in BOTH modes -- V2's
    stop-and-send flow only ever runs from the recorder's own stop
    success tail, so `_capture_open` must go False on every send. Acoustic
    mode's difference is WHEN the mic reopens (`on_reply_started`, not
    "never closed in the first place")."""
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=1.6)
    assert any(isinstance(e, CloseCapture) for e in ev)
    assert c.capture_open is False

    c2, ev2 = mk(acoustic_barge_in=True)
    c2.enter(capture_live=True)
    c2.on_voice_final()
    c2.tick(now=0.0)
    c2.tick(now=1.6)
    assert any(isinstance(e, CloseCapture) for e in ev2)
    assert c2.capture_open is False


def test_acoustic_mode_reopens_at_reply_started_not_at_drained():
    c, ev = mk(acoustic_barge_in=True)
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=1.6)
    assert c.capture_open is False  # closed by the send, like non-acoustic
    ev.clear()
    c.on_reply_started()
    assert any(isinstance(e, OpenCapture) for e in ev)
    assert c.capture_open is True


def test_non_acoustic_mode_does_not_reopen_at_reply_started():
    c, ev = mk()  # acoustic_barge_in defaults False
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=1.6)
    assert c.capture_open is False
    ev.clear()
    c.on_reply_started()
    assert not any(isinstance(e, OpenCapture) for e in ev)
    assert c.capture_open is False


def test_acoustic_mode_resume_in_speaking_works_across_multiple_turns():
    """F1 (review, HIGH): reproduces the reviewer's turn-2 sequence, and
    checks the model's OWN consistency at each step, not just that
    `on_speech_resumed` happens to still emit `SilenceSpeech` (nothing
    gates that emission on `capture_open`, so a stale model wouldn't
    change it -- the actual defect is the model believing the mic stayed
    open across a send that, in reality, always stops it). Mutation
    evidence: reverting `_begin_awaiting_reply` to skip `CloseCapture` in
    acoustic mode (the pre-fix, sticky-`_capture_open` model) makes the
    per-turn `CloseCapture` assertion fail from turn 1 onward."""
    c, ev = mk(acoustic_barge_in=True)
    c.enter(capture_live=True)
    for turn in range(3):
        base = float(turn * 10)
        c.on_voice_final()
        ev.clear()
        c.tick(now=base)
        c.tick(now=base + 1.6)
        assert c.state == "awaiting_reply", f"turn {turn}"
        assert any(isinstance(e, CloseCapture) for e in ev), \
            f"turn {turn}: send did not close the mic in the model"
        assert c.capture_open is False, f"turn {turn}: model out of sync with reality"
        ev.clear()
        c.on_reply_started()
        assert any(isinstance(e, OpenCapture) for e in ev), \
            f"turn {turn}: mic did not reopen for the reply"
        assert c.capture_open is True, f"turn {turn}: mic did not reopen for the reply"
        c.on_first_utterance()
        assert c.state == "speaking", f"turn {turn}"
        ev.clear()
        c.on_speech_resumed()
        assert any(isinstance(e, SilenceSpeech) for e in ev), \
            f"turn {turn}: acoustic resume did not silence -- loop went deaf"
        assert c.state == "listening", f"turn {turn}"


def test_capture_ended_in_awaiting_reply_corrects_model_issues_no_send():
    """F2 (review): a capture-ended report while a reply is outstanding
    must correct the mic model but must NOT issue a send mid-reply (that
    would interleave two turns)."""
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=1.6)
    assert c.state == "awaiting_reply"
    ev.clear()
    c.on_capture_ended(had_segments=True, limit_hit=True)
    assert c.state == "awaiting_reply"
    assert not any(isinstance(e, RequestStopAndSend) for e in ev)


def test_capture_ended_in_speaking_acoustic_reopens_the_mic():
    """F2 (review): the acoustic-mode mic can end (service-side limit)
    while the reply is still speaking -- must correct the model AND
    reopen (per F1's on_reply_started rule), not drop the user's speech
    silently with no intent at all."""
    c, ev = mk(acoustic_barge_in=True)
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=1.6)
    c.on_reply_started()
    c.on_first_utterance()
    assert c.state == "speaking"
    assert c.capture_open is True
    ev.clear()
    c.on_capture_ended(had_segments=True, limit_hit=True)
    # Corrects the model (closed by the service-side limit) and reopens it
    # (acoustic mode) within this one call -- the intermediate "closed"
    # instant is not separately observable, only the net effect is.
    assert c.state == "speaking"
    assert not any(isinstance(e, RequestStopAndSend) for e in ev)
    assert any(isinstance(e, OpenCapture) for e in ev)
    assert not any(isinstance(e, CloseCapture) for e in ev)  # a fact, not a command
    assert c.capture_open is True


def test_capture_ended_in_speaking_non_acoustic_only_corrects_the_model():
    c, ev = mk()  # acoustic_barge_in defaults False
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=1.6)
    c.on_reply_started()
    c.on_first_utterance()
    assert c.state == "speaking"
    assert c.capture_open is False  # never reopened outside acoustic mode
    ev.clear()
    c.on_capture_ended(had_segments=False, limit_hit=True)
    assert c.capture_open is False
    assert not any(isinstance(e, OpenCapture) for e in ev)
    assert c.state == "speaking"


def test_capture_ended_in_speaking_acoustic_reopen_respects_the_ceiling():
    """N3 (review, LOW): the acoustic mid-reply reopen used to bypass the
    reopen-once ceiling entirely -- 4 consecutive empty-limit endings
    mid-reply meant 4 `OpenCapture`s and 0 `ExitLoop`. `speaking` has no
    watchdog of its own (by design -- keypress barge-in stays available),
    so this ceiling is the only bound on it. Routed through the SAME
    consecutive-empty-limit accounting as `listening`/`countdown`: a
    SECOND consecutive empty-limit ending exits, not a fourth."""
    c, ev = mk(acoustic_barge_in=True)
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=1.6)
    c.on_reply_started()
    c.on_first_utterance()
    assert c.state == "speaking"

    ev.clear()
    c.on_capture_ended(had_segments=False, limit_hit=True)  # 1st: reopen
    assert any(isinstance(e, OpenCapture) for e in ev)
    assert not any(isinstance(e, ExitLoop) for e in ev)
    assert c.state == "speaking"

    ev.clear()
    c.on_capture_ended(had_segments=False, limit_hit=True)  # 2nd consecutive: exit
    assert any(isinstance(e, ExitLoop) for e in ev)
    assert c.state == "idle"


def test_capture_ended_in_speaking_acoustic_with_segments_resets_the_ceiling():
    """N3 (continued): a limit-hit ending WITH segments mid-reply is not
    a "silent room" ending -- it must reset the ceiling, same as a
    successful send does, not count toward the consecutive-empty streak."""
    c, ev = mk(acoustic_barge_in=True)
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=1.6)
    c.on_reply_started()
    c.on_first_utterance()
    assert c.state == "speaking"

    c.on_capture_ended(had_segments=False, limit_hit=True)  # 1st empty: reopen
    c.on_capture_ended(had_segments=True, limit_hit=True)  # a real capture: resets
    ev.clear()
    c.on_capture_ended(had_segments=False, limit_hit=True)  # fresh 1st again: reopen, not exit
    assert any(isinstance(e, OpenCapture) for e in ev)
    assert not any(isinstance(e, ExitLoop) for e in ev)
    assert c.state == "speaking"


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


# ---------------------------------------------------------------------------
# Review fix wave (task-3-review.md): F1 High, F2-F6 Medium, F7-F10 Low
# ---------------------------------------------------------------------------


def test_resume_latch_does_not_survive_a_turn_boundary():
    """F3 (review): reproduces the reviewer's cross-turn swallow (P2). A
    resume latched in `listening` before ANY final is pending must not
    outlive the turn boundary a send creates -- otherwise it swallows the
    NEXT turn's genuine `on_voice_final`."""
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_speech_resumed()  # latches with nothing pending to cancel
    assert c.state == "listening"
    c.on_capture_ended(had_segments=True, limit_hit=True)  # turn boundary: a send
    assert c.state == "awaiting_reply"
    c.on_reply_started()
    c.on_first_utterance()
    c.on_reply_finished()
    c.on_sequencer_drained()
    assert c.state == "listening"

    ev.clear()
    c.on_voice_final()  # next turn's genuine final must NOT be swallowed
    assert c.state == "countdown"


def test_resume_latch_cleared_after_a_relatch_following_consumption():
    """F3 (continued); N4 (review, test-honesty fix): a second variant
    that layers a consume-then-relatch sequence before the SAME turn
    boundary (`on_capture_ended`) the sibling test above already uses --
    NOT a genuinely different "countdown-expiry send" path, despite an
    earlier draft of this docstring claiming that. Reaching `countdown`
    (and its `tick()`-driven expiry) while latched is unreachable BY
    CONSTRUCTION: arming a countdown requires an UNlatched
    `on_voice_final` (a latched one just consumes the latch and stays
    `listening`, per carried finding #1), so "latched" and "about to
    expire via `tick()`" can never coexist -- there is no
    countdown-expiry variant of this test to write."""
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_speech_resumed()  # latches
    c.on_voice_final()  # consumes it (carried finding #1), stays listening
    assert c.state == "listening"
    c.on_speech_resumed()  # latches again, nothing pending
    # A full turn happens without ever consuming this second latch via
    # on_voice_final -- it must not leak into the NEXT turn either.
    c.on_capture_ended(had_segments=True, limit_hit=True)
    c.on_reply_started()
    c.on_first_utterance()
    c.on_reply_finished()
    c.on_sequencer_drained()
    assert c.state == "listening"

    ev.clear()
    c.on_voice_final()
    assert c.state == "countdown"


def test_awaiting_reply_deadline_not_yet_expired_stays_awaiting():
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=1.6)
    assert c.state == "awaiting_reply"
    c.tick(now=1.6)  # anchors _awaiting_armed_at = 1.6
    assert c.state == "awaiting_reply"
    c.tick(now=1.6 + AWAITING_REPLY_DEADLINE_SECONDS - 1.0)  # still inside the window
    assert c.state == "awaiting_reply"


def test_awaiting_reply_deadline_expiry_reopens_and_returns_to_listening():
    """F5 (review): a silently-refused send must not hang the loop in
    `awaiting_reply` forever ("never-started still fires at 30s" -- N1's
    pin). Mutation evidence: removing the deadline watchdog makes this
    fail (state stays `awaiting_reply` under an arbitrarily large `now`).
    N2 (review): expiry must also suppress the reply's speech, exactly
    like `on_reply_failed` -- a late reply must not be able to speak into
    the reopened mic."""
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=1.6)
    assert c.state == "awaiting_reply"
    ev.clear()
    c.tick(now=1.6)  # first awaiting_reply tick: anchors, does not expire yet
    assert c.state == "awaiting_reply"
    c.tick(now=1.6 + AWAITING_REPLY_DEADLINE_SECONDS + 0.1)  # past the deadline
    assert c.state == "listening"
    assert any(isinstance(e, OpenCapture) for e in ev)
    assert any(isinstance(e, SuppressReplySpeech) for e in ev)
    assert any(isinstance(e, ModeChanged) and e.state == "listening" for e in ev)


def test_watchdog_disarms_after_on_reply_started_and_never_fires():
    """N1 (review, MED): `on_reply_started()` is positive proof the send
    did NOT silently refuse -- 30s to the first SPEAKABLE sentence is
    routine (cold model load, a long thinking block, or a reply opening
    with a fenced code block the sequencer skips entirely), so the
    watchdog must disarm outright once generation is confirmed alive
    rather than keep ticking toward a false-positive abandonment.
    Mutation evidence: removing the disarm makes this fail (an
    arbitrarily late tick would still fire and abandon a live reply)."""
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=1.6)
    assert c.state == "awaiting_reply"
    c.on_reply_started()  # confirmed alive at t=10 (a slow cold-load reply)
    ev.clear()
    c.tick(now=1.6)  # first awaiting_reply tick: would anchor if not disarmed
    c.tick(now=1.6 + AWAITING_REPLY_DEADLINE_SECONDS + 100.0)  # way past 30s
    assert c.state == "awaiting_reply"  # never abandoned
    assert ev == []


def test_watchdog_expiry_then_late_reply_lifecycle_inputs_cannot_speak():
    """N2 (review, MED): the reviewer's exact compound sequence -- expiry,
    then a late reply eventually arriving anyway (`on_reply_started` /
    `on_first_utterance` / `on_reply_finished` in order) -- must never let
    speech reach the reopened mic, and the mic must stay ordinarily usable
    (a composer keypress remains a no-op in `listening`, exactly like
    normal typing). The late `on_reply_started` specifically must
    re-affirm suppression (idempotent): in the real wiring that input is
    what would otherwise reset the sentence sequencer's own suppression
    latch (`SentenceSequencer.begin_reply()`), which would let the late
    reply speak after all if this controller did not counteract it."""
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=1.6)
    c.tick(now=1.6)  # anchors the watchdog
    c.tick(now=1.6 + AWAITING_REPLY_DEADLINE_SECONDS + 0.1)  # expires
    assert c.state == "listening"
    assert any(isinstance(e, SuppressReplySpeech) for e in ev)

    ev.clear()
    c.on_reply_started()  # the abandoned reply shows up anyway
    assert any(isinstance(e, SuppressReplySpeech) for e in ev)  # re-affirmed
    assert c.state == "listening"  # no speech path opened

    ev.clear()
    c.on_first_utterance()  # its first "speakable" sentence, if any
    assert c.state == "listening"  # still cannot reach `speaking`
    assert ev == []

    ev.clear()
    c.on_reply_finished()
    c.on_sequencer_drained()
    assert c.state == "listening"
    assert ev == []

    ev.clear()
    c.on_composer_key()  # the mic stays ordinarily usable
    assert c.state == "listening"
    assert ev == []


def test_countdown_tick_clamped_and_nonincreasing_under_backwards_jitter():
    """F6 (review): reproduces the reviewer's backwards-jitter probe
    (100.0 / 100.5 / 99.0 / 100.2). Mutation evidence: removing the clamp
    lets `remaining` exceed `send_delay_seconds` and go non-monotonic."""
    c, ev = mk()  # send_delay_seconds=1.5
    c.enter(capture_live=True)
    c.on_voice_final()
    for now in (100.0, 100.5, 99.0, 100.2):
        c.tick(now=now)
    ticks = [e.remaining for e in ev if isinstance(e, CountdownTick)]
    assert len(ticks) == 4
    assert all(0.0 <= r <= 1.5 for r in ticks)
    assert all(ticks[i] >= ticks[i + 1] for i in range(len(ticks) - 1))


def test_backwards_reanchor_requires_full_delay_from_the_new_anchor():
    """N6 (review, INFO -- kept-behavior pin): the backwards re-anchor
    (`_armed_at = min(_armed_at, now)`) is a deliberate choice ("elapsed
    can never go negative"), with a documented side effect -- a clock
    step back effectively re-arms the countdown from the earlier point,
    so it can expire earlier in real terms than the full
    `send_delay_seconds` measured from the ORIGINAL arm. This pins the
    kept, intentional half: the FULL delay is still required, measured
    from the RE-ANCHORED (new, earlier) point -- the re-anchor does not
    additionally truncate past that. Mutation evidence: dropping the
    re-anchor (freezing `_armed_at` at its first value) makes this fail,
    since expiry would then need `now` to reach the ORIGINAL anchor plus
    the full delay, which this test's final tick falls well short of."""
    c, ev = mk()  # send_delay_seconds=1.5
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=10.0)  # arms at 10.0
    c.tick(now=9.0)  # a 1s backwards clock step: re-anchors to 9.0
    assert c.state == "countdown"  # not expired by the step itself
    c.tick(now=9.0 + 1.5 - 0.1)  # just short of the full delay from 9.0
    assert c.state == "countdown"
    c.tick(now=9.0 + 1.5 + 0.1)  # full delay elapsed from the re-anchored point
    assert c.state == "awaiting_reply"


def test_resume_latch_cleared_on_empty_limit_reopen_turn_boundary():
    """N5 (review, LOW, coverage): the `on_capture_ended` latch clear is
    the ONLY one of the three F3 clear sites that does unique work -- it
    is the sole `listening -> listening` self-transition turn boundary,
    where `_transition`'s "leaving listening" clear cannot fire (state
    never actually leaves `listening`). Reproduces the reviewer's exact
    probe: latch -> empty-limit reopen -> next capture's first final ->
    `countdown`. Mutation evidence: removing this specific clear survived
    45/45 without this dedicated pin."""
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_speech_resumed()  # latches, nothing pending
    c.on_capture_ended(had_segments=False, limit_hit=True)  # empty-limit reopen
    assert c.state == "listening"
    ev.clear()
    c.on_voice_final()  # next capture's first final must NOT be swallowed
    assert c.state == "countdown"


def test_enter_from_speaking_silences_before_resetting():
    """F7 (review): re-entering (or the wiring re-confirming) the loop
    while a reply is actively speaking must silence it first -- there is
    no AEC, so an unsilenced reply would otherwise transcribe itself once
    `listening` believes the mic is live again."""
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=1.6)
    c.on_reply_started()
    c.on_first_utterance()
    assert c.state == "speaking"
    ev.clear()
    c.enter(capture_live=True)
    assert any(isinstance(e, SilenceSpeech) for e in ev)
    assert c.state == "listening"


def test_enter_never_double_opens_an_already_open_capture():
    """F7 (continued): re-entering while already `listening` with
    `capture_live=False` must not blindly trust that stale claim over the
    controller's own bookkeeping and double-open the mic."""
    c, ev = mk()
    c.enter(capture_live=True)
    assert c.capture_open is True
    ev.clear()
    c.enter(capture_live=False)
    assert not any(isinstance(e, OpenCapture) for e in ev)
    assert c.capture_open is True
    assert c.state == "listening"


def test_mode_changed_full_cycle_all_five_payloads():
    """F8 (review): only `listening`/`countdown` were previously pinned;
    walk the whole cycle so `speaking`/`awaiting_reply`/`idle` payloads
    are pinned too (they drive the wiring task's chip states)."""
    c, ev = mk()
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=1.6)
    c.on_reply_started()
    c.on_first_utterance()
    c.on_reply_finished()
    c.on_sequencer_drained()
    c.on_exit_request()
    modes = [e.state for e in ev if isinstance(e, ModeChanged)]
    assert modes == [
        "listening", "countdown", "awaiting_reply", "speaking", "listening", "idle",
    ]


def test_reopen_never_double_opens_an_already_open_capture():
    """F9 (review): the "never double-opens" claim in the docstrings had
    no pin. Acoustic mode reopens once at `on_reply_started`; drained
    completion must see the mic already open and emit nothing further."""
    c, ev = mk(acoustic_barge_in=True)
    c.enter(capture_live=True)
    c.on_voice_final()
    c.tick(now=0.0)
    c.tick(now=1.6)
    ev.clear()
    c.on_reply_started()
    assert len([e for e in ev if isinstance(e, OpenCapture)]) == 1
    c.on_first_utterance()
    c.on_reply_finished()
    c.on_sequencer_drained()
    assert c.state == "listening"
    assert len([e for e in ev if isinstance(e, OpenCapture)]) == 1


def test_transition_rejects_an_invalid_state():
    """F10 (review): `_VALID_STATES` was dead. Use it as a real invariant
    on the sole state-mutation chokepoint."""
    c, ev = mk()
    c.enter(capture_live=True)
    try:
        c._transition("bogus")
    except AssertionError:
        return
    raise AssertionError("expected _transition to reject an invalid state")


def test_active_hands_free_loop_explicitly_owns_reply_speech() -> None:
    controller, _events = mk()
    controller.enter(capture_live=True)
    context = AutoSpeakContext(
        preferences=ConsoleSpeechPreferences(
            auto_speak=True,
            consent_destination=DESTINATION,
        ),
        destination_fingerprint=DESTINATION,
        active_session_id="active-session",
        hands_free_active=controller.state != "idle",
    )

    disposition = decide_auto_speak(
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="Ready.",
            status="complete",
        ),
        session_id="active-session",
        context=context,
    )

    assert disposition is AutoSpeakDisposition.HANDSFREE_OWNS
