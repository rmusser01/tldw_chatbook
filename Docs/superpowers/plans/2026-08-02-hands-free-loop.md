# Hands-Free Conversation Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The full voice loop — speak, pause sends, the reply is spoken sentence-by-sentence, barge-in silences it — composed from the shipped dictation/command/sink stack.

**Architecture:** Two headless modules (`HandsFreeController` FSM with injected scheduler; `SentenceSequencer`), one new dictation event (`VoiceSpeechResumed`, service-level detection), one cooldown-free speech entry + utterance-completion signal in `tts_events`, thin `ChatScreen` wiring. Spec: `Docs/superpowers/specs/2026-08-02-hands-free-loop-design.md` — binding; read before Task 1.

**Tech Stack:** Python ≥3.11, existing dictation/sink/TTS stacks, pytest.

## Global Constraints

- The controller and sequencer have NO Textual imports, NO wall-clock (`tick()` + injected scheduler only), and NO direct audio/TTS imports — intents out via injected emit.
- Countdown expiry drives V2's existing stop-and-send flow (`_console_pending_voice_action = "send"`); there is NO second send path.
- Sequencer utterances must NOT pass `_enforce_cooldown_limit()` (tts_events.py:392; the ad-hoc branch at :436 runs it — the `TTSMessageSpeechRequestEvent` branch does not and is the shape to follow).
- Barge-in silences audio only — reply GENERATION is never cancelled by this feature.
- Reply speech ignores `dictation.spoken_feedback`; new config keys are exactly `dictation.handsfree_send_delay_seconds` (default 1.5, warn+fallback validation like siblings) and `dictation.acoustic_barge_in` (default false).
- Esc exit must not shadow the existing `escape → focus_console_composer_home` binding (chat_screen.py:1627) outside hands-free-active.
- Every sink/utterance reaches a terminal call on every path (phase-1 carried rule); the sequencer's flush is part of that guarantee.
- `Tests/UI/test_console_dictation.py` is byte-identical contract — run, never modify.
- Foreground pytest only (`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <files> -q -p no:randomly`); never whole directories; RED-first + mutation checks per repo discipline; conventional commits ending `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; never push; never `git stash`; never `git checkout --` a file with uncommitted work.

## File structure

| File | Responsibility |
|---|---|
| `tldw_chatbook/Chat/reply_sentence_sequencer.py` (create) | splitter, normalizer, queue discipline, flush |
| `tldw_chatbook/Chat/console_hands_free.py` (create) | the FSM |
| `tldw_chatbook/Audio/dictation_service_lazy.py` (modify) | `VoiceSpeechResumed` detection (service level) |
| `tldw_chatbook/Chat/console_voice_input.py` (modify) | `VoiceSpeechResumed` event + wiring through `_run_begin` |
| `tldw_chatbook/UI/Screens/chat_screen.py` (modify) | session adapter passthrough; loop wiring; chip states; keys |
| `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py` (modify) | cooldown-free utterance entry + completion events |
| Tests | `Tests/Chat/test_reply_sentence_sequencer.py`, `Tests/Chat/test_console_hands_free.py`, extend `Tests/Audio/test_dictation_vad_finalization.py`-adjacent + `Tests/TTS_Events/` + `Tests/UI/test_console_hands_free_wiring.py` |

Verified anchors: bridge delta seam `store.append_stream_chunk(assistant_message_id, visible)` (`console_agent_bridge.py:700,:706`); cooldown split (`tts_events.py:420-436` — message-speech branch skips it); legacy play runs via `_run_blocking_tts_io` (`tts_events.py` ~:1400s) with `play_audio_file` at :1349/:1568; Esc binding :1627; V2 pending-send flow (`chat_screen.py` `_console_pending_voice_action` :3086/:5411, dispatch ~:6146); wall timer `_console_dictation_timer` :3065 (60 s); segment-at-silence service (`dictation_service_lazy.py`, `last_speech_time`, `_audio_callback` ~:631); capture-generation gating (`ConsoleStreamingDictationSession._handle_event`, chat_screen ~:795).

---

### Task 1: `VoiceSpeechResumed`

**Files:** modify `dictation_service_lazy.py`, `console_voice_input.py`; extend the dictation service tests (new file `Tests/Audio/test_dictation_speech_resumed.py`) and `Tests/Chat/test_console_voice_input.py`.

**Interfaces produced:** frozen `VoiceSpeechResumed()` dataclass in `console_voice_input.py` (collision-safe name — check the Message-name trap doesn't apply: these are plain dataclasses, not Textual Messages); service callback `on_speech_resumed: Optional[Callable[[], None]]` (class-level default None, `__new__`-safety like siblings); controller wires it in `_run_begin` with the capture-generation token exactly as `on_final_transcript` is wired.

- [ ] **Step 1 (RED):** service-level tests with a fake recorder: (a) first frame of a capture does NOT emit resume (capture start ≠ resume); (b) frames within a continuous run emit nothing; (c) a frame arriving after `_finalize_current_segment` reset `last_speech_time` to 0 emits exactly ONE resume; (d) a frame after a >threshold delivery gap (without finalize — degraded/no-VAD case must NOT fire it spuriously every frame; decide: resume fires only on the post-finalize condition when VAD is unavailable... simpler rule, implement THIS: emit iff `last_speech_time == 0 and not first-frame-of-capture`) — pin the chosen rule; (e) callback exceptions swallowed like siblings. Controller-level: `VoiceSpeechResumed` emitted with the generation token; stale generation dropped by the session adapter (extend the adapter passthrough in chat_screen: forward it like `VoicePartial` — proves recognizer-ran is NOT set by it; it is a mic-side fact, not recognizer output — pin that explicitly).
- [ ] **Step 2:** run RED. **Step 3:** implement — in `_audio_callback`, before updating `last_speech_time`: `resumed = (self.last_speech_time == 0 and self._capture_saw_first_frame)`; set `self._capture_saw_first_frame = True` on first delivery; invoke `on_speech_resumed` guarded. Controller: wire in `_run_begin`; adapter: forward without touching `_heard_recognizer_output`/`_segments`.
- [ ] **Step 4:** green. **Step 5 (mutation):** make first-frame emit resume → (a) fails; drop the guard → (e) fails. **Step 6:** ruff; commit `feat(dictation): emit VoiceSpeechResumed on the silence-to-speech transition`.

---

### Task 2: `SentenceSequencer`

**Files:** create `Chat/reply_sentence_sequencer.py`, `Tests/Chat/test_reply_sentence_sequencer.py`.

**Interfaces produced:**
```python
seq = SentenceSequencer(speak=fn, stop_speech=fn)   # both injected callables
seq.feed(delta: str) -> None        # streamed text
seq.reply_completed() -> None       # flush the final partial sentence
seq.utterance_finished(ok: bool) -> None   # completion signal from the speech path
seq.flush() -> None                 # barge-in: clear queue + stop_speech() if in-flight
seq.drained: bool                   # True when queue empty AND nothing in flight
on_drained: Optional[Callable]      # fired when the above becomes True post-completion
```
`speak(text)` is called with exactly one utterance at a time; the next only after `utterance_finished`. Pure module: no Textual, no TTS imports.

- [ ] **Step 1 (RED)** — the load-bearing cases, verbatim:

```python
def test_sentences_emit_one_at_a_time_gated_on_completion():
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.feed("First sentence. Second sentence. ")
    assert spoken == ["First sentence."]
    seq.utterance_finished(ok=True)
    assert spoken == ["First sentence.", "Second sentence."]

def test_abbreviations_and_decimals_do_not_chop():
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.feed("Dr. Smith measured 3.14 units. Then left. ")
    assert spoken == ["Dr. Smith measured 3.14 units."]

def test_code_fences_are_skipped_entirely():
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.feed("Here you go:\n```python\nx = 1. Yes.\n```\nDone now. ")
    seq.reply_completed()
    joined = " ".join(spoken)
    assert "x = 1" not in joined and "Done now." in joined

def test_markdown_is_stripped_links_keep_text():
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.feed("See **the [docs](https://x.y)** now. ")
    assert spoken == ["See the docs now."]

def test_flush_clears_queue_and_stops_inflight():
    spoken, stops = [], []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: stops.append(1))
    seq.feed("A one. A two. A three. ")
    seq.flush()
    assert stops == [1]
    seq.utterance_finished(ok=False)   # late completion of the stopped utterance
    assert spoken == ["A one."]        # nothing further spoken

def test_reply_completed_flushes_final_partial_and_drains():
    spoken, drained = [], []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.on_drained = lambda: drained.append(1)
    seq.feed("Only a fragment with no terminator")
    seq.reply_completed()
    assert spoken == ["Only a fragment with no terminator"]
    seq.utterance_finished(ok=True)
    assert drained == [1]

def test_zero_speakable_reply_drains_immediately():
    drained = []
    seq = SentenceSequencer(speak=lambda t: None, stop_speech=lambda: None)
    seq.on_drained = lambda: drained.append(1)
    seq.feed("```\ncode only\n```")
    seq.reply_completed()
    assert drained == [1]

def test_failed_utterance_skips_and_continues():
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.feed("A. B. ")
    seq.utterance_finished(ok=False)
    assert spoken == ["A.", "B."]
```
Add: max-length force-split (build a 600-char terminator-free string → split near the cap at whitespace); delta split across chunk boundaries (`"Half a sen"` + `"tence. "` → one utterance); ellipsis "..." not a boundary mid-thought (document chosen rule).

- [ ] **Step 2:** RED. **Step 3:** implement (state: `_buffer`, `_in_fence` flag toggled on ``` lines, `_queue`, `_inflight`, `_completed` flag; normalizer = regex passes for fences first, then links `[t](u)`→`t`, emphasis/heading markers, whitespace collapse; boundary scan with abbrev/decimal lookbehind guard + min-length 4 chars). **Step 4:** green. **Step 5 (mutation):** remove the completion gate (speak immediately) → gating test fails; remove fence skip → fence test fails. **Step 6:** ruff; commit `feat(chat): sentence sequencer for spoken replies`.

---

### Task 3: `HandsFreeController`

**Files:** create `Chat/console_hands_free.py`, `Tests/Chat/test_console_hands_free.py`.

**Interfaces produced:**
```python
ctrl = HandsFreeController(emit=fn, send_delay_seconds=1.5, acoustic_barge_in=False)
ctrl.state  # "idle"|"listening"|"countdown"|"awaiting_reply"|"speaking"
# inputs (all plain methods):
ctrl.enter(capture_live: bool); ctrl.tick(now: float)     # injected clock values
ctrl.on_voice_final(); ctrl.on_speech_resumed(); ctrl.on_voice_command(name)
ctrl.on_capture_ended(had_segments: bool, limit_hit: bool)
ctrl.on_reply_started(); ctrl.on_first_utterance(); ctrl.on_reply_finished()
ctrl.on_sequencer_drained(); ctrl.on_composer_key(); ctrl.on_exit_request()
# intents via emit: RequestStopAndSend, SilenceSpeech, SuppressReplySpeech,
# OpenCapture, CloseCapture, CountdownTick(remaining), ModeChanged(state), ExitLoop
```
All frozen dataclass intents. `tick(now)` drives the countdown from injected time — no wall clock anywhere.

- [ ] **Step 1 (RED)** — full transition matrix as scripted tests; the critical ones verbatim:

```python
def mk(**kw):
    events = []
    c = HandsFreeController(emit=events.append, send_delay_seconds=1.5, **kw)
    return c, events

def test_voice_final_arms_countdown_and_expiry_sends():
    c, ev = mk(); c.enter(capture_live=True)
    c.on_voice_final(); assert c.state == "countdown"
    c.tick(now=0.0); c.tick(now=1.6)
    assert any(isinstance(e, RequestStopAndSend) for e in ev)
    assert c.state == "awaiting_reply"

def test_speech_resumed_cancels_countdown():
    c, ev = mk(); c.enter(capture_live=True)
    c.on_voice_final(); c.tick(now=0.0)
    c.on_speech_resumed()
    assert c.state == "listening"
    c.tick(now=5.0)
    assert not any(isinstance(e, RequestStopAndSend) for e in ev)

def test_resume_vs_expiry_race_arrival_order_wins():
    c, ev = mk(); c.enter(capture_live=True)
    c.on_voice_final(); c.tick(now=0.0); c.tick(now=1.6)   # expiry first
    c.on_speech_resumed()                                   # late resume: rides next turn
    assert c.state == "awaiting_reply"

def test_keypress_in_speaking_barges_in():
    c, ev = mk(); c.enter(capture_live=True)
    c.on_voice_final(); c.tick(0.0); c.tick(1.6)
    c.on_reply_started(); c.on_first_utterance(); assert c.state == "speaking"
    c.on_composer_key()
    assert any(isinstance(e, SilenceSpeech) for e in ev)
    assert any(isinstance(e, OpenCapture) for e in ev)
    assert c.state == "listening"

def test_keypress_in_awaiting_suppresses_speech():
    c, ev = mk(); c.enter(capture_live=True)
    c.on_voice_final(); c.tick(0.0); c.tick(1.6)
    c.on_composer_key()
    assert any(isinstance(e, SuppressReplySpeech) for e in ev)
    assert c.state == "listening"

def test_limit_hit_with_segments_sends_without_segments_reopens_once_then_exits():
    c, ev = mk(); c.enter(capture_live=True)
    c.on_capture_ended(had_segments=True, limit_hit=True)
    assert any(isinstance(e, RequestStopAndSend) for e in ev)
    c2, ev2 = mk(); c2.enter(capture_live=True)
    c2.on_capture_ended(had_segments=False, limit_hit=True)
    assert any(isinstance(e, OpenCapture) for e in ev2)
    c2.on_capture_ended(had_segments=False, limit_hit=True)
    assert any(isinstance(e, ExitLoop) for e in ev2)

def test_reply_drained_reopens_capture():
    c, ev = mk(); c.enter(capture_live=True)
    c.on_voice_final(); c.tick(0.0); c.tick(1.6)
    c.on_reply_started(); c.on_first_utterance()
    c.on_reply_finished(); c.on_sequencer_drained()
    assert c.state == "listening"
    assert any(isinstance(e, OpenCapture) for e in ev)

def test_exit_reachable_from_every_state():
    # One helper per state that builds a controller INTO that state using only
    # the public inputs above (listening: enter; countdown: +on_voice_final;
    # awaiting: +tick to expiry; speaking: +on_reply_started/on_first_utterance),
    # then parametrize over the five builders:
    for build in (build_idle, build_listening, build_countdown,
                  build_awaiting, build_speaking):
        c, ev = build()
        c.on_exit_request()
        assert any(isinstance(e, ExitLoop) for e in ev), c.state
```
Plus: acoustic opt-in (`on_speech_resumed` in `speaking` → SilenceSpeech + listening; WITHOUT opt-in it is ignored in `speaking`); `on_voice_command("stop")` exits; countdown ticks emit `CountdownTick(remaining)` monotonically; the reopen-once flag RESETS after a successful send (only consecutive empty-limit endings exit); **reply failure**: `on_reply_failed()` input → sequencer-suppression intent + `OpenCapture` → `listening` (the loop never traps on a failed generation; the existing error toast is the screen's business).

- [ ] **Step 2:** RED (module absent). **Step 3:** implement the FSM (a `_transition(new_state)` chokepoint emitting `ModeChanged`; countdown = `_armed_at: float | None` compared in `tick`). **Step 4:** green. **Step 5 (mutation):** invert the acoustic gate → its two tests fail; drop the reopen-once reset → the consecutive-limit test fails. **Step 6:** ruff; commit `feat(chat): hands-free loop state machine`.

---

### Task 4: cooldown-free utterance entry + completion signals

**Files:** modify `tts_events.py`; tests in `Tests/TTS_Events/test_utterance_speech_entry.py`.

**Interfaces produced:** a method on the TTS events mixin (exact name the implementer's choice, e.g. `async speak_utterance(text, *, on_finished: Callable[[bool], None])`) that: skips `_enforce_cooldown_limit` (follow the `TTSMessageSpeechRequestEvent` branch shape at :420-436); reuses the SAME generation + playback path as spoken feedback (streaming branch included — sink path reports completion from `pump`'s result; legacy path gains the completion callback where the `_run_blocking_tts_io`-driven `play_audio_file` call returns, failure → `on_finished(False)`); never double-plays; respects one-voice.

- [ ] **Step 1 (RED):** with the fake-response harness from `Tests/TTS_Events/test_spoken_feedback_streaming.py`: (a) two utterances back-to-back both play (no cooldown throttle — RED against a naive TTSRequestEvent-based approach, or assert `_enforce_cooldown_limit` NOT called via spy); (b) completion fires exactly once per utterance on the legacy path (fake player), (c) and on the sink path (fake sink → drained), (d) failure path fires `on_finished(False)` once; (e) `stop` (the existing both-ways routine) interrupts and still fires completion exactly once (ok=False).
- [ ] **Step 2:** RED. **Step 3:** implement. **Step 4:** green + rerun `test_spoken_feedback_streaming.py` (22 expected) — the shared path must not regress. **Step 5 (mutation):** route through the ad-hoc branch → the cooldown spy test fails; drop the completion callback on the legacy path → (b) fails. **Step 6:** ruff; commit `feat(tts): cooldown-free utterance entry with completion signals`.

---

### Task 5: wiring + docs + sweep

**Files:** modify `chat_screen.py`, `console_voice_input.py` (grammar), `Docs/Features/Speech-Services-Guide.md`; create `Tests/UI/test_console_hands_free_wiring.py`.

- [ ] **Step 1:** grammar: `"hands free": "hands-free"` in `COMMAND_PHRASES` (+ classify test). Config readers for the two keys (sibling validation shape).
- [ ] **Step 2:** screen wiring: instantiate controller+sequencer per loop entry, and call
  `sequencer.begin_reply()` at each reply start (the Task-2 review added a per-reply lifecycle:
  suppression latch + fence/buffer state reset live there — a reused sequencer without
  `begin_reply()` never drains reply 2); scheduler = `set_interval(0.1, controller-tick with monotonic now)`; intents → existing machinery (`RequestStopAndSend` → the V2 pending-send seam; `OpenCapture`/`CloseCapture` → dictation start/stop with generation guards; `SilenceSpeech` → the both-ways stop routine + sequencer.flush; `SuppressReplySpeech` → sequencer suppression). Delta tap: subscribe at `append_stream_chunk` (bridge :700/:706 — wrap or observe the store seam; read-only; keyed by assistant message id; completion from `mark_message_complete`). Chip states incl. `CountdownTick` rendering and `thinking…`. Keys: hands-free-active branch in `on_key` BEFORE the Esc binding path (:1627) — Esc/mic exit, other keys barge-in per state; outside the loop `on_key` is byte-identical (pin).
- [ ] **Step 3 (RED-first where behavior is new):** wiring tests — grammar entry starts the loop from idle and from live capture; countdown chip painted (CSS-true harness); keypress in speaking silences (fake sink/player spies) and reopens capture; Esc exits and RESTORES normal Esc semantics after; two-stage send drives the real V2 send flow (stub gateway, message actually dispatched); spoken_feedback=false still speaks replies; acoustic flag opens capture in speaking. Byte-identity pin for `on_key` outside the loop.
- [ ] **Step 4:** docs — hands-free section: entry/exit, the honest ~4 s pause-to-send arithmetic, barge-in modes (headphones note), the ~2 min silent-room exit, both config keys, provider reality note (sentence audio via each provider's existing path; audio.cpp streams).
- [ ] **Step 5:** full named sweep with exact counts: the four new test files + `test_console_voice_input.py` + `test_console_dictation.py` (contract; diff must be empty) + `test_console_dictation_streaming.py` + `test_spoken_feedback_streaming.py` + sink/pump files + `test_console_hands_free.py`/sequencer. **Step 6:** ruff; commits per logical unit; the live gate is the controller's (real hardware: full loop + keyboard barge-in; acoustic = headphones manual).

## Out of scope (spec's list)

AEC; wake-word; TASK-1880; V4; Settings UI; non-hands-free reply speaking.
