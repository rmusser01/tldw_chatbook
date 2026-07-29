# Console Voice Control V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Spoken commands while the microphone is open ("console, send") and an opt-in spoken-feedback mode, so the Console can be operated without hands or eyes on the screen.

**Architecture:** Task 0 makes per-segment finalization real in `LazyLiveDictationService` (VAD-gated silence detection; silent chunks never transcribed). The command grammar lives in the headless `ConsoleVoiceInputController` — whole-segment prefix matching, fail-open to text — emitting a new `VoiceCommand` event. The session adapter in `ChatScreen` consumes inline commands (line breaks) and counts command-consumed segments; the screen routes capture-ending commands through existing paths; spoken feedback rides the task-559 TTS pipeline under a hard microphone/speaker mutual-exclusion rule.

**Tech Stack:** Python ≥3.11, Textual 8.x, webrtcvad (already in the `speech_recording` extra), existing `TTSRequestEvent`/`TTSPlaybackEvent` pipeline, pytest.

**Source spec:** `Docs/superpowers/specs/2026-07-29-console-voice-control-design.md` — the spec's decisions are settled; do not relitigate them.

## Global Constraints

- **Worktree:** all work in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/voice-control-v2`; `git branch --show-current` must print `feat/console-voice-control-v2`.
- **Never** `git stash`/`git stash pop` (stack shared across ~100 worktrees; a bare pop destroyed another session's WIP once).
- **Never** run whole `Tests/UI`/`Tests/Chat` directories (~2 h each), nor `Tests/Audio/test_recording_service.py`/`test_audio_integration.py` (hang on real hardware). No background commands, no Monitors, no model downloads, no real transcription. Foreground pytest only: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest ... -q -p no:randomly`.
- **`Tests/UI/test_console_dictation.py` (4 contract tests) must pass unmodified and stay out of every diff.** Needing to edit it means the change is wrong — stop and report.
- **V1 invariants:** `probe()` stays `find_spec`-only and the subprocess guard `test_screen_import_does_not_load_transcription_stack` must keep passing; controller events leave via `post_message`, never `call_from_thread`; `VoiceFailed` precedes `VoiceStateChanged(idle)` (two index-comparison tests pin it); chip text renders via `textual.Content`; chip-text tests assert on `render_line(0).text`, never `renderable`.
- **Textual message-name trap:** a `Message` subclass whose snake_case name matches an existing `_on_<name>` method is auto-dispatched into it (`handler_name` convention) — this caused an unbounded loop once. New message classes must be name-checked against existing methods.
- **Mutation-check every behavioral change:** revert it, confirm its covering test fails, restore byte-identical.
- **Config validation:** every new config read mirrors the `stop_join_timeout_seconds` shape — `math.isfinite(...) or value <= 0` → warn once, fall back to the class-constant default (`dictation_service_lazy.py:247`).
- If anything under `tldw_chatbook/css/` changes, regenerate the tracked bundle with `build_css.py` and commit it; never hand-edit `tldw_cli_modular.tcss`. (No task below should need CSS.)
- Python ≥3.11; type hints; Google-style docstrings with `Args:`/`Returns:` on public callables (a PR bot enforces this).
- Backlog tasks: mark In Progress with a plan before implementing, Done with notes after; assign any NEW task ids only after an `os.listdir` sweep across ALL worktrees (`backlog task create` scans only the local one and WILL collide).

---

### Task 0: VAD-gated segment finalization

Makes mid-capture segment finals real. Today `VoiceFinal` fires only at stop because `_audio_callback` refreshes `last_speech_time` on every delivered chunk (`dictation_service_lazy.py:539`) and the recorder delivers every chunk (VAD is stored, never applied).

**Files:**
- Modify: `tldw_chatbook/Audio/dictation_service_lazy.py` (`_audio_callback` at :498, `_processing_loop` silence branch at :584-590, class constants)
- Test: `Tests/Audio/test_dictation_vad_finalization.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces: `LazyLiveDictationService._chunk_has_speech(chunk: bytes) -> bool`; class constant `SILENCE_THRESHOLD_SECONDS = 2.0`; config key `dictation.silence_threshold_seconds`; behavior — `on_final_transcript` fires per ≥threshold pause mid-capture; fully-silent chunks are neither transcribed nor speech-time-refreshing.

**Details the implementer must know:**

- webrtcvad: `webrtcvad.Vad(aggressiveness)` with `.is_speech(frame_bytes, sample_rate)`; frames must be exactly 10/20/30 ms of 16-bit mono. At 16 kHz a 30 ms frame is 480 samples = **960 bytes**. Delivered chunks are ~500 ms (not a multiple of 960 is possible) — iterate complete 960-byte frames, ignore the remainder. **A chunk is speech if ANY frame is speech** — only fully-silent chunks are excluded, so soft speech is never dropped.
- Import webrtcvad lazily/defensively (module may be absent): build the `Vad` once per capture in `start_dictation`, `None` when unavailable. With `vad is None`, `_chunk_has_speech` returns `True` — exact today's behavior (finals at stop only), never a crash. Aggressiveness 2, matching the recorder's stored-but-unused setting.
- In `_audio_callback`: gate BOTH the `last_speech_time` refresh AND the enqueue-for-transcription on `_chunk_has_speech(audio_chunk)`. Silent chunks still count toward `captured_bytes` (the capture-outcome logic must keep seeing that audio arrived) but are not queued to `processing_queue`.
- **The silence-timeout check stays at the loop level** (`_processing_loop`, currently :584-590) — it runs every ~0.1 s iteration independent of chunk arrival, which is exactly why skipping silent chunks cannot starve finalization. Do not move it. Replace its literal `2.0` with the validated instance value.
- Threshold config: `dictation.silence_threshold_seconds`, default `SILENCE_THRESHOLD_SECONDS = 2.0`, validated with the exact `isfinite`/`<= 0` shape at :247.
- The three other service consumers and the four contract tests must be behaviorally unaffected when VAD is unavailable, and only *gain* per-pause finals when it is.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Audio/test_dictation_vad_finalization.py` (module-level `pytestmark = pytest.mark.unit`; construct the service via `__new__` with real `threading.Event`/locks/queue — copy the fixture shape from `Tests/Audio/test_dictation_tail_flush.py`, which this suite sits beside):

```python
def test_silent_chunks_do_not_refresh_last_speech_time():
    """A VAD-negative chunk must not push the finalize deadline out."""
    service = _service(vad=_FakeVad(speech=False))
    service.last_speech_time = 111.0
    service._audio_callback(_chunk())
    assert service.last_speech_time == 111.0


def test_speech_chunks_refresh_last_speech_time():
    service = _service(vad=_FakeVad(speech=True))
    service.last_speech_time = 0
    service._audio_callback(_chunk())
    assert service.last_speech_time > 0


def test_silent_chunks_are_not_queued_for_transcription():
    """Whisper hallucinates on silence; silent audio never reaches the provider."""
    service = _service(vad=_FakeVad(speech=False))
    service._audio_callback(_chunk())
    assert service.processing_queue.empty()


def test_silent_chunks_still_count_captured_bytes():
    """The capture-outcome logic must still see that audio arrived."""
    service = _service(vad=_FakeVad(speech=False))
    before = service.captured_bytes
    service._audio_callback(_chunk())
    assert service.captured_bytes == before + len(_chunk())


def test_chunk_with_any_speech_frame_is_speech():
    """Only fully-silent chunks are excluded — soft speech is never dropped."""
    service = _service(vad=_FakeVad(speech_frames=[False] * 15 + [True]))
    assert service._chunk_has_speech(_chunk()) is True


def test_no_vad_degrades_to_always_speech():
    service = _service(vad=None)
    assert service._chunk_has_speech(_chunk()) is True


def test_pause_finalizes_a_segment_mid_capture():
    """The whole point: a >threshold pause fires on_final_transcript before stop."""
    # Drive _processing_loop on a real thread with a monkeypatched short
    # threshold (0.2s); feed a speech chunk, wait past the threshold, assert
    # the final fired and current_transcript reset — without stop_dictation().
```

(`_FakeVad(speech=...)` returns a fixed `is_speech`; `speech_frames=[...]` pops per call. `_chunk()` returns 16000 bytes = 500 ms of int16.)

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest Tests/Audio/test_dictation_vad_finalization.py -v` → FAIL (`_chunk_has_speech` missing; silent chunk currently refreshes and queues).

- [ ] **Step 3: Implement** per the details above. `_chunk_has_speech`:

```python
    def _chunk_has_speech(self, audio_chunk: bytes) -> bool:
        """Return True when any 30 ms frame of the chunk contains speech.

        Args:
            audio_chunk: 16-bit mono PCM at the recorder's sample rate.

        Returns:
            True if the VAD marks any complete frame as speech, or when no
            VAD is available (degrading to today's always-speech behavior).
        """
        vad = self._vad
        if vad is None:
            return True
        frame_bytes = 960  # 30 ms of 16-bit mono at 16 kHz
        for start in range(0, len(audio_chunk) - frame_bytes + 1, frame_bytes):
            try:
                if vad.is_speech(audio_chunk[start : start + frame_bytes], 16000):
                    return True
            except Exception:  # noqa: BLE001 - a VAD failure must never kill capture
                return True
        return False
```

- [ ] **Step 4: Run to verify pass**, plus the neighbours: `.venv/bin/python -m pytest Tests/Audio/test_dictation_vad_finalization.py Tests/Audio/test_dictation_tail_flush.py Tests/Audio/test_dictation_lazy_transcription.py Tests/Audio/test_dictation_stop_join.py Tests/Audio/test_dictation_capture_release.py Tests/UI/test_console_dictation.py -q -p no:randomly` — contract tests unmodified.

- [ ] **Step 5: Mutation-check** (drop the VAD gate → silent-chunk tests fail; move the timeout check into `_process_audio_buffer` → the pause test fails) **and commit** `feat(audio): make per-segment finalization real with VAD-gated silence`.

---

### Task 1: Command grammar in the controller

**Files:**
- Modify: `tldw_chatbook/Chat/console_voice_input.py`
- Test: `Tests/Chat/test_console_voice_input.py` (append)

**Interfaces:**
- Consumes: the controller's existing `_emit` seam — `on_final_transcript=lambda text: self._emit(VoiceFinal(text))` inside `_run_begin`.
- Produces: frozen dataclass `VoiceCommand(name: str)`; `COMMAND_PHRASES: dict[str, str]` mapping normalized phrase → command name (`"new paragraph"→"new-paragraph"`, `"new line"→"new-line"`, `"stop"→"stop"`, `"send"→"send"`, `"discard"→"discard"`, `"read that back"→"read-that-back"`, `"new session"→"new-session"`); `normalize_spoken(text: str) -> str`; `command_prefix() -> str` (config-read, blank→default `"console"`); `classify_segment(text: str) -> VoiceCommand | VoiceFinal`.

**Grammar rules (from the spec, settled):** normalization = lowercase, remove ALL punctuation (`string.punctuation`), collapse whitespace. Whole-segment match only: normalized segment == `f"{prefix} {phrase}"` exactly. Anything else — including prefixed typos — is `VoiceFinal` text (fail open). `classify_segment` replaces the lambda: `on_final_transcript=lambda text: self._emit(classify_segment(text))`.

- [ ] **Step 1: Failing tests** (append; hermetic via `_stub_settings`):

```python
def test_console_comma_send_period_matches(monkeypatch):
    """Recognizers emit 'Console, send.' — punctuation must not block the match."""
    _stub_settings(monkeypatch, {})
    result = cvi.classify_segment("Console, send.")
    assert isinstance(result, cvi.VoiceCommand) and result.name == "send"


def test_trailing_words_fail_open_to_text(monkeypatch):
    _stub_settings(monkeypatch, {})
    result = cvi.classify_segment("Console send button is broken")
    assert isinstance(result, cvi.VoiceFinal)
    assert result.text == "Console send button is broken"


def test_prefixed_typo_fails_open_to_text(monkeypatch):
    _stub_settings(monkeypatch, {})
    assert isinstance(cvi.classify_segment("console sned"), cvi.VoiceFinal)


def test_every_command_phrase_matches(monkeypatch):
    _stub_settings(monkeypatch, {})
    for phrase, name in cvi.COMMAND_PHRASES.items():
        result = cvi.classify_segment(f"Console, {phrase}!")
        assert isinstance(result, cvi.VoiceCommand) and result.name == name


def test_custom_prefix(monkeypatch):
    _stub_settings(monkeypatch, {"dictation.command_prefix": "hey app"})
    assert isinstance(cvi.classify_segment("Hey app, stop."), cvi.VoiceCommand)
    assert isinstance(cvi.classify_segment("Console, stop."), cvi.VoiceFinal)


def test_blank_prefix_falls_back_to_default(monkeypatch):
    _stub_settings(monkeypatch, {"dictation.command_prefix": "   "})
    assert isinstance(cvi.classify_segment("Console, stop."), cvi.VoiceCommand)


def test_plain_segments_are_untouched(monkeypatch):
    _stub_settings(monkeypatch, {})
    result = cvi.classify_segment("hello world")
    assert isinstance(result, cvi.VoiceFinal) and result.text == "hello world"


def test_controller_emits_voice_command_for_command_segment(monkeypatch):
    """End-to-end through the controller's on_final seam."""
    controller, events, service = _controller(monkeypatch)
    controller.start()
    service.emit_final("Console, stop.")
    commands = [e for e in events if isinstance(e, cvi.VoiceCommand)]
    finals = [e for e in events if isinstance(e, cvi.VoiceFinal)]
    assert [c.name for c in commands] == ["stop"] and finals == []
```

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** (`VoiceCommand` frozen dataclass beside the other events; `normalize_spoken` removing every character whose `unicodedata.category` starts with `P` (Unicode punctuation — covers ASCII, curly quotes U+2019, and ellipsis U+2026, all of which Whisper emits) plus ASCII `string.punctuation` as a belt; prefix read via `get_cli_setting("dictation", "command_prefix", None)`, `str.strip()`, blank→`"console"`; wire `classify_segment` into `_run_begin`'s lambda). Docstrings with `Args:`/`Returns:` on all new public callables.
- [ ] **Step 4: Run** `Tests/Chat/test_console_voice_input.py` + the subprocess import guard (same file) — all green.
- [ ] **Step 5: Mutation-check** (preserve internal punctuation → `test_console_comma_send_period_matches` fails; drop whole-segment rule → trailing-words test fails) **and commit** `feat(console): whole-segment spoken-command grammar in the voice controller`.

---

### Task 2: Adapter — inline breaks, break-aware join, command-consumed outcome

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`ConsoleStreamingDictationSession`: `_handle_event` ~:713, `_segments` join at :824, `stop_and_transcribe` empty-raise ~:706-715)
- Test: `Tests/UI/test_console_dictation_streaming.py` (append)

**Interfaces:**
- Consumes: `VoiceCommand` from Task 1.
- Produces: inline commands (`new-paragraph`→`"\n\n"`, `new-line`→`"\n"`) appended to `_segments` as break entries; `_join_segments(segments: list[str]) -> str` (break-aware); `commands_consumed: int` on the session; capture-ending `VoiceCommand`s re-emitted to the screen's event channel untouched (the screen routes them in Task 3 — this task only ensures they do NOT land in `_segments` and DO increment `commands_consumed`).

**Break-aware join** (the `" ".join` at :824 would produce `"para. \n\n para"`):

```python
def _join_segments(segments: list[str]) -> str:
    """Join transcript segments with single spaces, without padding breaks."""
    out = ""
    for segment in segments:
        if segment in ("\n", "\n\n"):
            out = out.rstrip(" ") + segment
        elif out and not out.endswith((" ", "\n")):
            out += " " + segment
        else:
            out += segment
    return out
```

**Capture-outcome correction:** in `stop_and_transcribe`, when the transcript **strips to empty** but `self.commands_consumed > 0` (inline and capture-ending commands both count), return `""` instead of raising — the screen skips insertion for an empty return (add that skip if absent: empty transcript + commands consumed → no insert, no error, no whitespace). The V1 silent-capture errors still fire for genuinely empty captures (`commands_consumed == 0`).

- [ ] **Step 1: Failing tests** — break entries join unpadded (`["one.", "\n\n", "two"]` → `"one.\n\ntwo"`); inline `VoiceCommand("new-paragraph")` appends a break AND increments `commands_consumed` (a capture of only inline commands joins to whitespace and must not raise); capture-ending `VoiceCommand("stop")` increments `commands_consumed` and adds nothing to `_segments`; command-only capture returns `""` and raises nothing; genuinely empty capture still raises the V1 message.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the streaming file + the four contract tests — green, contract unmodified.
- [ ] **Step 5: Mutation-check** (naive join → padding test fails; drop `commands_consumed` → false-error test fails) **and commit** `feat(console): inline voice commands and command-aware capture outcome`.

---

### Task 3: Screen routing for capture-ending commands

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_handle_console_dictation_event`, new wrapper message, dispatch)
- Test: `Tests/UI/test_console_dictation_streaming.py` (append)

**Interfaces:**
- Consumes: `VoiceCommand` events surfacing from the adapter (Task 2); existing paths — `_request_console_dictation_stop()`, the cancel flow, `#console-send-message`, the new-tab action, `store.messages_for_session()` / `ConsoleChatMessage(role, content, status)` / `self._console_run_active()` (:9870), task-559's `TTSRequestEvent` handoff and `_console_speaking_message_id`.
- Produces: wrapper message `ConsoleVoiceCommandSignal(name: str)` (**name-checked**: no `_on_console_voice_command_signal` exists — grep before committing, per the Global Constraints trap); `_console_pending_voice_send: bool`; dispatch:
  - `stop` → `_request_console_dictation_stop()`
  - `discard` → the existing cancel flow
  - `send` → set `_console_pending_voice_send`, then `_request_console_dictation_stop()`; in `_stop_console_dictation`, after successful insertion, if the flag is set: clear it and `self.query_one("#console-send-message", Button).press()`. **`_notify_console_dictation_error` clears the flag without sending** — a failed dictation must never ship the message.
  - `new-session` → `_request_console_dictation_stop()`, then after successful completion invoke the existing new-tab action.
  - `read-that-back` → `_request_console_dictation_stop()`; after completion: if `self._console_run_active()` → ack "Still responding." (toast; spoken in Task 4); else find the last `ConsoleChatMessage` with `role == "assistant"` and `status == "complete"` in `store.messages_for_session(store.active_session_id)`; none → ack "Nothing to read yet."; found → post `TTSRequestEvent(text=message.content, message_id=message.id)` via `self.app_instance.post_message`, set `_console_speaking_message_id = message.id`, resync — mirroring the task-559 handler (:15451).

- [ ] **Step 1: Failing tests** — each command routed (fake adapter emitting `VoiceCommand`); send-after-insertion ordering pinned with the deferred-flag interleaving technique already used in this file; send flag cleared on stop failure with nothing sent; read-back with a streaming run acks instead of speaking; read-back with no assistant message acks; read-back posts `TTSRequestEvent` with the completed message's content.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** streaming + contract files — green.
- [ ] **Step 5: Mutation-check** (press send in the same tick as stop → ordering test fails; drop the error-path flag clear → failure test fails) **and commit** `feat(console): route spoken capture-ending commands through existing paths`.

---

### Task 4: Opt-in spoken feedback with microphone/speaker mutual exclusion

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_dictation_streaming.py` (append)

**Interfaces:**
- Consumes: `TTSRequestEvent(text)` (arbitrary text, no message id), `TTSPlaybackEvent(action="stop")`, the capture lifecycle.
- Produces: `_speak_status(text: str) -> None` — posts `TTSRequestEvent(text)` iff `dictation.spoken_feedback` is truthy (coerced with the repo's `coerce_bool_setting`) AND no capture is active; called for: capture ended, command acks ("Sent.", "Discarded.", "New session.", "Still responding.", "Nothing to read yet."), and dictation errors (same reason strings as the toasts). **"Capture started" is never spoken** (spec: it cannot be, with the mic open). `_request_console_dictation_start` posts `TTSPlaybackEvent(action="stop")` **before** opening capture — the single-slot player does not stop on mic-open by itself, and an in-flight ack would transcribe itself into the new draft.

- [ ] **Step 1: Failing tests** — toggle off → zero `TTSRequestEvent` for status; toggle on → events for the enumerated moments; `_speak_status` while a capture is active posts nothing (mutual exclusion); capture start posts `TTSPlaybackEvent("stop")` before the recorder opens (ordering asserted); read-back speech itself is unaffected by the toggle.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** streaming + contract files — green.
- [ ] **Step 5: Mutation-check** (drop the playback-stop → ordering test fails; drop the mic-open guard → exclusion test fails) **and commit** `feat(console): opt-in spoken feedback with mic/speaker mutual exclusion`.

---

### Task 5: Config docs, verification, and follow-ups

**Files:**
- Modify: `Docs/Features/Speech-Services-Guide.md` (the three new keys, the pause-command-pause choreography and its ~2-threshold latency, the command table); the Console F1 help if it documents dictation.
- Test: full targeted sweep.
- Create: one backlog follow-up task.

- [ ] **Step 1: Docs** — the spec's config table verbatim (`dictation.command_prefix`, `dictation.spoken_feedback`, `dictation.silence_threshold_seconds`); the choreography stated as the spec words it ("pause briefly before and after a command"), including that a command fires ~threshold after the utterance.
- [ ] **Step 2: Targeted sweep** — `.venv/bin/python -m pytest Tests/Chat/test_console_voice_input.py Tests/UI/test_console_dictation.py Tests/UI/test_console_dictation_streaming.py Tests/UI/test_console_voice_chip.py Tests/UI/test_console_dictation_firstrun.py Tests/Audio/test_dictation_vad_finalization.py Tests/Audio/test_dictation_tail_flush.py Tests/Audio/test_dictation_lazy_transcription.py Tests/Audio/test_dictation_capture_release.py Tests/Audio/test_dictation_stop_join.py Tests/Audio/test_dictation_privacy_allowlist.py Tests/Audio/test_console_dictation.py -q -p no:randomly` — everything green, contract file untouched (`git diff --name-only origin/dev.. | grep -c test_console_dictation.py` → 0).
- [ ] **Step 3: File the follow-up** — retire `Audio/voice_commands.py` (dormant, stale APP_NAVIGATION actions, no callers once V2's grammar ships). ID via the all-worktrees sweep; task file per repo format.
- [ ] **Step 4: Write the live-verification checklist into the task report** (a human runs it before merge; the spec's list verbatim): command executing mid-capture ("console, stop") with acceptable latency at the default threshold; prose beginning with "console" landing in the draft; staccato "Console. Send." false-fire check; "console, send" shipping the full utterance including the last segment; command-only capture producing no error; spoken feedback audible with the toggle on, absent off; capture start stopping an in-flight read-back with no self-transcription.
- [ ] **Step 5: Commit** `docs(console): voice-control configuration and verification for V2`.

---

## Self-review notes

- **Spec coverage:** Task 0 ↔ spec "Task 0" (VAD gating, silent-chunk exclusion, loop-level pin, threshold config); Task 1 ↔ Grammar (normalization tradeoff, whole-segment, fail-open, prefix validation); Task 2 ↔ command table inline rows + join note + capture-outcome correction; Task 3 ↔ Screen routing incl. pending-send failure rule and completed-message read-back; Task 4 ↔ Spoken feedback incl. the no-capture-started rule and explicit playback stop; Task 5 ↔ Configuration, Testing's live list, and the voice_commands.py follow-up. Latency documentation lands in Task 5. No spec section is uncovered.
- **Type consistency:** `VoiceCommand(name: str)` (Task 1) consumed by Tasks 2/3; `classify_segment` name used consistently; `_join_segments` defined where used; command names kebab-case in `COMMAND_PHRASES` and the Task 3 dispatch match one-to-one.
- **Known refinement:** `classify_segment` is module-level (like `probe`/`resolve`) rather than a controller method, so grammar tests need no controller instance; the controller lambda calls it. This mirrors how V1 kept pure logic module-level.
