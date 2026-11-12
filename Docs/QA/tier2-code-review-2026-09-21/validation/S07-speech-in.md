# S07 — `Audio/` + `STT/` validation

Zero commits touched `tldw_chatbook/Audio/` or `tldw_chatbook/STT/` between the review commit
`3722a85748` and `HEAD`. All 12 findings checked directly against current code.

## 1. P1 [D1] — VAD gate drops the trailing partial frame every call, no carry
- Verdict: CONFIRMED
- Site now: `Audio/recording_service.py:518-543` (`_process_audio_chunk`,
  `for i in range(0, len(chunk) - frame_size + 1, frame_size)`), unchanged
- Proof: read confirms no `_gate_carry`-style residual buffer exists in this file — any bytes
  past the last full 640-byte frame are silently dropped every call. Comparison read of
  `Audio/meeting_capture.py:517-548` `_gate()` confirms the correct sibling: `data = bytes(self.
  _gate_carry) + mixed` at entry, `self._gate_carry = bytearray(data[complete_end:])` at exit,
  with the explicit comment "no audio is discarded ... just because a chunk crosses a 640-byte
  boundary." `grep -n "640" Tests/Audio/test_recording_vad_preroll.py
  test_dictation_vad_finalization.py test_audio_integration.py` confirms all three pinning tests
  use exact 640-byte multiples, so none exercises the drop.

## 2. P1 [D1/D4] — `voiceprint.py`'s atomic write re-rolls the helper without `fsync`
- Verdict: CONFIRMED
- Site now: `Audio/voiceprint.py:235-256` (`_atomic_write`: `mkstemp` → `fdopen.write` →
  `os.replace`, no `flush()`/`fsync()`), used at `:365` (key file), `:532` (`save()`), `:656`
  (`export()`)
- Proof: `grep -n "fsync" tldw_chatbook/Audio/voiceprint.py tldw_chatbook/Audio/
  diarizer_engine_onnx.py` → zero hits in either file. `Utils/atomic_file_ops.py:98-103` does
  `f.flush(); os.fsync(f.fileno())` before `os.replace`. `backlog/tasks/task-32808.5` status is
  `Done`; its Implementation Notes enumerate a census that does not mention `voiceprint.py` or
  `Scheduling/scheduler_heartbeat.py` — confirms the "missed by the census" framing.

## 3. P2 [D1] — `swiftc` invoked with no `timeout=` inside the exclusive `meetings-prepare` worker
- Verdict: CONFIRMED
- Site now: `Audio/system_audio_tap.py:234-239` (`run([swiftc, ...], capture_output=True,
  text=True)`, no `timeout=`) vs `:305` (`run(["pactl", ...], timeout=5)`)
- Proof: direct read of both call sites confirms the asymmetry. `grep -n "meetings-prepare"
  tldw_chatbook/UI/Screens/meetings_screen.py` → `:437` `@work(exclusive=True,
  group="meetings-prepare", thread=True, exit_on_error=False)` wraps the call path into `probe()`.

## 4. P2 [D1/D4] — two ffmpeg conversions run with no `timeout=`
- Verdict: CONFIRMED
- Site now: `STT/parakeet_onnx.py:136-153` (`_prepared_wav`, `subprocess.run([executable, ...],
  check=True, capture_output=True)`) and `STT/transcribe_cpp.py:284-301` (same shape, no
  `timeout=`)
- Proof: direct read of both; neither passes `timeout=`.

## 5. P2 [D3] — `dictation.buffer_duration_ms` is the one of five config reads with no validation
- Verdict: CONFIRMED
- Site now: `Audio/dictation_service_lazy.py:323-326` (`self.buffer_duration_ms =
  get_cli_setting("dictation.buffer_duration_ms", self.BUFFER_DURATION_MS)`, bare) vs siblings
  `_resolve_stop_join_timeout`/`_resolve_silence_threshold`/`_resolve_vad_aggressiveness`/
  `_resolve_vad_preroll_ms` (`:327-330`, each a validating classmethod)
- Proof: direct read confirms the asymmetry; `:1437` `set_buffer_duration` DOES clamp
  (`max(100, min(2000, …))`) — the UI path is guarded, the config path is not, same field.

## 6. P2 [D3] — `Audio/dictation_metrics.py` (380 lines) is now fully orphaned
- Verdict: CONFIRMED
- Site now: `Audio/dictation_metrics.py` (380 lines)
- Proof: `grep -rn "dictation_metrics|DictationPerformanceMonitor|get_performance_monitor"
  tldw_chatbook/ Tests/` → zero hits outside the file itself. `Widgets/
  dictation_performance_widget.py` no longer exists (`ls` → No such file). `git log --oneline -3
  -- tldw_chatbook/Widgets/dictation_performance_widget.py` → deleted by `dda82ba6dd
  fix(polish): Widget bundles .5/.6/.10 — delete dead widget ... (task-32810)`.

## 7. P2 [D3] — `Audio/console_dictation.py` (252 lines) is dead code kept alive only by its own test
- Verdict: CONFIRMED
- Site now: `Audio/console_dictation.py` (252 lines)
- Proof: `grep -rn "ConsoleDictationSession|ConsoleDictationError"` → all hits are in
  `Tests/Audio/test_console_dictation.py`. `CONSOLE_DICTATION_MAX_BYTES` is independently
  redefined (not imported) at `UI/Console_Modules/dictation.py:169`, whose `:567` comment names
  `Audio/console_dictation.py` as "the one-shot backend this replaced."

## 8. P3 [D3] — `Audio/dictation_service.py` (652 lines) reachable only through a module being deleted
- Verdict: CONFIRMED
- Site now: `Audio/dictation_service.py` (652 lines), exported via `Audio/__init__.py`'s
  `_LAZY_EXPORTS["LiveDictationService"] = ".dictation_service"`
- Proof: the only production importer of the lazy export is
  `Widgets/voice_input_widget.py:19` (`from ..Audio import LiveDictationService, DictationState`).
  `grep -rn "voice_input_widget|VoiceInputWidget" --include="*.py" tldw_chatbook/` → zero hits
  outside `voice_input_widget.py` itself (all other hits are in `Tests/`).
  `backlog/tasks/task-32807.1` status `In Progress`; its notes explicitly list
  `voice_input_widget` among the "deferred" unreferenced widgets still pending deletion.

## 9. P3 [D1] — downloaded ONNX diarizer models `os.replace`d with no fsync, but hash-verified
- Verdict: CONFIRMED
- Site now: `Audio/diarizer_engine_onnx.py:561-567` (`os.replace(final_tmp, path)` gated on
  `_sha256_of(final_tmp) == asset.sha256` immediately above), `load()`'s `verify_hashes: bool =
  True` default at `:822`
- Proof: direct read confirms both the size/hash gate before replace and the re-verify-on-load
  default. Correctly filed P3 (torn write surfaces loudly as a hash mismatch, not silent data loss).

## 10. P3 [D3] — the two ruff fatals in `meeting_owner.py` are annotation-only, nothing introspects them
- Verdict: CONFIRMED
- Site now: `Audio/meeting_owner.py:1184` — `from .meeting_capture import MeetingCapture` is a
  real runtime late-import (numpy-avoidance), not inside `if TYPE_CHECKING:`
- Proof: `grep -rn "get_type_hints|TypeAdapter|validate_call" tldw_chatbook/` → the only
  `get_type_hints` call in production is `MCP/gateway_runtime.py:515` on MCP tool handlers; no
  hit touches `meeting_owner.py` or `MeetingCapture`. Matches the review's "Not promoted" call.

## 11. P3 [D3] — dead per-sample `struct` fallback inside the PortAudio callback
- Verdict: CONFIRMED
- Site now: `Audio/recording_service.py:471-487` (`else` branch of `if NUMPY_AVAILABLE and np is
  not None` inside `audio_callback`)
- Proof: `__init__` (`:184-195`) raises `AudioRecordingError` unconditionally when `not
  NUMPY_AVAILABLE`, so no instance can exist without numpy — the else branch is unreachable dead
  code, confirmed by direct read of both sites.

## 12. P3 [D1] — `id()`-keyed delivery map in `voice_process_lifetime` lacks its sibling's pinning
- Verdict: CONFIRMED (same "inferred" confidence as filed)
- Site now: `Audio/voice_process_lifetime.py:360/391` (`self._deliveries[id(record)] =
  receiver.receive(record, mailbox=self.input)`), popped at `:445`
  (`self._deliveries.pop(id(record), None)`)
- Proof: traced `ReceiveStream.receive` (`voice_process_protocol.py:978-993`) → constructs
  `Delivery(record, self, self._received, size, mailbox)`; `Delivery.record` (`:911`) is a real
  field, so the stored `Delivery` DOES hold a strong reference to `record` at storage time — a
  refinement the review's "no strong reference pinning the key" phrasing doesn't state. But
  `_consume()` (`voice_process_protocol.py:995-1002`) sets `delivery.record = None` on use,
  *before* `release()` pops the dict entry — so the risk window is real but narrower than
  implied: it exists only in the gap between a delivery being consumed and `release()` being
  called for it, not for the delivery's whole lifetime. Net: the underlying defect class stands;
  confidence matches the review's own "inferred" label and its Left-UNVERIFIED entry for this item.
- Note: minor refinement worth carrying forward — the exploitable window is "post-consume,
  pre-release," not "the whole time the id is a dict key," which the review's phrasing implies.

TOTALS: confirmed=12 fixed=0 wrong=0 demoted=0 promoted=0
