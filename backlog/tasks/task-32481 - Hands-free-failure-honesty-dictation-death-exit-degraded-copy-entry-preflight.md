---
id: TASK-32481
title: >-
  Hands-free failure honesty: dictation-death exit, degraded copy, entry
  preflight
status: Done
assignee:
  - '@robert'
created_date: '2026-09-11 23:51'
updated_date: '2026-09-12 01:00
'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three failure-honesty gaps in the Console hands-free loop: (1) a dictation start failure (missing mic extras, no device) leaves the loop running with the Switch ON and no microphone -- the FSM has no capture-failed input and the error path never signals it; (2) the VAD-degraded entry toast recommends spoken 'Console, stop.' which can never fire without webrtcvad, and does not say that nothing is ever auto-sent in degraded mode (the 60s limit exits the loop and inserts text into the draft); (3) hands-free entry performs no readiness preflight, so preventable failures surface only after the loop claims to be live.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Dictation start/mid-capture failure exits the hands-free loop via on_exit_request (dictation.py precedent at `_handle_console_dictation_limit`) — `Tests/UI/test_console_hands_free_wiring.py::test_dictation_start_failure_exits_hands_free_loop` (entry proven by the capture attempt; loop torn down; failure toast still fires; switch repaints OFF through the ExitLoop path's `_teardown_console_hands_free_loop`)
- [x] Degraded-mode entry copy recommends only working exits (mic button, Esc, Ctrl+Shift+H, switch) and states nothing is sent automatically — `test_degraded_entry_copy_does_not_recommend_spoken_stop` pins: no "Console, stop", "auto-send" retained, "draft" named
- [x] Hands-free entry refuses with reason+remedy when `console_voice_input.probe()` is not ok, and warns once (still entering) when no playback path exists at all — `test_entry_refused_when_dictation_probe_fails` (refusal + error toast with remedy + switch repaint) and `test_entry_warns_once_when_no_playback_path_exists` (warning incl. WAV guidance, session still created)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TDD: dictation start/mid-capture failure exits hands-free loop via on_exit_request (dictation.py precedent _handle_console_dictation_limit)
2. TDD: corrected degraded-mode entry copy (no spoken-stop advice; states nothing auto-sends)
3. TDD: entry preflight (probe() refusal with remedy; no-playback warning once)
4. Targeted test runs; task notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- **Dictation-death exit** (`UI/Console_Modules/dictation.py`, `_notify_console_dictation_error`): before resetting dictation state, if a hands-free session exists, `controller.on_exit_request()` — the exact precedent `_handle_console_dictation_limit` set for bounded endings. No new FSM input needed: the ExitLoop path already silences speech (no-op), closes the capture (guarded, already dead), tears the session down, and repaints the Switch OFF. Covers both capture-START failures (missing extras/no device — the Fedora trap) and mid-capture `VoiceFailed` drainings, both of which route through this one method.
- **Degraded copy** (`hands_free.py`, `CONSOLE_HANDS_FREE_DEGRADED_MESSAGE`): dropped the impossible spoken-stop advice (without webrtcvad, segments only finalize at capture stop, so mid-capture spoken commands can never fire — `VoiceVadUnavailable`'s own docstring) and now says what actually happens: nothing is auto-sent; retained speech lands in the draft; end a capture with the mic button or Esc/ctrl+shift+h. The once-per-run `VAD_UNAVAILABLE_MESSAGE` was already accurate and is unchanged.
- **Entry preflight** (`hands_free.py`, `_enter_console_hands_free_pipeline_loop`): mic half refuses BEFORE the session exists when `console_voice_input.probe()` is not ok, with the probe's reason+remedy, and repaints the Switch OFF (the Switch gesture flips the widget visually before the refusal runs — same fix applied to the existing realtime forced-unconfigured refusal, which had the same desync). A probe crash is not a refusal (mirrors `_console_pipeline_hands_free_blocker`). Playback half: when `locally_playable_formats()` is empty, one warning per app run (latch on the app instance, the established two-tier pattern) naming the remedy (install a player / Output format WAV) — entry still proceeds, since mic-only hands-free is legitimate and TASK-32480's per-failure surfacing owns the ongoing signal.
- **Tests.** Four new tests in `Tests/UI/test_console_hands_free_wiring.py`, all watched RED first. One test-shape lesson: with a synchronous failing fake the loop tears down within the first `pilot.pause()`, so "entry happened" is proven by the capture attempt (`fake.start_calls`), not by a surviving session.
- **ADR required: no** — failure-path hardening within existing state machines; no interface or storage changes.
- **Verification.** `Tests/UI/test_console_hands_free_wiring.py` 50 passed; regression sweep `test_console_dictation*.py`, `test_console_realtime_*.py`, `Tests/Chat/test_console_hands_free.py` — 244 passed.
