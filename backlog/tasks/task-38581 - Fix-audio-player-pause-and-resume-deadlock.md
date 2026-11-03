---
id: TASK-38581
title: Fix audio player pause and resume deadlock
status: Done
created_date: 2026-11-02 14:26
labels:
- bug
- audio
priority: high
updated_date: 2026-11-02 17:10
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
One Pause click permanently wedges all TTS audio on players without native pause such as afplay on macOS. pause takes a non-reentrant lock then calls get_position and stop which re-acquire it. See bug-sweep-2026-11-02-dev.md section 1.2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 pause and resume return promptly on players without native pause such as afplay
- [x] #2 Reentrant lock or calls moved outside the lock
- [x] #3 Regression test pauses a real player without hanging
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Approach: replaced `SimpleAudioPlayer._lock`'s non-reentrant
  `threading.Lock()` with `threading.RLock()` (smallest safe diff, per the
  batch ambiguity resolution), with a comment documenting the reentrancy
  requirement: pause() -> get_position() + stop(), resume() -> play() ->
  stop(), on players without native pause (afplay/ffplay/paplay/pw-play/
  aplay). A file scan confirmed pause()/resume() are the only
  nested-acquisition sites; every other `with self._lock` body only
  touches state directly.
- Regression tests (`Tests/TTS/test_audio_player_pause_resume_deadlock.py`)
  construct the real player forced into the afplay shape and call
  pause()/resume() on a helper thread with a 3s join timeout, asserting
  `not thread.is_alive()` (fails fast pre-fix instead of hanging: 3 tests
  failed in 16.1s pre-fix; all pass post-fix). No real audio: `Popen` is
  faked with a live-process mock; test cleanup `stop()` is
  deadlock-guarded too (pre-fix it wedged the suite on the held lock).
- Files: `tldw_chatbook/TTS/audio_player.py`,
  `Tests/TTS/test_audio_player_pause_resume_deadlock.py`.
- Commit: 370984d1a3 on `fix/audit-burn-down-38580-38613`.
- ADR required: no — routine bug fix, no architectural decision.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
