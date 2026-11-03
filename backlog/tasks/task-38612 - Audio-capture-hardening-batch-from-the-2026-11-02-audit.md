---
id: TASK-38612
title: Audio capture hardening batch from the 2026-11-02 audit
status: Done
created_date: 2026-11-02 14:30
labels:
- hardening
- audio
priority: low
updated_date: 2026-11-02 17:10
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two capture-path items: the VAD loop drops sub-frame tail bytes of every chunk for non-default buffer durations causing continuous small audio loss, and stop_recording can raise out of terminate when the join times out. See bug-sweep-2026-11-02-dev.md Audio findings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 VAD path carries partial frames across chunks instead of dropping them
- [x] #2 stop_recording wraps terminate so it cannot raise over callers
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- (a) Extracted frame assembly into a pure module-level helper
  `_assemble_vad_frames(pending, chunk, frame_size)` (per the batch
  ambiguity resolution) that prepends the carried partial frame, slices
  whole frames, and returns the new tail -- the same windowing carry
  `speech_presentation.py`'s `PcmEnvelope` uses. `_process_audio_chunk`
  now stores the tail in `self._vad_pending_frame` (class-level b""
  fallback for `__new__`-built instances, matching `_preroll_frames`);
  `start_recording` resets it so a new capture never inherits the previous
  capture's partial frame. Pre-roll deque semantics unchanged.
- (b) `stop_recording()` wraps `pyaudio_instance.terminate()` in
  try/except-with-logging, mirroring `__del__`'s guard; success path
  unchanged (terminate runs, handle cleared), failure keeps the handle so
  `__del__` can retry. Previously the raise escaped AFTER the capture was
  saved but BEFORE `audio_data` was returned -- callers lost the
  recording to a cleanup error.
- Tests (`Tests/Audio/test_recording_vad_frame_carry.py`,
  `Tests/Audio/test_recording_stop_terminate_guard.py`): pure logic,
  scripted fake VAD + patched backend probe, no microphone. Helper
  invariants (including chunks smaller than one frame and a zero-loss
  property over non-aligned chunk streams), byte-exact cross-chunk
  assembly, pre-roll interplay, and terminate failure/success behavior.
  Pre-fix: 9 of 10 fail; post-fix all pass.
- Files: `tldw_chatbook/Audio/recording_service.py`,
  `Tests/Audio/test_recording_vad_frame_carry.py`,
  `Tests/Audio/test_recording_stop_terminate_guard.py`.
- Commit: 903a7305d3 on `fix/audit-burn-down-38580-38613`.
- ADR required: no -- hardening/bug fixes, no architectural decision.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
