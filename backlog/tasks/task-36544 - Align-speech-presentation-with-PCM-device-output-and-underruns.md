---
id: TASK-36544
title: Align speech presentation with PCM device output and underruns
status: To Do
created_date: 2026-10-23 22:55
dependencies:
- TASK-36543
labels:
- audio
- tts
documentation:
- Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md
- Docs/superpowers/plans/2026-10-23-console-radio-portraits-implementation.md
- backlog/decisions/242-console-radio-portrait-performance.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep portrait speech levels aligned with the audio actually scheduled to the output device, including buffering, inserted silence and supported WAV artifacts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Optional presentation observation leaves existing sink callers and audio output unchanged when disabled or when observation fails.
- [ ] #2 Source-frame consumption and device-output positions remain distinct; validated output-DAC timing maps both to monotonic presentation time, including partial-block zero padding and resumed audio after underruns.
- [ ] #3 Rejected or retried feed chunks never advance envelope position; device-scheduled metadata survives until its output time and remains bounded by admitted audio plus the one-second tail reserve.
- [ ] #4 Slow or hidden readers cannot block audio, drop samples or accumulate presentation events; the realtime callback performs no image, database, UI, provider or blocking observer work.
- [ ] #5 Eligible PCM16 WAV artifacts reuse the existing sink; unqualified external timing retains working audio with status-only presentation, preserving one-voice ownership and fallback without double playback.
- [ ] #6 Fake-device tests prove delayed DAC output, full/partial underruns, backpressure, stop/replacement, invalid timestamps, observer failure and bounded cleanup; targeted sink/format regression checks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
