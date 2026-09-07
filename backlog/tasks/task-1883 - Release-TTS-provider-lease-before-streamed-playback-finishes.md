---
id: TASK-1883
title: 'Release the TTS provider lease before streamed playback finishes'
status: To Do
assignee: []
created_date: '2026-08-02 12:00'
labels: [tts, audio, streaming]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The streaming spoken-feedback branch holds the provider lease and operation slot for the
whole audible playback (pump awaits the sink), not just for synthesis — harmless for short acks,
but it serializes any concurrent TTS work behind playback and will bite V3 phase 2's hands-free
loop (long spoken replies). Investigate releasing the lease once the byte stream is fully
consumed while playback drains. Origin: streaming-sink Task-4 review F8 note, final-review triage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The provider lease/operation slot is released when synthesis output is fully consumed, not when playback ends.
- [ ] #2 Barge-in and displacement behavior are unchanged (pins stay green).
- [ ] #3 A concurrent TTS request during long streamed playback is not blocked by the finished synthesis's lease.
<!-- AC:END -->
