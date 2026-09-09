---
id: TASK-32150
title: Retain Kokoro ONNX native workers through cancellation and close
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 05:42'
updated_date: '2026-09-09 08:40'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A cancelled Kokoro ONNX async producer can leave its executor inference running after application backend close. Keep native work and model ownership joined while preserving incremental generation and bounded buffering.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cancellation during native ONNX execution cannot report backend cleanup complete or release model ownership before the actual worker finishes.
- [x] #2 Cancelling or closing an encoded, PCM, mixed-voice or timestamp request stops further work and preserves bounded streaming and propagated errors.
- [x] #3 A successor request succeeds after cancellation, and targeted lifecycle regressions plus a real ONNX inference-overlap run verify cleanup.
- [x] #4 ONNX model loading and asset downloads run off the event loop with retained ownership; cancellation joins work and removes temporary downloads before cleanup.
- [x] #5 A native inference failure produces a retryable structured generation error rather than exception text masquerading as audio bytes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A; existing backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md and backlog/decisions/140-official-kokoro-pytorch-runtime.md establish retained native ownership.
Reason: Close an ONNX lifecycle implementation gap within the existing backend close/cancellation contract; preserve runtime selection and public settings.

1. Reproduce early cleanup with the real upstream async-stream producer and a controlled blocked native operation; preserve the failing targeted regression.
2. Own each ONNX stream and its executor lifetime in a retained private worker, bridge chunks with bounded buffering, and join the actual executor before releasing the runtime. Route every ONNX streaming caller and direct phoneme worker through retained ownership.
3. Cover cancellation, repeated cancellation, close, generator abandonment, prefetched native work, error propagation, bounded streaming and successor output with focused regressions.
4. Run the opt-in real ONNX validator to prove Stop occurs inside session.run and cleanup waits for exit; verify complete successor playback and unchanged user configuration.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Retained ONNX stream workers own their private event loop and actual native executor through cancellation, repeated Stop, generator close and backend close. A bounded acknowledged handoff preserves streaming while all encoded, PCM, mixed-voice and timestamp routes join before model release. Constructor, voice, direct-phoneme and download work run outside the app loop with retained ownership; interrupted downloads clean their temporary files. Native errors now remain structured retryable generation errors rather than exception text returned as audio.

Failure-first targeted lifecycle tests and independent review cover blocked native work, early-close/error paths, repeated cancellation, successor behavior, bounded buffering and frontend cancellation. Real ONNX session.run overlap, full successor/repeated playback and the final installed-wheel control pass with all owners joined. Existing ADR-023/140 apply; ADR-142 describes the added language frontend within this retained worker. Evidence: Docs/QA/tts-macos-burndown-2026-09-09/{kokoro-english,kokoro-languages,integration}/README.md. The active native call is joined cooperatively rather than forcibly preempted.
<!-- SECTION:NOTES:END -->
