---
id: TASK-1466
title: >-
  Fix deterministic 300s hang in test_pyaudio_recording_flow that kills every full-suite run when webrtcvad is installed
status: In Progress
assignee: []
created_date: '2026-07-30 09:20'
labels:
  - testing
  - bug
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Audio/test_recording_service.py::test_pyaudio_recording_flow` stops its recording loop from inside the chunk callback — but `_process_audio_chunk` only invokes the callback for frames VAD classifies as speech, and the test's synthetic buffer (`b"\\x00\\x01" * 512`) is non-speech. With the optional `webrtcvad` extra installed (it is, locally and in CI via requirements-test.txt), the callback never fires, `is_recording` never flips, and the loop spins for the full 300s pytest timeout — after which `timeout_method="thread"` **terminates the entire pytest process**. Every full-suite run on a webrtcvad machine dies at ~3% progress. Found by the 2026-07-30 audit's baseline run (stack captured in the timeout dump); with webrtcvad absent the callback fires per chunk and the test passes, which is why it ever looked green.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [x] The test cannot hang regardless of whether webrtcvad is installed
- [x] It still exercises the pyaudio recording flow end to end (chunks delivered to the callback, loop exits via `is_recording`)
- [x] `test_sounddevice_recording_flow` — same VAD root cause, unmasked once the hang stopped killing the run (its 4-sample chunk is smaller than one 20ms VAD frame, so the frame loop never executes and the queue stays empty; fails on clean dev) — also passes
- [x] `pytest Tests/Audio/test_recording_service.py` passes on a machine with webrtcvad installed (34 passed, 1.4s; previously 300s hang + process kill)

## Implementation Plan

1. Construct the service with `use_vad=False` — VAD is not this test's subject, and it is what routes the callback away from silence
2. Bound the mock `stream.read` as a second guard so no chunk-handling change can make the loop unbounded again
3. Verify on this machine (webrtcvad installed — previously the hanging configuration)

## Implementation Notes

`use_vad=False` restores the intended flow (every chunk reaches the callback, the
callback's counter stops the loop at 3); a read-side counter also flips
`is_recording` after 10 reads as a hard bound. The sounddevice flow test had the
same root cause in a different costume — sub-frame chunks are silently dropped by
the VAD loop — verified pre-existing on clean `origin/dev` before fixing, and
fixed the same way. No production code changed.
Modified: `Tests/Audio/test_recording_service.py`.
