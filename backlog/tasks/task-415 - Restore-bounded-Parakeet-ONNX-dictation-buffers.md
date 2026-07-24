---
id: TASK-415
title: Restore bounded Parakeet ONNX dictation buffers
status: To Do
assignee: []
created_date: '2026-07-24 01:04'
labels:
  - stt
  - dictation
  - onnx
dependencies:
  - TASK-414
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve microphone and in-memory buffer transcription after legacy Parakeet removal without claiming true streaming or allowing unbounded queued audio.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The public bounded-buffer transcription path uses LocalSTTExecutor and returns normalized Parakeet ONNX results without creating another model process.
- [ ] #2 The Parakeet ONNX streaming factory reports unsupported through the existing fallback contract rather than advertising true streaming.
- [ ] #3 At most one dictation inference is pending; new audio coalesces within explicit duration and byte limits and never silently drops captured samples.
- [ ] #4 When limits would be exceeded, capture pauses visibly with a recoverable overrun state and resumes only through an explicit user action.
- [ ] #5 Dictation is selected before the next batch item without preempting active native inference, and users can pause future batch dispatch while local transcription is busy.
- [ ] #6 Latency, backpressure, cancellation, shutdown, and batch coexistence tests pass on representative supported platforms before legacy providers can be removed.
<!-- AC:END -->
