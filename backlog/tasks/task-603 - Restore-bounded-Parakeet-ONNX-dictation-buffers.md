---
id: TASK-603
title: Restore bounded Parakeet ONNX dictation buffers
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 01:04'
updated_date: '2026-08-09 05:31'
labels:
  - stt
  - dictation
  - onnx
dependencies:
  - TASK-602
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
  - >-
    Docs/superpowers/specs/2026-08-08-task-603-bounded-parakeet-dictation-design.md
  - Docs/superpowers/plans/2026-08-08-task-603-bounded-parakeet-dictation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve microphone and in-memory buffer transcription after legacy Parakeet removal without claiming true streaming or allowing unbounded queued audio.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The public bounded-buffer transcription path uses LocalSTTExecutor and returns normalized Parakeet ONNX results without creating another model process.
- [x] #2 The Parakeet ONNX streaming factory reports unsupported through the existing fallback contract rather than advertising true streaming.
- [x] #3 At most one dictation inference is pending; new audio coalesces within explicit duration and byte limits and never silently drops captured samples.
- [x] #4 When limits would be exceeded, capture pauses visibly with a recoverable overrun state and resumes only through an explicit user action.
- [x] #5 Dictation is selected before the next batch item without preempting active native inference, and users can pause future batch dispatch while local transcription is busy.
- [ ] #6 Latency, backpressure, cancellation, shutdown, and batch coexistence tests pass on representative supported platforms before legacy providers can be removed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
Reason: ADR-025 already governs the shared executor, bounded in-memory dictation, dictation-next admission, fallback, and release gates.
Plan: Docs/superpowers/plans/2026-08-08-task-603-bounded-parakeet-dictation.md
1. Extend LocalSTTExecutor requests from file-only to file-or-buffer with validated logical frame boundaries.
2. Share exact Parakeet artifact identity and add normalized in-memory recognition.
3. Add one bounded dictation-next admission coordinator in front of the queue-less executor.
4. Route Library and Console through the same app-owned coordinator and shutdown boundary.
5. Adapt the public facade and live dictation service without blocking the audio processing thread.
6. Wire Console busy, limit, explicit resume, and faster-whisper retry through existing UI controllers.
7. Run only focused verification, collect macOS evidence, and keep Windows/Linux/TASK-605 gates open.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented bounded Parakeet ONNX dictation under ADR-025: one app-owned LocalSTTExecutor and LocalSTTDispatchCoordinator serve Library file work and Console PCM buffers with frame-aligned 60-second/byte limits, one pending inference, generation fencing, cancellation/shutdown ownership, dictation-next admission, visible limit recovery, explicit Mic resume, and bounded local-only faster-whisper retry. The Parakeet streaming factory remains unsupported rather than claiming true streaming; dead legacy code remains intentionally retained for TASK-605. Evidence is recorded in Docs/STT_Evaluation/task-603. At rebased commit 24a2ba3cf, a real macOS Console Mic smoke opened PyAudio, captured the verified speech fixture, routed 159360 PCM bytes through v2 INT8 ONNX CPU, inserted the exact transcript at the existing caret without sending, returned Mic to idle, and emitted no failure. The live run fixed configured local-model handoff and Textual fileno-less-stderr spawning. Fresh directly related gates passed: 94 executor/facade tests and 7 limit/resume/batch-ordering nodes; focused review found no issues. AC1-AC5 are complete. TASK-603 remains In Progress because AC6 still requires representative Windows/Linux and complete release-gate evidence; the aborted changed-test union and TASK-605 default/legacy work remain open.
<!-- SECTION:NOTES:END -->
