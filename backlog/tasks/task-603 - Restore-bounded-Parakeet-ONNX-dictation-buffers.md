---
id: TASK-603
title: Restore bounded Parakeet ONNX dictation buffers
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 01:04'
updated_date: '2026-08-09 04:22'
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
- [ ] #4 When limits would be exceeded, capture pauses visibly with a recoverable overrun state and resumes only through an explicit user action.
- [ ] #5 Dictation is selected before the next batch item without preempting active native inference, and users can pause future batch dispatch while local transcription is busy.
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
Implemented the bounded Parakeet ONNX dictation path under [ADR-025](../decisions/025-shared-stt-artifacts-and-runtime-routing.md). One app-owned LocalSTTExecutor and LocalSTTDispatchCoordinator now serve both Library file work and bounded Console PCM buffers; frame-based logical boundaries, one pending inference, format-derived 60-second/byte limits, generation fencing, and one-shot retry ownership prevent a second model process or unbounded queue. The Parakeet ONNX streaming factory continues to report unsupported, so this remains bounded whole-segment recognition rather than true streaming.

Console wiring uses the existing chip/controller for busy, limit, explicit physical resume, retained caret insertion without auto-send, and bounded faster-whisper retry. The retired production route is bypassed but its dead legacy implementation remains intentionally present for TASK-605. Evidence is recorded in [Docs/STT_Evaluation/task-603](../../Docs/STT_Evaluation/task-603/README.md).

Fresh post-rebase focused evidence at f8827ddff24b7415acc1b7f40dc40564b55a014d: 148 coordinator/executor/runtime/facade tests passed in 7.31s; ten exact app/Chat/UI contract nodes passed in 13.18s; review-fix boundary nodes passed and the full coordinator file passed 38 tests; changed-package py_compile and git diff --check passed. Changed-file Ruff reports only two proven pre-existing findings in Tests/Library/test_library_ingest_runner.py. Whole-branch review fixed one Important one-shot over-limit defect and approved focused re-review with no remaining finding.

The macOS runtime evidence is a non-UI fallback: the real app factory, app-owned coordinator/executor, and Parakeet v2 INT8 ONNX CPU transcribed 4.7025 seconds / 150480 PCM bytes to the exact expected English sentence in 4.407s without network or profile/store mutation. It does not prove a real Mic press or Console surface. The one required changed-test union aborted at 96% with exit 134 during a background retained Parakeet MLX native warm-up; its exact active node passed alone, but the union is not green. Real Mic/caret, visible busy ordering, live limit/explicit resume, safe real retry, Windows, Linux, and TASK-605 remain open. Mergeable implementation evidence is partial, not release-gate completion; status remains In Progress and AC4-AC6 remain unchecked.
<!-- SECTION:NOTES:END -->
