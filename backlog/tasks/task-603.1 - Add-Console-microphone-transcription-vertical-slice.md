---
id: TASK-603.1
title: Add Console microphone transcription vertical slice
status: Done
assignee:
  - '@codex'
created_date: '2026-07-27 15:52'
updated_date: '2026-07-27 17:30'
labels:
  - stt
  - dictation
  - console
  - ui
dependencies:
  - TASK-931
  - TASK-942
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
parent_task_id: TASK-603
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a user record speech from the native Console composer and insert the resulting English Parakeet v2 transcript into the active draft without leaving Console.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The expanded Console composer exposes a visible microphone control with clear idle, recording, and transcribing states.
- [x] #2 Clicking the control records at most 60 seconds from the default microphone and a second click stops capture and transcribes off the Textual event loop.
- [x] #3 Dictation defaults to explicit en, Parakeet v2 INT8, and the configured or verified Library-installed local bundle without triggering a download.
- [x] #4 A successful transcript is inserted at the current Console caret with sensible boundary spacing and does not send the message automatically.
- [x] #5 Missing dependencies, missing model files, microphone failures, empty audio, and transcription errors are shown clearly and leave the existing draft unchanged.
- [x] #6 Focused service and mounted Console tests cover start, stop, insertion, failure recovery, recording bounds, and composer layout.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add and test a strict 60-second wall-clock stop plus a complete-PCM-frame memory bound in AudioRecordingService/Console lifecycle.
2. Add a one-shot ConsoleDictationSession and direct NumPy-buffer Parakeet ONNX adapter so captured audio remains memory-only, uses explicit en/v2 INT8, and accepts only a configured or verified installed bundle.
3. Add the Mic control and thread-worker lifecycle to Console, with visible limit and failure states, caret insertion, no automatic send, and draft preservation.
4. Verify focused audio/UI behavior including every user-visible failure class, lint, diff hygiene, and document completion.

ADR required: yes
ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
Reason: ADR-025 already governs English Parakeet v2 INT8 dictation, memory-only buffer handling, and no implicit downloads; this child task adds the missing user-facing vertical slice without changing the future LocalSTTExecutor boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the Console Mic vertical slice with idle/recording/transcribing states, a strict 60-second and PCM-byte recording bound, off-event-loop capture/transcription, direct memory-only Parakeet ONNX buffer inference, explicit English Parakeet v2 INT8 routing, configured-or-verified local model resolution with no downloads, caret insertion without auto-send, and clear failure recovery that preserves drafts. Kept optional audio/STT imports lazy, updated Console layout and transcription documentation, and left parent TASK-603 open. PR review hardening now validates configured model paths through validate_path_simple, retains the recorder handle after stop failures so cleanup can retry, performs best-effort Console cleanup before error recovery, dispatches buffer-limit callbacks away from the recording thread, and documents the public APIs in Google style. Verification: Ruff and diff checks passed; 149 focused audio/transcription/UI tests passed; a real Parakeet v2 INT8 buffer smoke produced the expected transcript. Live microphone hardware was not exercised in this environment because its optional recording backend is not installed.
<!-- SECTION:NOTES:END -->
