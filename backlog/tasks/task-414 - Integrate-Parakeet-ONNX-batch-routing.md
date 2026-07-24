---
id: TASK-414
title: Integrate Parakeet ONNX batch routing
status: To Do
assignee: []
created_date: '2026-07-24 01:04'
labels:
  - stt
  - onnx
  - ingestion
dependencies:
  - TASK-405
  - TASK-407
  - TASK-411
  - TASK-412
  - TASK-413
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make managed Parakeet v2 and v3 ONNX the gated batch STT paths while retaining faster-whisper for automatic language, unsupported languages, translation, and explicit recovery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The CPU installation profile pins onnx-asr cpu 0.12.0, includes compatible ONNX Runtime packages in audio, video, media-processing, transcription-parakeet, and all-tools extras, and does not combine CPU and accelerator distributions.
- [ ] #2 Omitted or explicit en routes to managed Parakeet v2; explicit validated non-English routes to v3; auto, excluded or unsupported languages, and translation route to faster-whisper.
- [ ] #3 V3 records requested language, effective auto, null detected language, and requested_language_not_enforced without passing a false decoder constraint.
- [ ] #4 INT8 is selected by default and F32 only when explicit; only artifacts and languages approved by TASK-405 can participate in semantic default routing.
- [ ] #5 Long-form Parakeet uses the exact managed VAD dependency offline with VAD ASR batch size one and cancellation checks before every segment batch.
- [ ] #6 Audio and video batch ingestion use the app-owned executor and normalized provenance, never download in a worker, and offer an explicit Retry with faster-whisper action on eligible clear failures.
- [ ] #7 Every required wheel platform passes package resolution, probe, INT8 v2 and v3 CPU smoke, long-form, cancellation, batch reuse, and retry tests.
<!-- AC:END -->
