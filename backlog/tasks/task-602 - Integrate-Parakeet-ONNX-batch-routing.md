---
id: TASK-602
title: Integrate Parakeet ONNX batch routing
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 01:04'
updated_date: '2026-08-08 15:59'
labels:
  - stt
  - onnx
  - ingestion
dependencies:
  - TASK-593
  - TASK-595
  - TASK-599
  - TASK-600
  - TASK-601
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
- [ ] #4 INT8 is selected by default and F32 only when explicit; only artifacts and languages approved by TASK-593 can participate in semantic default routing.
- [ ] #5 Long-form Parakeet uses the exact managed VAD dependency offline with VAD ASR batch size one and cancellation checks before every segment batch.
- [ ] #6 Audio and video batch ingestion use the app-owned executor and normalized provenance, never download in a worker, and offer an explicit Retry with faster-whisper action on eligible clear failures.
- [ ] #7 Every required wheel platform passes package resolution, probe, INT8 v2 and v3 CPU smoke, long-form, cancellation, batch reuse, and retry tests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Complete the remaining parent gaps in seven focused slices: (1) pin CPU-only Parakeet package profiles; (2) add exact v2/v3 INT8/F32 root descriptors with a managed Silero VAD dependency; (3) carry explicit precision through batch routing; (4) implement one executor-native offline Parakeet runtime with batch-size-one VAD cancellation; (5) wire managed closure paths and normalized provenance through LocalSTTExecutor and the existing parent writer/retry flow; (6) make the first-run speech step install and configure the exact selected model/precision; and (7) run only focused tests plus native macOS smoke, preserving unavailable Windows/Linux gates. ADR required: no. ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md. Reason: ADR-025 already governs the affected artifact, runtime, routing, provenance, package, and retry boundaries. Detailed plan: Docs/superpowers/plans/2026-08-08-task-602-parakeet-onnx-batch-routing.md
<!-- SECTION:PLAN:END -->
