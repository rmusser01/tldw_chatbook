---
id: TASK-602
title: Integrate Parakeet ONNX batch routing
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 01:04'
updated_date: '2026-08-12 15:26'
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
- [x] #1 The CPU installation profile pins onnx-asr cpu 0.12.0, includes compatible ONNX Runtime packages in audio, video, media-processing, transcription-parakeet, and all-tools extras, and does not combine CPU and accelerator distributions.
- [x] #2 Omitted or explicit en routes to managed Parakeet v2; explicit validated non-English routes to v3; auto, excluded or unsupported languages, and translation route to faster-whisper.
- [x] #3 V3 records requested language, effective auto, null detected language, and requested_language_not_enforced without passing a false decoder constraint.
- [x] #4 INT8 is selected by default and F32 only when explicit; only artifacts and languages approved by TASK-593 can participate in semantic default routing.
- [x] #5 Long-form Parakeet uses the exact managed VAD dependency offline with VAD ASR batch size one and cancellation checks before every segment batch.
- [x] #6 Audio and video batch ingestion use the app-owned executor and normalized provenance, never download in a worker, and offer an explicit Retry with faster-whisper action on eligible clear failures.
- [ ] #7 Every required wheel platform passes package resolution, probe, INT8 v2 and v3 CPU smoke, long-form, cancellation, batch reuse, and retry tests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Complete the implemented provider in seven focused slices documented at Docs/superpowers/plans/2026-08-08-task-602-parakeet-onnx-batch-routing.md, then close AC7 with the approved five-platform native evidence plan at Docs/superpowers/plans/2026-08-12-task-602-platform-evidence.md. The evidence addendum runs exact managed v2/v3 INT8 plus VAD on Linux x86_64/aarch64, Windows x86_64, and macOS arm64/x86_64, aggregates only same-commit green results, and keeps semantic-default promotion reserved for TASK-605. ADR required: no. ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md. Reason: ADR-025 already governs the affected runtime, artifact, platform, cancellation, reuse, and recovery boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added CPU-only `onnx-asr[cpu]==0.12.0` package profiles and exact managed
  Parakeet v2/v3 INT8/F32 roots sharing a pinned Silero VAD dependency.
- Carried explicit precision through routing and batch options, implemented
  the offline cancellable Parakeet runtime, and wired it through the existing
  app-owned executor, writer, provenance, and retry flow without worker-side
  downloads or semantic-default promotion.
- Updated first-run setup to inspect, install, activate, summarize, and persist
  the exact selected language/model/precision while retaining its skip-safe
  no-clobber gate and direct-local GGUF import.
- Focused verification passed 454 tests plus the separately permitted
  localhost acquisition test. Native Apple-silicon macOS smoke passed v2
  INT8, v3 INT8, v2 F32, long-form VAD, cancellation, resident reuse, and the
  faster-whisper retry action; see `Docs/STT_Evaluation/task-602/`.
- ADR required: no. ADR-025 already governs the implemented artifact,
  runtime, routing, provenance, package, and retry boundaries.
- Windows and Linux native gates remain unavailable and open, so AC #7 is
  unchecked and TASK-602 intentionally remains `In Progress`.
<!-- SECTION:NOTES:END -->
