---
id: TASK-545
title: Deliver runnable Parakeet ONNX batch transcription vertical slice
status: Done
assignee:
  - '@codex'
created_date: '2026-07-27 14:45'
updated_date: '2026-07-27 14:53'
labels:
  - stt
  - onnx
  - ingestion
dependencies: []
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver an immediately usable local Parakeet ONNX transcription path through the existing batch ingestion service before resuming broader STT architecture work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A local WAV/audio file can be transcribed through `TranscriptionService` with the `parakeet-onnx` provider on macOS.
- [x] #2 Omitted language defaults to `en`, and the provider loads Parakeet v2 with INT8 by default.
- [x] #3 The provider requires an explicit local model directory and performs no implicit model download.
- [x] #4 The provider returns the existing dictionary shape consumed by batch ingestion.
- [x] #5 Focused unit tests and one real local transcription smoke pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a lazy `onnx-asr` availability probe and one `parakeet-onnx` branch to
   the existing `TranscriptionService`.
2. Require an explicit local v2 model directory, use CPU execution and INT8,
   and call the installed `onnx-asr` API without downloads.
3. Return the existing `{text, segments, language, provider, model}` result
   shape and expose the provider/model through existing discovery methods.
4. Add focused tests before production changes, then run one real local WAV
   smoke with the actual runtime and model.

ADR required: yes

ADR path:
`backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`

Reason: ADR-025 already approves Parakeet ONNX as the batch STT path. This is a
deliberately narrow vertical slice of that decision; it does not introduce a
new runtime boundary, persistence schema, executor, routing framework, or
default promotion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a direct parakeet-onnx branch in the existing TranscriptionService. It lazily loads onnx-asr 0.12.0 with CPUExecutionProvider, defaults to English Parakeet v2 INT8, requires and validates an explicit local model directory, performs no implicit download, caches the model, and returns both current and legacy segment keys. Added the provider-specific install extra and config key. Verified two focused tests, Ruff on changed implementation/test files, TOML parsing, diff checks, exact model bundle sizes/SHA-256 values, and a real macOS WAV smoke that transcribed the spoken sentence exactly in 1.19 seconds. Reused ADR-025; no new ADR was required. Full CI was intentionally not run per user direction.
<!-- SECTION:NOTES:END -->
