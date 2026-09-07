---
id: TASK-930
title: Use a local Parakeet ONNX bundle in batch ingestion
status: Done
assignee:
  - '@codex'
created_date: '2026-07-27 15:23'
updated_date: '2026-07-27 15:27'
labels:
  - stt
  - ingestion
  - import
dependencies: []
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a user point Library audio/video ingestion at an existing verified Parakeet v2 ONNX bundle, without editing config.toml or changing the transcription runtime.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Parakeet ONNX batch options expose a local model-directory field that is hidden or disabled for faster-whisper.
- [x] #2 The selected directory is preserved through job normalization and audio/video processing into TranscriptionService as model_dir.
- [x] #3 A blank per-job directory continues to use the configured model directory, while missing or incomplete bundles fail with the existing clear error.
- [x] #4 Focused UI, option-routing, and real batch smoke verification cover the local bundle path.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing tests for a Parakeet-only model-directory option and end-to-end option preservation.
2. Route the directory through the existing Library job options and audio/video processor parameters into TranscriptionService.transcribe(model_dir=...).
3. Preserve the existing config fallback when the per-job field is blank; do not add a downloader or new service layer in this slice.
4. Run focused tests and a real parse_local_file_for_ingest smoke using the verified local v2 INT8 bundle.

ADR required: yes
ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
Reason: ADR-025 already requires explicit local ONNX bundle selection and forbids provider-initiated downloads; this task exposes that existing boundary to the user.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Exposed a Parakeet-only local model folder in the existing Library audio/video options and routed it through job normalization, local audio/video processing, and TranscriptionService model_dir. Faster-whisper ignores stale Parakeet paths, and blank paths remain None so the existing configured-directory fallback still applies. No new coordinator, downloader, store, or runtime abstraction was added.

Verification: the 117 focused ingestion, UI, processor, and Parakeet ONNX tests passed; Ruff passed on the changed surface (with only the unrelated existing app.py baseline findings excluded); git diff --check passed. A real parse_local_file_for_ingest run used /private/tmp/tldw-parakeet-v2-int8 solely through transcription_model_dir and returned the exact expected transcript in 1.41 seconds.

ADR: Reused backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md; no new architectural decision.
<!-- SECTION:NOTES:END -->
