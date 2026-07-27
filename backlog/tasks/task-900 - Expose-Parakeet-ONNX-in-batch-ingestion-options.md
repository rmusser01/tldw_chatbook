---
id: TASK-900
title: Expose Parakeet ONNX in batch ingestion options
status: Done
assignee:
  - '@codex'
created_date: '2026-07-27 15:13'
updated_date: '2026-07-27 15:20'
labels:
  - stt
  - ingestion
  - ui
dependencies: []
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let Library audio and video ingestion select the already-working local Parakeet ONNX provider without editing Python or changing the transcription runtime architecture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Audio/video ingestion exposes a transcription provider selector containing Parakeet ONNX and faster-whisper.
- [x] #2 The selected provider is preserved from the Library form through job option normalization into the local audio/video processor.
- [x] #3 Parakeet ONNX keeps language default en and model default nemo-parakeet-tdt-0.6b-v2 without exposing incompatible faster-whisper model choices.
- [x] #4 Focused capability and job-submission tests cover the provider path.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing tests for a provider selector and preservation of parakeet-onnx through Library job normalization.
2. Add the provider field to the existing audio/video capability schema and forward it through the existing app option map.
3. Make the model field provider-aware using the existing enabled_when support so Parakeet ONNX does not present faster-whisper model names.
4. Run the focused capability, submission, and local media option tests, then record the real batch smoke.

ADR required: yes
ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
Reason: ADR-025 already governs the approved batch-first provider and language behavior; this task introduces no new boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the existing Library ingestion option path without a new coordinator or runtime layer. Added a Parakeet ONNX/faster-whisper provider selector, provider-aware Whisper model control, en/v2 normalization for Parakeet, and provider forwarding into both local audio and video processors. Registered the optional dependency metadata so install hints resolve to the existing transcription_parakeet_onnx extra. Added capability, UI, job-normalization, and processor-routing coverage.

Verification: 113 focused tests passed; Ruff passed for the changed surface (app.py retains five unrelated baseline findings, so the touched app path was checked with those existing rules ignored); git diff --check passed. A real Library parse_local_file_for_ingest smoke transcribed /private/tmp/tldw-parakeet-smoke.wav with the verified v2 INT8 ONNX bundle in 1.23 seconds and returned the exact expected sentence.

ADR: Reused backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md; no new architectural decision.
<!-- SECTION:NOTES:END -->
