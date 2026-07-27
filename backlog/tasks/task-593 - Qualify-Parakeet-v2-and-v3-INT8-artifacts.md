---
id: TASK-593
title: Qualify Parakeet v2 and v3 INT8 artifacts
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 01:02'
updated_date: '2026-07-27 21:35'
labels:
  - stt
  - evaluation
  - artifacts
dependencies: []
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
  - Docs/superpowers/specs/2026-07-27-stt-int8-artifact-qualification-design.md
  - Docs/superpowers/plans/2026-07-27-stt-int8-artifact-qualification.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Produce reproducible evidence that the proposed stock INT8 Parakeet ONNX artifacts are safe default candidates relative to their F32 references and the faster-whisper baseline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A versioned corpus manifest covers short, noisy, long-form, silence, English, and every proposed routed v3 language with source, license, revision, size, and digest metadata.
- [ ] #2 The harness reports WER and CER populations separately, paired-bootstrap confidence intervals, throughput, peak RSS, and timestamp validity for INT8, F32, and faster-whisper baselines.
- [ ] #3 Stock v2 and v3 INT8 artifacts are evaluated against every quality, long-form, memory, and throughput threshold in the approved design.
- [ ] #4 Results produce a machine-readable promotion decision per artifact and per v3 language; failed languages are excluded and failed INT8 blocks default promotion without silently selecting F32.
- [ ] #5 Exact Python, ONNX Runtime, onnx-asr, artifact, VAD, thread, operating-system, and hardware revisions are recorded for reproduction.
- [ ] #6 The report and harness run without modifying production routing or removing legacy providers.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md. Reason: ADR-025 already fixes the artifact candidates, runtime boundary, thresholds, VAD behavior, and promotion gates. Implement the approved design test-first in ten checkpoints: strict manifests and fingerprints; bounded corpus preparation; auditable multilingual normalization; clustered confidence intervals and fail-closed gates; local-only child adapters; isolated measurement profiles; deterministic CLI/reporting; immutable production manifests; conclusive macOS evidence; final verification and review. Detailed executable plan: Docs/superpowers/plans/2026-07-27-stt-int8-artifact-qualification.md
<!-- SECTION:PLAN:END -->
