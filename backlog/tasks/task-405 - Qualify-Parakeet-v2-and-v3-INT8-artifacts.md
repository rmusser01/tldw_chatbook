---
id: TASK-405
title: Qualify Parakeet v2 and v3 INT8 artifacts
status: To Do
assignee: []
created_date: '2026-07-24 01:02'
labels:
  - stt
  - evaluation
  - artifacts
dependencies: []
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
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
