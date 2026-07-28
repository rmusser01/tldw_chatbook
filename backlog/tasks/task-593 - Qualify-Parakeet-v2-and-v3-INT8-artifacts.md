---
id: TASK-593
title: Qualify Parakeet v2 and v3 INT8 artifacts
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 01:02'
updated_date: '2026-07-28 00:06'
labels:
  - stt
  - evaluation
  - artifacts
dependencies: []
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-27-simple-stt-model-comparison-design.md
  - Docs/superpowers/plans/2026-07-27-simple-stt-model-comparison.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run a small, indicative macOS comparison of Parakeet v2/v3 INT8 against their F32 forms and faster-whisper so the next routing decision has concrete evidence without building a reusable evaluation framework.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One helper script accepts a local JSONL case list and explicit local model directories for Parakeet v2/v3 INT8 and F32 plus faster-whisper
- [ ] #2 The curated cases cover English, every currently routed v3 language, and representative accent, noise, silence, and long-form inputs
- [ ] #3 The JSON report records per-case hypotheses, Unicode-aware WER/CER edit counts, elapsed time, audio duration, real-time factor, model identity, and environment
- [ ] #4 A real macOS run produces a committed JSON report and short interpretation clearly labeled as indicative rather than an automated promotion decision
- [ ] #5 The comparison does not modify production routing, artifact ownership, or legacy providers
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: this one-shot diagnostic does not change routing, storage, artifacts, or runtime contracts. Delete the abandoned qualification framework, implement the approved one-script comparison test-first, acquire the small local case set and pinned model snapshots, run the indicative macOS comparison, record the JSON report and interpretation, then complete TASK-593 hygiene. Detailed plan: Docs/superpowers/plans/2026-07-27-simple-stt-model-comparison.md
<!-- SECTION:PLAN:END -->
