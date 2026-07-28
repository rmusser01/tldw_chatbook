---
id: TASK-593
title: Qualify Parakeet v2 and v3 INT8 artifacts
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 01:02'
updated_date: '2026-07-28 01:43'
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
- [x] #1 One helper script accepts a local JSONL case list and explicit local model directories for Parakeet v2/v3 INT8 and F32 plus faster-whisper
- [x] #2 The curated cases cover English, every currently routed v3 language, and representative accent, noise, silence, and long-form inputs
- [x] #3 The JSON report records per-case hypotheses, Unicode-aware WER/CER edit counts, elapsed time, audio duration, real-time factor, model identity, and environment
- [x] #4 A real macOS run produces a committed JSON report and short interpretation clearly labeled as indicative rather than an automated promotion decision
- [x] #5 The comparison does not modify production routing, artifact ownership, or legacy providers
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: this one-shot diagnostic does not change routing, storage, artifacts, or runtime contracts. Delete the abandoned qualification framework, implement the approved one-script comparison test-first, acquire the small local case set and pinned model snapshots, run the indicative macOS comparison, record the JSON report and interpretation, then complete TASK-593 hygiene. Detailed plan: Docs/superpowers/plans/2026-07-27-simple-stt-model-comparison.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Recorded a real indicative macOS comparison using 25 pinned FLEURS test utterances plus synthetic accent, deterministic noise, 63.36-second long-form, and silence cases with pinned local model snapshots. The single run completed 89/89 scheduled rows with no execution errors. V2 INT8/F32 measured 16.87%/15.06% WER and 0.0230/0.0315 RTF; v3 INT8/F32 measured 17.45%/10.43% WER and 0.0207/0.0288 RTF; faster-whisper measured 43.24% WER and 0.0693 RTF and emitted `you` on silence. Added cases.jsonl, the unedited generated report.json, and README.md. INT8 reduced matching-family RTF by about 27-28% but was less accurate; no production routing, artifact ownership, or legacy-provider behavior changed. Verification: 70 focused tests passed; Ruff check/format, JSON parsing, and git diff checks passed. ADR required: no. ADR path: N/A. Reason: this one-shot diagnostic is governed by ADR-025 and makes no architectural change.
<!-- SECTION:NOTES:END -->
