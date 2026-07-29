---
id: TASK-593
title: Qualify Parakeet v2 and v3 INT8 artifacts
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 01:02'
updated_date: '2026-07-28 02:30'
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
  - Docs/STT_Evaluation/task-593/README.md
  - Docs/STT_Evaluation/task-593/provenance.json
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
Refreshed the real indicative macOS evidence with reviewed runner commit `5994108c1180efe25f023a04eed857e41a7f0ba9`; the single 29-case rerun completed 89/89 scheduled rows with zero execution errors and now records `vocabulary.txt` in faster-whisper model identity. V2 INT8/F32 retained 16.87%/15.06% WER and measured raw aggregate 0.0221/0.0381 RTF, but v2 timing remains first-call-sensitive and inconclusive: INT8 was faster on 3/5 cases and 7.92% faster after excluding the first timed case in a fixed-order run with no warm-up or repetition. V3 INT8/F32 retained 17.45%/10.43% WER and measured raw aggregate 0.0276/0.0285 RTF; faster-whisper retained 43.24% WER and measured 0.0671 RTF, with `you` on silence. Updated the portable plan/README and refreshed provenance report/runner hashes, timestamp, and run bindings while preserving cases, audio, and pinned models. No production routing, artifact ownership, or legacy-provider behavior changed. ADR required: no. ADR path: N/A. Reason: this one-shot diagnostic is governed by ADR-025 and makes no architectural change.

Refresh verification: 71 focused tests passed; Ruff lint/format and report/provenance JSON parsing passed; the report has 89 rows and zero errors with `vocabulary.txt` in faster-whisper identity; all 29 audio and 18 model hashes/sizes/revisions matched local artifacts; stress rows and v2 timing matched the report; cases remained unchanged; portability, diff, and local-ignore checks passed.
<!-- SECTION:NOTES:END -->
