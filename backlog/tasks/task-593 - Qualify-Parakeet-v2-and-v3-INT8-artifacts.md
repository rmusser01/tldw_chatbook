---
id: TASK-593
title: Qualify Parakeet v2 and v3 INT8 artifacts
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 01:02'
updated_date: '2026-07-28 02:07'
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
Recorded a real indicative macOS comparison using 25 pinned FLEURS test utterances plus synthetic accent, deterministic -12.76 dB SNR noise, 63.36-second repeated long-form, and silence cases with pinned local model snapshots. The single run completed 89/89 scheduled rows with zero execution errors; observed quality failures are documented separately. V2 INT8/F32 measured 16.87%/15.06% WER and raw aggregate 0.0230/0.0315 RTF, but v2 timing is first-call-sensitive: INT8 was faster on only 1/5 cases and was 16.14% slower after excluding the first timed case, so the v2 timing result is inconclusive. V3 INT8/F32 measured 17.45%/10.43% WER and raw aggregate 0.0207/0.0288 RTF; faster-whisper measured 43.24% WER and 0.0693 RTF and emitted `you` on silence. Added cases.jsonl, the unedited generated report.json, corrected README.md, and provenance.json with exact source rows, generation argv, portable run identity, and independently checked local audio/model SHA-256 values and sizes. No production routing, artifact ownership, or legacy-provider behavior changed. ADR required: no. ADR path: N/A. Reason: this one-shot diagnostic is governed by ADR-025 and makes no architectural change.

Follow-up evidence verification: provenance JSON parsed successfully; all 29 recorded audio hashes and all 18 loader-relevant model hashes/sizes matched the local files and pinned snapshot metadata; the README stress table matched report rows; 70 focused tests, Ruff lint/format, report/provenance JSON parsing, unchanged cases/report checks, and diff checks passed.
<!-- SECTION:NOTES:END -->
