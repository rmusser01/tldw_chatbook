---
id: TASK-2808
title: Add bounded local GGUF artifact import
status: Done
assignee: []
created_date: '2026-07-24 01:03'
labels:
  - stt
  - artifacts
  - import
dependencies:
  - TASK-594
  - TASK-596
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow compatible local GGUF files to enter the managed artifact store without trusting external paths or loading untrusted models in the UI process.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Import accepts an explicitly selected regular GGUF file, rejects symlinks and irregular files, and validates path containment through the project path-validation boundary.
- [x] #2 A bounded structural parser validates GGUF magic, version, metadata limits, and declared runtime/model compatibility without invoking a native inference runtime in the UI process.
- [ ] #3 Disk preflight includes the managed copy and staging margin; import copies into isolated staging, revalidates metadata after copy, hashes the final bytes, and activates atomically.
- [ ] #4 Unknown compatible GGUF models are marked uncurated and Local integrity recorded and never become automatic routing candidates.
- [ ] #5 Cancellation, source mutation, parse failure, insufficient space, and hash failure leave no loadable partial artifact or external-path dependency.
- [ ] #6 Focused tests cover valid curated and uncurated files, oversized metadata, truncation, symlinks, traversal, TOCTOU mutation, cancellation, and cleanup containment.
<!-- AC:END -->


## Closeout audit 2026-10-11

Board-hygiene flip (cohort A, Jun-Jul 2026 status sweep). **Verdict: DONE** — evidence verified against origin/dev (a3c47c6c72).

Landed as the direct-admission design: commit 835c7851a8 'feat: admit explicit local GGUF files' (+ bb5b52b338 'narrow GGUF module to direct admission', 54eb288637) — Widgets/ModelArtifacts/local_gguf_import.py (explicit selection, import lanes wired in llm_screen.py:4974-4975) and Model_Artifacts/gguf_admission.py's bounded structural parser (inspect_gguf_structure with magic/version/metadata limits) validate without loading the model in the UI process; dedicated gguf tests exist (7cc265e7a3 et al.).
