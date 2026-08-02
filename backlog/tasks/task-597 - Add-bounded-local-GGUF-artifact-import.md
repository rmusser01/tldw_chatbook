---
id: TASK-597
title: Add bounded local GGUF artifact import
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 01:03'
updated_date: '2026-08-02 04:56'
labels:
  - stt
  - artifacts
  - import
dependencies:
  - TASK-594
  - TASK-595
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
  - Docs/superpowers/specs/2026-08-01-task-597-local-gguf-import-design.md
  - Docs/superpowers/plans/2026-08-01-task-597-local-gguf-import.md
parent_task_id: TASK-596
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow compatible local GGUF files to enter the managed artifact store without trusting external paths or loading untrusted models in the UI process.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Import accepts an explicitly selected regular GGUF file, rejects symlinks and irregular files, and validates path containment through the project path-validation boundary.
- [ ] #2 A bounded structural parser validates GGUF magic, version, metadata limits, and declared runtime/model compatibility without invoking a native inference runtime in the UI process.
- [ ] #3 Disk preflight includes the managed copy and staging margin; import copies into isolated staging, revalidates metadata after copy, hashes the final bytes, and activates atomically.
- [ ] #4 Unknown compatible GGUF models are marked uncurated and Local integrity recorded and never become automatic routing candidates.
- [ ] #5 Cancellation, source mutation, parse failure, insufficient space, and hash failure leave no loadable partial artifact or external-path dependency.
- [ ] #6 Focused tests cover valid curated and uncurated files, oversized metadata, truncation, symlinks, traversal, TOCTOU mutation, cancellation, and cleanup containment.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan: Docs/superpowers/plans/2026-08-01-task-597-local-gguf-import.md

ADR required: no
ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
Reason: direct implementation of ADR-025's approved managed local GGUF import boundary.

1. Add bounded GGUF v3 parsing and pinned transcribe.cpp admission.
2. Add content-derived descriptors and curated matching.
3. Add lease-protected staging, copy/hash, cancellation, immutable install, and activation.
4. Add Installed-view selection, progress, cancellation, and precise errors.
5. Verify focused/regression/full-suite/static gates and complete review.
<!-- SECTION:PLAN:END -->
