---
id: TASK-596
title: Renovate the local model artifact browser
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 01:02'
updated_date: '2026-08-03 20:12'
labels:
  - stt
  - artifacts
  - ui
dependencies:
  - TASK-595
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
  - Docs/superpowers/specs/2026-08-01-task-596-model-artifact-browser-design.md
  - Docs/superpowers/plans/2026-08-01-task-596-model-browser-phase-1.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the existing downloader-oriented GGUF browser with a provider-neutral artifact UI for curated discovery, remote trust labeling, installed inventory, consent, versions, disk use, and deletion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The browser exposes distinct Curated, Remote, and Installed views backed only by ModelArtifactService and catalog interfaces.
- [x] #2 Curated, Integrity verified, and Local integrity recorded provenance labels are displayed precisely and never imply malware safety.
- [x] #3 Install confirmation shows the full dependency closure, immutable source revision, license, precision, download bytes, staging requirement, destination, and free-space result.
- [x] #4 Installed inventory shows active and retained revisions, dependencies, installed versus staging space, and deletion blockers including idle resident models.
- [ ] #5 Deletion can request an idle heavy-worker recycle but cannot bypass an active lease or silently cancel an active job.
- [x] #6 Remote search, inventory refresh, install progress, and deletion run off the Textual event loop with bounded results and focused UI tests.
- [x] #7 Users can select and persist the active installed artifact revision and precision; unavailable or unverified versions cannot be selected.
- [x] #8 The model picker, install confirmation, progress, activation, and installed-state controls are reusable by Settings and onboarding without duplicating artifact or download logic.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. ADR required: no
ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
Reason: ADR-025 already governs the shared model store, trust labels, acquisition, activation, and deletion boundaries.
2. Implement Phase 1 Tasks 1-4 test-first: neutral store access, curated registry, preflight provenance, and pure state mapping.
3. Implement the minimum shared install plan/progress/modal controls and migrate the existing Library Parakeet flow.
4. Add offline Curated and Installed Lab views, preserving lazy I/O and lease-safe activation/deletion behavior.
5. Run focused regression tests, static checks, and code review; leave Remote, GGUF import, legacy-browser retirement, and idle-worker recycle for their approved later phases.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed 2026-08-03. Delivered across six PRs and three parallel implementations (see Docs/superpowers/reviews/2026-08-01-task-595-duplicate-implementation-reconciliation.md for the process history): #1175 (Curated/Installed, parallel), #1185 (delta port: Escape fix, single format_mib, recompose-surviving progress), #1190 (Remote/HF discovery, parallel), #1210 (TASK-1803: curated workers to LLMScreen, fallback chain deleted), #1245 (TASK-1914: remote workers to LLMScreen, one install lock, lifecycle concurrency guard). ACs 1-4 and 6-8 met on dev. AC 5 split: the lease-safety half is delivered and tested; the idle-recycle half is TASK-2061 (no mechanism exists; cross-subsystem design against the heavy-worker pool owner). Spec Phase 3 (GGUF adoption, retire Widgets/HuggingFace) is TASK-2062. Reference branch feat/model-artifact-browser deleted after its deltas were ported and TASK-1803 landed; spec at Docs/superpowers/specs/2026-08-01-task-596-model-artifact-browser-design.md remains the design of record.
<!-- SECTION:NOTES:END -->
