---
id: TASK-596
title: Renovate the local model artifact browser
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 01:02'
updated_date: '2026-08-01 19:56'
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
- [ ] #1 The browser exposes distinct Curated, Remote, and Installed views backed only by ModelArtifactService and catalog interfaces.
- [x] #2 Curated, Integrity verified, and Local integrity recorded provenance labels are displayed precisely and never imply malware safety.
- [x] #3 Install confirmation shows the full dependency closure, immutable source revision, license, precision, download bytes, staging requirement, destination, and free-space result.
- [ ] #4 Installed inventory shows active and retained revisions, dependencies, installed versus staging space, and deletion blockers including idle resident models.
- [ ] #5 Deletion can request an idle heavy-worker recycle but cannot bypass an active lease or silently cancel an active job.
- [ ] #6 Remote search, inventory refresh, install progress, and deletion run off the Textual event loop with bounded results and focused UI tests.
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
Claimed 2026-08-01 (ppqq clone). Design approved section-by-section; spec at Docs/superpowers/specs/2026-08-01-task-596-model-artifact-browser-design.md. Phased: Phase 1 = curated registry + pure view-model + shared widgets + Curated/Installed views + Library modal refactored onto the shared modal (AC 1 partial, 2, 3, 4, 6 local, 7, 8). Phase 2 = Remote search with resolve-on-select. Phase 3 = GGUF import + server path resolution + retire Widgets/HuggingFace. AC 5 idle heavy-worker recycle is deferred because no pool-owner unload mechanism exists; Phase 1 reports blockers without bypassing leases.

Recovery 2026-08-01: the stale ppqq claim was exhaustively checked before resuming. No corresponding local/remote branch, worktree, GitHub PR, or visible Codex task was found. Recovery continues Phase 1 in codex/task-596-model-browser-phase-1 from origin/dev; no duplicate implementation was located.

Phase 1 implementation (2026-08-01): centralized the shared managed model store, added the offline curated registry and preflight provenance, introduced pure browser state plus reusable consent/progress/activation controls, migrated Library Parakeet installation to the shared flow, and added lazy Curated and Installed Lab views. Installed includes bounded legacy-file discovery, managed/staging/free-space totals, lease-safe activation/deletion, explicit deletion confirmation, and user-triggered repair. Download Models remains available until Phase 3. Verification: focused affected suite 1184 passed, 1 expected skip; Installed/shared-widget safety follow-up 16 passed; Ruff passed on all changed Python files; mypy passed on the 10 new shared/browser modules; py_compile and git diff --check passed. The attempted repository-wide UI-inclusive run was stopped after 316 passing tests because the unrelated UI corpus projected an excessive local runtime; the scoped gate covers all changed modules plus Model_Artifacts, Local_Ingestion, STT, and console dictation. Remaining TASK-596 work: Phase 2 Remote and Phase 3 GGUF adoption/legacy browser retirement; AC #5 idle heavy-worker recycle remains deferred to a pool-owner unload mechanism.
<!-- SECTION:NOTES:END -->
