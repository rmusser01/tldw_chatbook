---
id: TASK-596
title: Renovate the local model artifact browser
status: In Progress
assignee: []
created_date: '2026-07-24 01:02'
updated_date: '2026-08-01 16:42'
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
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the existing downloader-oriented GGUF browser with a provider-neutral artifact UI for curated discovery, remote trust labeling, installed inventory, consent, versions, disk use, and deletion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The browser exposes distinct Curated, Remote, and Installed views backed only by ModelArtifactService and catalog interfaces.
- [ ] #2 Curated, Integrity verified, and Local integrity recorded provenance labels are displayed precisely and never imply malware safety.
- [ ] #3 Install confirmation shows the full dependency closure, immutable source revision, license, precision, download bytes, staging requirement, destination, and free-space result.
- [ ] #4 Installed inventory shows active and retained revisions, dependencies, installed versus staging space, and deletion blockers including idle resident models.
- [ ] #5 Deletion can request an idle heavy-worker recycle but cannot bypass an active lease or silently cancel an active job.
- [ ] #6 Remote search, inventory refresh, install progress, and deletion run off the Textual event loop with bounded results and focused UI tests.
- [ ] #7 Users can select and persist the active installed artifact revision and precision; unavailable or unverified versions cannot be selected.
- [ ] #8 The model picker, install confirmation, progress, activation, and installed-state controls are reusable by Settings and onboarding without duplicating artifact or download logic.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Claimed 2026-08-01 (ppqq clone). Design approved section-by-section; spec at Docs/superpowers/specs/2026-08-01-task-596-model-artifact-browser-design.md. Phased: Phase 1 = curated registry + pure view-model + shared widgets + Curated/Installed views + Library modal refactored onto the shared modal (AC 1 partial, 2, 3, 4, 6 local, 7, 8). Phase 2 = Remote search with resolve-on-select. Phase 3 = GGUF import + server path resolution + retire Widgets/HuggingFace. AC 5's idle heavy-worker recycle is DEFERRED: no mechanism exists to ask a worker to unload a resident model; Phase 1 reports lease blockers honestly. Claiming per the TASK-595 duplicate-implementation guard -- check this note and the spec filename before starting parallel work.
<!-- SECTION:NOTES:END -->
