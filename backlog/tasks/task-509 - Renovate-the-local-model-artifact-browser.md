---
id: TASK-509
title: Renovate the local model artifact browser
status: To Do
assignee: []
created_date: '2026-07-24 01:02'
labels:
  - stt
  - artifacts
  - ui
dependencies:
  - TASK-508
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
<!-- AC:END -->
