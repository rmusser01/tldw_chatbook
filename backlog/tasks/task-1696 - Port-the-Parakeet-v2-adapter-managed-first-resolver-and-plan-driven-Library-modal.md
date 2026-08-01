---
id: TASK-1696
title: >-
  Port the Parakeet v2 adapter, managed-first resolver, and plan-driven Library
  modal
status: To Do
assignee: []
created_date: '2026-08-01 07:02'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reconciliation item 3: the merged TASK-595 downloader has zero production consumers — no descriptor exists, so nothing in the app can download anything. Port from codex/task-595-managed-downloads-v2's design: the Parakeet v2 module becomes a thin adapter supplying the first exact descriptor (pinned repo/revision/license/files/sizes/digests) plus its source map; a model-directory resolver prefers the active managed artifact, then a verified legacy .tldw-verified.json bundle, with explicitly configured directories highest priority and never described as integrity verified; the existing Library install modal renders values from the immutable plan rather than hard-coded constants, keeping its current controls and post-install batch selection. Most of what TASK-1301 needs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Parakeet v2 INT8 installs end-to-end through the shared downloader from the existing Library action
- [ ] #2 Console dictation resolves configured dir, then active managed artifact, then verified legacy bundle
- [ ] #3 The Library modal's content derives from the preflight plan
<!-- AC:END -->
