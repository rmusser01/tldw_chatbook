---
id: TASK-36552
title: Persist source-bound portrait regions and applied composition provenance
status: To Do
created_date: 2026-10-24 14:24
dependencies:
- TASK-36551
labels:
- console
- characters
- portrait-compositing
references:
- backlog/decisions/242-console-radio-portrait-performance.md
documentation:
- Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md
- Docs/superpowers/plans/2026-10-24-portrait-compositing-implementation.md
- backlog/decisions/243-source-bound-portrait-compositing.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make one-time region selection reusable across editor restarts while keeping active portraits and ordinary portable exports independent of preparation settings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An executed next-version migration preserves all existing performances and stores one bounded current region recipe per local character/expression with exact source identity and an independent optimistic revision.
- [ ] #2 Explicit Save regions persists only validated preparation settings; a stale source, stale region revision or replaced profile cannot overwrite newer settings or change active frames/performance revision; an unchanged current-revision Save is a no-op.
- [ ] #3 Apply publishes six frames and optional composition provenance in one transaction, checking both performance and captured region revisions; existing callers without provenance retain their prior behavior.
- [ ] #4 Reopened composited sets recover reusable components without double feathering only after exact recipe/base/frame invariant validation; old/manual/invalid provenance remains usable as ordinary six-frame artwork without inferred component lineage.
- [ ] #5 Full-profile restore retains regions/provenance and supports another edit; soft-delete/restore preserves them, permanent deletion purges them, Remove performance retains saved regions, and portable character/Actor Pack/Chatbook/sync output remains unchanged.
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
