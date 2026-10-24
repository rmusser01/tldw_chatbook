---
id: TASK-36552
title: Persist source-bound portrait regions and applied composition provenance
status: Done
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
assignee:
- '@codex'
updated_date: 2026-10-24 16:11
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make one-time region selection reusable across editor restarts while keeping active portraits and ordinary portable exports independent of preparation settings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An executed next-version migration preserves all existing performances and stores one bounded current region recipe per local character/expression with exact source identity and an independent optimistic revision.
- [x] #2 Explicit Save regions persists only validated preparation settings; stale source/revision/profile cannot overwrite newer settings or change active frames/performance revision; an unchanged current-revision Save is a no-op.
- [x] #3 Saved settings and applied provenance use the same canonical base_pixel_sha256 contract; equivalent prepared PNG encodings retain region revision, reopen and retry eligibility, while original source identity and true RGBA/geometry changes remain fenced.
- [x] #4 Apply publishes six frames and optional composition provenance in one transaction, checking both performance and captured region revisions; existing callers without provenance retain prior behavior; unchanged stored roles are content-validated and retained byte-for-byte under CAS rather than rejected or rewritten solely by encoder differences.
- [x] #5 Reopened composited sets recover finished components only after exact recipe/canonical-base/frame invariant validation, without double feathering; old/manual/invalid provenance retains ordinary six-frame artwork without inferred lineage.
- [x] #6 Full-profile restore retains settings/provenance and supports another edit; soft-delete/restore preserves them, permanent deletion purges them, Remove performance retains saved regions, and portable character/Actor Pack/Chatbook/sync output remains unchanged.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/243-source-bound-portrait-compositing.md
Reason: Implement the approved independent region-settings CAS and optional composition provenance; no new storage boundary.

1. Read the task, revised Unit 2, existing source-fenced performance writes, executed migration registry and recovery/export validators.
2. Write failing SQLite tests for independent/idempotent settings Save, source and region CAS, canonical encoding equivalence, optional provenance, unchanged stored bytes and legacy upgrade.
3. Allocate the next live schema version and implement bounded recipe serialization/settings repository; extend Apply/get/Remove with backward-compatible optional provenance and both revision fences.
4. Verify rollback, malformed/unknown metadata fallback, soft/permanent delete, full native restore/fresh-open/edit and portable export/sync boundaries.
5. Run targeted repository/migration/SQL packaging/restore/export checks plus affected callers, lint/format and diff guards. Update architecture/evidence, self-review and commit before TASK-36553.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented schema 83, independent source-bound region Save/CAS and atomic optional applied recipe provenance under both revisions (ADR-243). Unchanged stored roles are decoded/content-checked and retained byte-for-byte under CAS; manual Apply/Remove clear provenance, Remove retains regions. Native recovery admission retains historical variants and includes the fixed v82→v83 migration. No portable or sync fields added. Repository/migration/admission/SQL gate 131 passed; final added controls and affected UI/core/Chatbook callers 71 passed; actual Actor Pack/JSON export and migration packaging 12 passed; native full F9 restore/fresh-open/recover/another-edit 1 passed. Ruff formatting passes; baseline comparison 673→669 diagnostics with no new findings; diff check passes. Self-review complete, no unresolved task findings. See Docs/superpowers/reviews/2026-10-24-task-36552-review.md and Docs/Architecture/character-chat.md. Original source preserved; no full suite/live provider claim.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
