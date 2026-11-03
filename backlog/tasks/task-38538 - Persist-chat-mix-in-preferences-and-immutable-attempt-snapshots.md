---
id: TASK-38538
title: Persist chat mix-in preferences and immutable attempt snapshots
status: Done
created_date: 2026-11-01 19:30
dependencies:
- TASK-38536
labels:
- prompts
- console
references:
- backlog/decisions/286-console-prompt-mix-ins.md
documentation:
- Docs/superpowers/specs/2026-11-01-prompt-mix-ins-design.md
- Docs/superpowers/plans/2026-11-01-prompt-mix-ins-implementation.md
- backlog/decisions/286-console-prompt-mix-ins.md
assignee:
- '@codex'
updated_date: 2026-11-01 21:20
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Retain each saved chat's choices and exact retry instructions independently of mutable library definitions and optional diagnostic capture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Versioned local preferences preserve source-qualified references, enabled states, ordering and last-use facts with optimistic save and future-format preservation.
- [x] #2 Immutable used revisions and explicit empty/nonempty attempt records are atomically stored through the existing Console transaction contribution owner without copying growing provider requests or exposing bodies in metadata/logs.
- [x] #3 Real SQLite rollback, capture-off reopen, temporary promotion, copy/fork identity remapping, backup/restore and final-owner soft-delete→hard-purge tests prove retained instructions remain available and private, search stays healthy for same-title surviving owners, and new chat selections are empty.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin strict immutable values and preference preservation with RED tests; implement bounded codecs and caller-cursor tables/repository.
2. Register the descriptive SQL migration at the integrated schema version and extend constructor-free recovery policy.
3. Add the restricted transaction-writer operation/contribution; keep exact native anchors and body-free markers; integrate fork copying and existing promotion ownership.
4. Verify real SQLite rollback, immutable reuse, deletion, reopen, fork, promotion and disposable restore with targeted checks; update architecture docs and self-review.
ADR required: no
ADR path: backlog/decisions/286-console-prompt-mix-ins.md
Reason: direct implementation of the approved ADR-286 storage and lifecycle contract; existing ADR-126 recovery ownership is retained.
Controller rulings: v81→v82 is correct for the integrated branch; unrelated uncommitted portrait migrations are excluded and require integration-time renumbering/requalification. Fresh acceptance already persists staged native IDs unchanged: exact contributions receive only the native identity map; promotion uses its existing exact native→durable map, with no new acceptance fields. After exact-BASE reproduction of conversations_ad double-deleting soft-deleted FTS rows, AC3 includes the combined soft-delete→hard-purge journey. Fix only new v82 SQL/current v82 catalog, preserving all historical schema variants; qualify same-title survivors and add the evidenced lesson. Body-free assistant markers survive terminal metadata rewrites; checkpoint marker is nullable for legacy rows.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented ADR-286 operational storage: strict immutable values and preference CAS codecs; four conversation-owned tables with immutable used definitions, explicit empty attempts, ordered-item integrity and exact retry predecessors; a revoked-after-use append_mixin_attempt capability and frozen exact-ID contribution. Existing acceptance, promotion and fork transactions retain body-free markers; terminal metadata writes preserve them, forks copy owned receipts with empty preferences, and native hard purge cascades them.
ChaChaNotes v81→v82 uses the descriptive chachanotes_prompt_mixins.sql and exact current recovery catalogs/restricted candidate migration; all v81 catalogs remain historical. The controller-approved lifecycle correction guards conversations_ad against double FTS deletion after soft delete, with exact-BASE evidence and a recorded testing lesson. Existing whole-DB backup ownership and admission remain unchanged.
Verification: latest three new files 29 passed; storage/current-schema/historical migration matrix 51 passed. Existing fork/acceptance/promotion/writer/migration/package regression run: 207 passed, two exact-BASE failures (sparse context-policy postcommit write; seven unrelated decorative migration entries). Voice-pair locator and built migration packaging matrix: 106 passed plus an expected current-version-list assertion, updated and passed in the focused 51 run. Counts overlap. New-file Ruff and format pass; modified-file differential Ruff adds no diagnostics; diff whitespace check and self-review complete. Full commands/output and self-review: .superpowers/sdd/2026-11-01-prompt-mix-ins-implementation/task-3-report.md.
No new ADR: directly implements backlog/decisions/286-console-prompt-mix-ins.md while retaining ADR-126 recovery ownership. Architecture docs updated. Known integration requirement: parent/newer dev has an unrelated migration-number reservation; rebase/renumber and requalify before integration, without importing uncommitted portrait work. No full suite, runtime capture/composition/UI work, push or merge. Left In Progress for independent review.
Review round 1 fixes I1/I2: the strict fork codec accepts only the exact validated body-free mix-in marker pair alongside the fork namespace, preserving stopped/failed status, attachment labels and trace aliases while rejecting unknown fields. The v82 marker trigger normalizes valid nonobject JSON to an object before retaining markers; historical catalogs remain unchanged. Added native persisted/hydrated fork assertions, adversarial codec cases and real settled/reopened missing-record checks for object, SQL NULL, JSON null, array, string, number and boolean metadata. Exact RED/GREEN and static evidence is appended to task-3-report.md; task remains In Progress for independent review.
Independent review cleared after fixes for strict fork-envelope interoperability and nonobject JSON marker retention (7528f8e3c4..e4560ff180). New storage tests, native fork/acceptance/promotion and installed recovery checks passed in targeted runs; exact-BASE failures, aggregate FD-growth warnings and differential static debt remain qualified in the task notes. ADR-286 applies; v82 requires rebase/renumber and affected migration/restore requalification before integration with newer dev.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
