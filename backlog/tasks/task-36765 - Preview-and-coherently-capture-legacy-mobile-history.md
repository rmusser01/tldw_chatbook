---
id: TASK-36765
title: Preview and coherently capture legacy mobile history
status: Done
created_date: 2026-10-25 17:39
dependencies:
- TASK-36729
labels:
- console
- web-server
- mobile
- adr-244
priority: high
documentation:
- Docs/superpowers/plans/2026-10-25-shared-console-mobile-supervision-implementation.md
- Docs/superpowers/specs/2026-10-25-shared-console-mobile-supervision-design.md
- backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
updated_date: 2026-10-25 23:45
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make legacy mobile data and its destination reviewable and safely capturable before any shared-storage import changes user history.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Read-only preview identifies the physical legacy source, admitted canonical destination candidates, conflicting identities and ambiguous cross-configuration attribution without selecting a destination silently.
- [x] #2 The three legacy stores and required referenced files participate in normal coherent recovery capture, WAL consistency and destination rollback coverage without constructor reconciliation.
- [x] #3 Unsupported schemas, unsafe paths or aliases, unknown populated data and missing dependencies block the operation with actionable issues and original bytes preserved.
- [x] #4 An absent or empty legacy source is reported without creating legacy databases, and the existing recovery CLI exposes the preview without mutating source or destination data.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
Reason: Direct implementation of accepted read-only physical-source review and coherent capture boundaries.

1. Add failing absent-source/no-constructor-write preview tests and build reusable real-schema legacy/canonical fixtures with full history/reference/authority cases for later consolidation.
2. Resolve admitted physical source and canonical paths, exact schema/data/dependency inventory and ambiguous shared-source destination attribution without mutation or heuristic ownership.
3. Add explicit legacy-store/reference capture participants through existing MaintenanceSession, WAL-consistent capture, owner/read catalogs and destination rollback coverage.
4. Expose preview-only recovery CLI and service with actionable unsupported schema/alias/unknown populated data/missing dependency issues; preserve original source bytes and avoid DB construction.
5. Run specified targeted preview/inventory/capture/service/schema checks including WAL/absence/ambiguity/unsafe-source cases, scoped defaultRuff/nativeformatratchet/syntax/diff, update docs, commit explicit paths and obtain independent spec+quality review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented read-only exact legacy-mobile preview and explicit canonical destination/collision metadata under ADR-244. Existing native owners now declare legacy stores and qualified run artifacts for normal maintenance/WAL capture; unknown or missing legacy content blocks even partial capture. Added real closed-owner fixtures, preview CLI/service, stable destination namespace plus strict physical witnesses, architecture/user docs, and owner-lifetime testing lesson. Verification: required five modules 119 passed with 2 inventory failures reproduced on immutable Task 7 BASE c21e2d0c32; final metadata/public-capture run 19 passed, exact-source/public-capture/schema follow-up 30 passed. Scoped AST, full-default Ruff exact-BASE per-file comparison, native format snapshot/verify and diff check pass. Detailed commands, outputs, fixture/Task8 contracts and inherited-failure caveats are in .superpowers/sdd/2026-10-25-shared-console-mobile-supervision-implementation/task-7-report.md. AC/Done and independent review remain with controller.
Independent Task7 spec/quality review completed after physical case-alias fix1e17471a79(implementationa417fde3e1); scoped re-review clean. Real admitted sources group by dev/inode independently of strict path-bound fingerprint; differently spelled witnessed destination aliases refuse, including absent descendants, no lowercase/authority expansion. Stable metadata namespace hash is separate from immutable complete ancestor/object witnesses; Task8 must validate/journal maintenance-owned creation/replacement transitions before new witnesses. Implemented constructor-free exact-schema preview, immutable counts/IDs/dependencies/conflicts/candidates/capture+rollback scope, explicit legacy native-owner declarations and normal MaintenanceSession/WAL/reference capture; preview-only CLI/service creates no absent paths. Native helper extraction retains current/historical binding-kind compatibility and strict path/unknown-content blockers even allow_partial. Fixture has real branch/workspace/exclusion/automatic/Goal/artifact rows, explicit owner teardown; concrete fixture lifecycle lesson recorded. Evidence119pass/2exactTask7BASEfailures, final19/30/2passes, fix3realcasealiasRED→24covering+3postformatpasses/noskips;0introduced perfileRuff/17AST/nativeformatratchet committedHEAD/newformat/diffpass. Two inventory failures reproduced exactc21BASE; original1bb source comparison supports inherited origin but no original1bb runtime rerun claimed. Source bytes/deliveryNULL/preparedclaim preserved in native captures. ADR-244 applies; Task8 owns quiescence/import/holds/witnessjournal/cutover, no user-data migration/fullsuite/push/merge claimed. Full report and retained logs in plan-specific task-7-report.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
