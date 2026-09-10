---
id: TASK-32002
title: Gate restored projections and invalidate stale retrieval state
status: In Progress
assignee: []
created_date: 2026-09-07 23:58
labels:
- backup-recovery
dependencies:
- task-31978
- task-31989
- task-31990
- task-31991
- task-32000
- task-32001
updated_date: 2026-09-10 19:35
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: No stale or incompatible projection can serve restored source data, including after relaunch or rollback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No stale or incompatible projection can serve restored source data, including after relaunch or rollback.
- [ ] #2 Omitted/shared indexes obey explicit previewed retirement/quarantine and scope rules.
- [ ] #3 Retrieval resumes only after qualified compatibility and reconciliation, without automatic rebuilds.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Original component04 Task19 only. Start with pure RAG definitions/persistent-root discovery and correct unused/empty states using exact installed selectors; add actual indexing SQLite schema adapter and qualify finite definition writes through existing admission. Then establish supported persistent engine capture lifetimes, and implement original durable projection quarantine/readiness plus actual query/cache generation gates when restore plan/journal dependencies exist. No engine startup during discovery or automatic rebuild. Source map: /private/tmp/chatbook-rag-task19-source-map.md. Preserve unsupported nonempty projection coverage until actual qualification; no blanket placeholder removal.
<!-- SECTION:PLAN:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-04-restore-recovery.md#task-19)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started original Task19 discovery/capture sub-slice because missing rag.definitions/rag.projections currently block all complete inventory even with absent RAG stores. Restore quarantine/readiness still depends on original Tasks17/18 and is not claimed complete.
Inert factory and exact RAG indexing SQLite adapter implemented without importing RAG runtime or vector engines; original RAG package startup remains unchanged. Existing core borrower/transaction/close seams, literal v0 schema, source-preserving native snapshot and exact main-thread idle close verified. Definition/projection files remain explicitly unsupported until actual writer/engine qualification. Focused existing/new25 passed, latest discovery8/indexing6 passed; review caught and fixed auto env fallback. Root public factory integration then exposed semantic misuse of shared_group; removing those labels rather than weakening physical identity checks. Remaining Task19 runtime/projection readiness contracts are not complete. Source report /private/tmp/chatbook-rag-discovery-report.md.
Installed inert RAG factory integrated with public capture. Removed semantic shared_group tags; that field retains physical alias identity only. Actual public discover/classify regression covers absent stores and two retained definition files. Three public capture variants and final combined146-test cohort pass; RAG production Bandit0. Nonempty RAG definitions/Chroma and restored retrieval gates remain explicitly unqualified; Task19 stays In Progress.
Bounded RAG profile native admission implemented in config_profiles.py. Exact synchronous manager/root scopes cover mkdir, CRUD, loader self-heal, legacy migration and selected experiment output paths, preserving existing mutation/error semantics. Root rerun10 focused passed52.55s, including12 unchanged existing profile test functions in fixed-selector child. Direct existing44-test pytest cohort encountered documented root-fixture selector drift (42fail2pass); ineffective adaptation removed, source guard preserved. Scoped review no findings, report /private/tmp/chatbook-rag-profile-admission-review.md; implementation/source map /private/tmp/chatbook-rag-profile-admission-report.md. Ruff38→38/Bandit2→2, no new findings. No runtime/engine/pipeline/format or pending-marker qualification claimed; remaining original Task19 work is explicit.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->