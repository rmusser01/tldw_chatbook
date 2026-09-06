---
id: TASK-31902
title: Defer Chunking Lab action imports beyond screen preimport
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 19:25'
updated_date: '2026-09-05 19:30'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the unchanged ADR-097 screen preimport module budget after Chunking Lab joined the route registry, without changing authoring, dialogs, recovery, or template-save behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The real whole-registry preimport census fits all unchanged module and LOC caps without refreshing snapshots.
- [x] #2 An isolated regression proves recovery, template-save, and dialog implementation modules remain absent until their feature is used while region event class identities remain canonical.
- [x] #3 Complete affected Chunking Lab UI, core, database, and service tests plus scoped static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the unchanged census failure and measured eager edges; add an isolated closure regression and observe RED before production edits.
2. Relocate only Chunking Lab recovery, save, and dialog imports to their existing action methods; retain eager region classes and canonical callable identities.
3. Run the unchanged census and complete affected Chunking Lab tests; inspect and independently review the scoped diff before commit.
ADR required: no
ADR path: backlog/decisions/097-boot-budget-ratchets.md; backlog/decisions/118-chunking-lab-local-execution-and-recovery.md
Reason: Routine first-use import relocation implements the existing boot ratchet and lazy Lab ownership contract; no new boundary, dependency, or behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Moved only recovery, template-save, and dialog imports from Chunking Lab screen discovery into existing action methods. Canonical region event classes remain eager; no wrappers, compatibility facade, budget changes, or snapshot refresh. Existing ADR-097 and ADR-118 apply; no new ADR required.
Evidence: unchanged census RED at 504 modules / 378,865 LOC; isolated closure RED named all four eager implementation modules. The final complete 16-file Lab UI/core/DB/service/closure/census matrix passed 346 tests in 85.24 seconds; report: /private/tmp/tldw-31732-lab-full.xml. Final census: 500/500 modules, 377,271/380,000 LOC, Library 131,572/145,000 LOC; limits unchanged. Ruff check and format pass for both Python files; diff whitespace clean. Root independently reviewed the scoped diff with no actionable finding. Existing real save/conflict, recovery/restore/cancel, and dialog workflows remain covered. No new generalizable lesson beyond existing first-use import guidance.
<!-- SECTION:NOTES:END -->

## PR 2427 rebase renumbering provenance

Review-owned TASK-31732 was renumbered to TASK-31902 on 2026-09-06
while rebasing PR 2427 onto dev c4d45c0926. The user approved preserving
upstream task identities and renumbering review-created collisions only.
Original creation dates, task history, and literal verification artifact paths
are retained. See backlog/docs/pr-2427-rebase-reconciliation.md for the mapping.
