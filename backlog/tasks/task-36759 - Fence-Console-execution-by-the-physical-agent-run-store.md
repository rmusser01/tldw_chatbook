---
id: TASK-36759
title: Fence Console execution by the physical agent run store
status: To Do
created_date: 2026-10-25 17:38
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
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent native and served processes from competing for the same Console execution resource or falsely reconciling another process's live runs, while keeping non-owning inspection available.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Exactly one executing process owns an admitted physical agent_runs.db, including missing-store creation and different chat filenames sharing its directory; unsafe aliases and lock failures refuse execution.
- [ ] #2 The executing child or native App retains ownership through physical cleanup; parent death, missing heartbeats and a second process never authorize takeover.
- [ ] #3 Run-store construction, explicit reconciliation, automatic recovery and auxiliary writers require the same execution authority; non-owning inspection cannot migrate, reconcile or write the store.
- [ ] #4 Existing native Console callers keep their behavior under the owner, and real separate-process contention, retirement and read-only cases prove the boundary.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
