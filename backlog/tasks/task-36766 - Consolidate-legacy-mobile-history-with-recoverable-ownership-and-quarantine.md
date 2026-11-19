---
id: TASK-36766
title: Consolidate legacy mobile history with recoverable ownership and quarantine
status: To Do
created_date: 2026-10-25 17:39
dependencies:
- TASK-36759
- TASK-36764
- TASK-36765
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
Preserve existing mobile conversations, workspaces and run history in canonical Console stores without duplicate imports, lost restrictions or automatic execution of imported work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One reviewed claim binds a physical legacy source and fingerprint to admitted destination resources across configurations; another configuration cannot re-import or redirect the same source.
- [ ] #2 Journaled per-owner transactions preserve mapped conversations, message branches, settings, workspaces, exclusions, run trees, steps and referenced artifacts; each committed batch carries its own idempotency evidence.
- [ ] #3 All imported automatic-work, goal, wake/chat-start and result-delivery authority remains quarantined across startup and journal recovery, including completed undelivered results, while truthful run statuses and delivery stamps are preserved.
- [ ] #4 Interruption fences writes until resume or rollback of the same operation; verified cutover enables canonical use, retains originals and coherent recovery copies, and never overwrites newer post-cutover data with a stale backup.
- [ ] #5 Repeated, empty and conflicting imports are verified without duplicates; native recovery, history reading and a new manual turn do not release old imported execution authority.
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
