---
id: TASK-36542
title: Persist validated character portrait performance sets atomically
status: To Do
created_date: 2026-10-23 22:55
dependencies:
- TASK-36509
labels:
- character-chat
- database
documentation:
- Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md
- Docs/superpowers/plans/2026-10-23-console-radio-portraits-implementation.md
- backlog/decisions/242-console-radio-portrait-performance.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep a character's six-pose performance artwork durable and tied to the exact source expression, without modifying its original portrait or portable visual packs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Exactly six unique still-image roles normalize to PNG with matching dimensions; malformed, animated, oversized and over-budget inputs are rejected and decoded buffers are released.
- [ ] #2 Fresh and upgraded character databases store one active set per character/expression; Apply and Remove are atomic, revision-checked and reject stale source or deleted owners.
- [ ] #3 Revision identities remain distinct across remove and reapply, including identical source/frame bytes; stale clients cannot overwrite or mistake a replacement for its predecessor.
- [ ] #4 Character soft delete/restore preserves attachments, permanent deletion removes their rows, and existing portraits/expression images remain byte-identical.
- [ ] #5 Full-profile backup data includes the attachments while existing card, Actor Pack, Chatbook and sync formats remain unchanged.
- [ ] #6 Targeted real-SQLite tests cover migration, rollback, stale concurrent writes, source mutation and purge; changed code passes applicable static checks.
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
