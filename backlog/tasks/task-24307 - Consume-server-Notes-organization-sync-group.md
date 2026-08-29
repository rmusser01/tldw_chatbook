---
id: TASK-24307
title: Consume the server Notes organization sync group
status: To Do
assignee: []
created_date: '2026-08-29'
labels:
  - notes
  - sync-v2
  - migration
priority: high
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-08-29-agent-lessons-notes-organization-sync-design.md
  - Docs/superpowers/plans/2026-08-29-notes-organization-sync-parity.md
  - backlog/decisions/102-portable-notes-organization-and-agent-lessons.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Chatbook a conforming consumer of the server's complete six-domain Notes organization group so folders, keywords, collections, and their memberships can synchronize without changing filesystem ownership or the locked `notes.note` contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Chatbook enrolls and advertises all six Notes organization domains as one schema-v1 capability and refuses partial group readiness
- [ ] #2 Active and soft-deleted legacy organization resources receive stable portable identities without repurposing local primary keys or silently merging same-name resources
- [ ] #3 Incoming and outgoing resources, links, hierarchies, tombstones, suppressions, and dependency checks conform to the reviewed server contract and normative identity vectors
- [ ] #4 Interrupted bootstrap, pull, adoption review, local mutation, outbox copy, retry, and acknowledgement preserve recoverable state without claiming cross-database atomicity
- [ ] #5 Explicit folder deletion does not emit unintended descendant tombstones, while dormant descendants and memberships become effective again after restore
- [ ] #6 Targeted migration, conformance, two-device, and two-real-SQLite crash tests pass, including genuine historical-schema reopen coverage
- [ ] #7 ADR-102 and relevant Sync-v2 and Notes organization documentation describe the shipped ownership, enrollment, conflict, and recovery behavior
<!-- AC:END -->
