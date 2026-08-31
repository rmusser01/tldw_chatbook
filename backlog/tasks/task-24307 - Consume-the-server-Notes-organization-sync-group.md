---
id: TASK-24307
title: Consume the server Notes organization sync group
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29'
updated_date: '2026-08-30 17:15'
labels:
  - notes
  - sync-v2
  - migration
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-08-29-agent-lessons-notes-organization-sync-design.md
  - Docs/superpowers/plans/2026-08-29-notes-organization-sync-parity.md
  - backlog/decisions/105-portable-notes-organization-and-agent-lessons.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Chatbook a conforming consumer of the server's complete six-domain Notes organization group so folders, keywords, collections, and their memberships can synchronize without changing filesystem ownership or the locked `notes.note` contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chatbook enrolls and advertises all six Notes organization domains as one schema-v1 capability and refuses partial group readiness
- [x] #2 Active and soft-deleted legacy organization resources receive stable portable identities without repurposing local primary keys or silently merging same-name resources
- [x] #3 Incoming and outgoing resources, links, hierarchies, tombstones, suppressions, and dependency checks conform to the reviewed server contract and normative identity vectors
- [x] #4 Interrupted bootstrap, pull, adoption review, local mutation, outbox copy, retry, and acknowledgement preserve recoverable state without claiming cross-database atomicity
- [x] #5 Explicit folder deletion does not emit unintended descendant tombstones, while dormant descendants and memberships become effective again after restore
- [x] #6 Targeted migration, conformance, two-device, and two-real-SQLite crash tests pass, including genuine historical-schema reopen coverage
- [x] #7 ADR-105 and relevant Sync-v2 and Notes organization documentation describe the shipped ownership, enrollment, conflict, and recovery behavior
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin the complete six-domain contract and server identity vectors with red-first tests.
2. Add the v57→v58 stable-identity and durable organization-state migration using genuine historical reopen coverage.
3. Implement the cursor-aware Notes organization repository, explicit tombstone semantics, and resumable legacy inventory.
4. Apply incoming organization envelopes transactionally and journal outgoing immutable intents across the Notes DB/general outbox boundary.
5. Enroll/bootstrap/adopt the six domains as one capability and prove two-device convergence.
6. Document and run targeted plus schema-safe live verification before closure.

ADR required: yes
ADR path: backlog/decisions/105-portable-notes-organization-and-agent-lessons.md
Reason: This task changes persistent identity, schema/migrations, synchronization ownership, conflict policy, and the client/server contract.

Detailed plan: Docs/superpowers/plans/2026-08-29-notes-organization-sync-parity.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the complete schema-v1 Notes organization group under ADR-105. Chatbook now pins the server contract and identity vectors, migrates real v55 databases to v56 stable resource identities and durable checkpoints/intents/reviews/suppressions, applies and publishes all six domains transactionally at the Notes owner, resumes bootstrap/adoption/inventory, and preserves non-cascading folder tombstones plus effective-union suppression semantics.

Verification: the exact Task 9 matrix collected 2,064 tests; every feature-specific migration, contract, adapter, dispatch, enrollment, two-real-SQLite crash, two-device, API schema, and client test passed. Its 32 failures are confirmed pre-existing Tests/DB baseline failures outside this feature. The index-census intent-sequence pin was corrected; the remaining two census failures are the confirmed pre-existing four console-memory index omissions. A focused post-fix run passed 65 tests; migration/schema/index-shape coverage passed 49 tests and the 80-table allowlist gate; Ruff, MyPy on eight feature modules, compileall, and git diff --check pass.

Live-safe verification launched and quit the real app with a disposable HOME/XDG/config/data root, resolved both SQLite files under that root, and confirmed ChaChaNotes schema v56 plus all Notes organization state tables. Real transport/product UAT was unavailable because the isolated profile had no endpoint or credentials, so it was not faked; the server origin/dev contract was revalidated at 52774a0453 with no normative Notes contract delta from the reviewed baseline.

Documentation: Docs/Development/Sync-v2-client.md is the canonical runtime reference and Docs/User_Guide/library/notes.md links user behavior to it. ADR required: yes; ADR path: backlog/decisions/105-portable-notes-organization-and-agent-lessons.md. A reusable fake-transport validation lesson was added to backlog/docs/lessons-testing-evidence.md.

Final collision sweep: no separate TASK-24307 claimant was found. The personal-context worktree's decision-102 add commit (b9b83c5, 2026-08-28 22:45 -0700) predates this Notes decision's original add commit (6952980, 2026-08-29 14:43 -0700), so that older decision keeps identifier 102 and this later Notes decision moved to ADR-105. The still-later task-19504 decision-102 claimant (e62ae7c, 2026-08-30 09:21 -0700) must renumber independently.
<!-- SECTION:NOTES:END -->
