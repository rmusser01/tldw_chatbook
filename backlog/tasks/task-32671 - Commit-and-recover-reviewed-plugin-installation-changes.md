---
id: TASK-32671
title: Commit and recover reviewed plugin installation changes
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:18'
updated_date: '2026-10-01 01:53'
labels:
  - plugins
  - implementation
  - foundation
dependencies:
  - TASK-32670
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-foundation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make reviewed local installation and authority changes recoverable without accidentally publishing uncommitted capabilities.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Review binds exact package/catalog digests, installation identity, interpretation, selection, workspace and authority inputs; stale reviews cannot commit.
- [x] #2 Publication follows prepared snapshot and intent, durable SQLite commit, post-commit certificate, secure marker, then projections; retries use the same operation identity.
- [x] #3 Crash recovery follows every specified evidence state in another process, preserving unrelated installations and refusing automatic promotion when commitment proof is missing.
- [x] #4 Full disks, missing responses and interrupted writes preserve old or fenced committed state without recreating grants or executing package code; recovery material cannot be pruned.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes (existing accepted cross-store publication/recovery protocol). ADR paths: backlog/decisions/162-managed-agent-plugins.md and backlog/decisions/163-expanded-console-hook-runtime.md. 1. Trace reviewed F4 coordinator, protected evidence, retained-byte reconstruction and persistent-worker fixture. 2. Reuse behavior/crash tests and establish missing coordinator/recovery RED. 3. Integrate the reviewed six-stage commit, exact-review invalidation, bounded transition discovery and conservative reconstruction through the existing owner. 4. Qualify actual owner death at every durable milestone, fresh-process loss/rollback, missing proof, stale review, capacity/disk failure and same-operation retries using isolated child config/profile/network refusal. 5. Run targeted authority/registry neighbors and full task-owned static checks; self-review, record evidence/limits, complete ACs and commit exact files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented reviewed F4 exact session-bound reviews, immutable package retention, six-stage commit ordering and authenticated conservative recovery on the existing persistent worker/runtime owner. Commit certificates are issued only after the guarded SQLite context returns durable success. Recovery validates exact marker lineage/retained bytes, preserves unrelated authority/process evidence and refuses missing commitment proof or unknown reference owners. Bounded transition discovery retains current recovery artifacts. Final targeted commit/recovery: 51 passed; authority/registry/owner neighbors: 175 passed; no skips/warnings. Real isolated owners were killed at every durable milestone and recovered in new processes, including database loss/rollback. Ten owned Python files pass full Ruff/format; syntax/whitespace pass. ADR162/163 apply. Evidence: Docs/superpowers/reviews/2026-09-30-expanded-hooks-integration.md. No full sweep, real keychain, hardware power-loss, Linux/Windows or network/synchronized storage qualification.
<!-- SECTION:NOTES:END -->
