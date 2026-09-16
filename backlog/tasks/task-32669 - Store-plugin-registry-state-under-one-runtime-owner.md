---
id: TASK-32669
title: Store plugin registry state under one runtime owner
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-16 04:17'
updated_date: '2026-09-16 06:14'
labels:
  - plugins
  - implementation
  - foundation
dependencies:
  - TASK-32668
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-foundation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep plugin metadata and runtime ownership isolated from conversation storage so concurrent app instances cannot mutate or execute the same installation.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A private plugin SQLite schema v1 persists installations, revisions, selections, activation, operation intents and data ownership with bounded reads and parameterized SQL.
- [ ] #2 Exactly one app process owns plugin execution and mutation for a user-data directory; secondary instances can browse validated state without affecting other app features.
- [ ] #3 Process launch records and immutable revision leases distinguish runs, pending launches, idle connections and archived history; lock acquisition never establishes that surviving children stopped.
- [ ] #4 Private database inventories and reopen/migration tests cover the new owner; stale PID reuse never authorizes termination.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md
Reason: Direct implementation of the accepted private registry, single execution owner and durable surviving-process evidence contract.
1. Establish RED through registry/runtime-owner production entries using isolated SQLite and independent child processes.
2. Register and package schema v1 with bounded parameterized reads, validated reopen and owner-gated transactions; add the private-owner inventory documentation row.
3. Implement strict local-profile OS locking, read-only secondary browsing, and durable pending/published/unresolved process provenance without an execution or PID-termination API.
4. Exercise contention, readonly/WAL reopen, owner death and PID-reuse evidence with successful controls; run the exact feature and private-SQLite inventory regressions and changed-file static checks.
5. Update operation documentation, self-review and report evidence; controller handles exact-path Git commits, independent review and AC/Done closeout.
<!-- SECTION:PLAN:END -->
