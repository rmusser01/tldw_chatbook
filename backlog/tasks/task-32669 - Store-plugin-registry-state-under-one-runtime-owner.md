---
id: TASK-32669
title: Store plugin registry state under one runtime owner
status: To Do
assignee: []
created_date: '2026-09-16 04:17'
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
