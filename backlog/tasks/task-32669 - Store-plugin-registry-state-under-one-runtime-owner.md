---
id: TASK-32669
title: Store plugin registry state under one runtime owner
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:17'
updated_date: '2026-10-01 01:39'
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
- [x] #1 A private plugin SQLite schema v1 persists installations, revisions, selections, activation, operation intents and data ownership with bounded reads and parameterized SQL.
- [x] #2 Exactly one app process owns plugin execution and mutation for a user-data directory; secondary instances can browse validated state without affecting other app features.
- [x] #3 Process launch records and immutable revision leases distinguish runs, pending launches, idle connections and archived history; lock acquisition never establishes that surviving children stopped.
- [x] #4 Private database inventories and reopen/migration tests cover the new owner; stale PID reuse never authorizes termination.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes (existing accepted private storage and runtime ownership contracts). ADR paths: backlog/decisions/162-managed-agent-plugins.md and backlog/decisions/163-expanded-console-hook-runtime.md. 1. Read F2 registry, lifecycle and private SQLite inventory contracts and reviewed checkpoint. 2. Reuse reviewed F2 behavior tests and demonstrate missing registry/owner through the intended entry. 3. Integrate the existing private SQLite owner policy and packaged plugin schema, preserving all current-dev inventory entries and source admission guards. 4. Qualify real schema reopening, rollback, read-only secondary browse, independent child owner contention and uncertain process custody with private profile/provenance. 5. Run targeted static and neighbor checks, self-review, record evidence/platform limits, complete ACs/notes and commit exact task files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented reviewed F2 private plugin registry/schema v1 and one stable runtime lock owner. Added the existing checked SQLite policy, current-dev C93/C94 inventory entries, packaged SQL and real process controls; pending/unresolved evidence survives owner death and lock reacquisition does not authorize process termination. Targeted final qualification: 34 passed without skips/warnings; 145 neighbor cases passed and the single documentation disposition failure was corrected and rerun successfully. A noninstalled wheel contains exact migration bytes. Full task-owned Ruff/format, unchanged shared lint, syntax and whitespace pass. ADR162/163 implement the accepted boundaries; evidence: Docs/superpowers/reviews/2026-09-30-expanded-hooks-integration.md. Runtime ownership is qualified on 64-bit local macOS APFS only; no full sweep, Linux/Windows or network/synchronized filesystem qualification.
<!-- SECTION:NOTES:END -->
