---
id: TASK-32669
title: Store plugin registry state under one runtime owner
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:17'
updated_date: '2026-09-16 06:53'
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
ADR required: yes
ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md
Reason: Direct implementation of the accepted private registry, single execution owner and durable surviving-process evidence contract.
1. Establish RED through registry/runtime-owner production entries using isolated SQLite and independent child processes.
2. Register and package schema v1 with bounded parameterized reads, validated reopen and owner-gated transactions; add the private-owner inventory documentation row.
3. Implement strict local-profile OS locking, read-only secondary browsing, and durable pending/published/unresolved process provenance without an execution or PID-termination API.
4. Exercise contention, readonly/WAL reopen, owner death and PID-reuse evidence with successful controls; run the exact feature and private-SQLite inventory regressions and changed-file static checks.
5. Update operation documentation, self-review and report evidence; controller handles exact-path Git commits, independent review and AC/Done closeout.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented private plugin SQLite schema v1, atomic schema creation/reopen checks, bounded parameterized browsing, and a strict profile-root OS execution/mutation lock under ADR-162/163. Secondary views remain read-only; launch reservation/provenance survives owner death and never authorizes stale-PID termination. Tri-state activation and typed scope generations preserve absent/Inherit/Disabled distinctions. Active/pending revision leases remain separate from idle connections and archived history. The reviewed transaction fix denies implicit/explicit caller commits and post-rollback writes, including prepared-statement reuse. Private-owner inventory and package data are registered; operation documentation and a real-host statvfs lesson were added. Verification: 137 combined tests passed before the final owner refinement; 21 feature tests passed afterward; final transaction-fix coverage passed 33 tests. New-file Black/Ruff, syntax and whitespace checks pass. Two legacy DB files retain exactly their baseline diagnostics with no introduced violations; blanket full-file lint cleanliness is not claimed. Independent review and scoped re-review are clean. Controller built a committed-source wheel and verified the SQL resource through actual importlib.resources with exact source bytes. Runtime qualification is macOS local APFS only; unknown/network/known-synchronized roots refuse, and arbitrary third-party sync remains unsupported. No full suite, real profiles/keyring or production children were used. ADRs: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. Commits 879d0d67ef and 602d93b88f.
<!-- SECTION:NOTES:END -->
