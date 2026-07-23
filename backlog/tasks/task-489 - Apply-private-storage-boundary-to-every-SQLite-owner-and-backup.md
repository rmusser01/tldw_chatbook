---
id: TASK-489
title: Apply private storage boundary to every SQLite owner and backup
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-23 13:55'
updated_date: '2026-07-23 18:43'
labels:
  - security
  - privacy
  - database
  - storage
dependencies:
  - TASK-488
references:
  - backlog/decisions/022-local-private-data-boundary.md
documentation:
  - Docs/superpowers/specs/2026-07-23-local-privacy-containment-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure every Chatbook-owned file-backed SQLite database, journal sidecar, and database backup uses the verified private-path boundary instead of relying on the process umask.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A checked inventory classifies every production SQLite connection and backup owner as private file-backed, in-memory, URI/read-only, or an explicitly justified exclusion.
- [ ] #2 Every private file-backed owner creates new databases as `0600` and hardens eligible existing databases before connecting on POSIX.
- [ ] #3 Custom database paths require a trusted non-attacker-writable namespace; unsafe targets fail closed before SQLite opens them.
- [ ] #4 In-memory and supported URI/read-only connections retain their intended semantics and are not misinterpreted as filesystem paths.
- [ ] #5 WAL, SHM, rollback journals, and Chatbook-created database backups retain the private posture on supported POSIX platforms.
- [ ] #6 A source or registry guard prevents new production file-backed SQLite owners from bypassing the approved private connection seam.
- [ ] #7 Behavioral tests cover every classified owner plus first creation, existing migration, sidecars, backups, URI handling, and target/parent replacement failures.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/022-local-private-data-boundary.md
Reason: Implements the accepted SQLite private-data boundary in ADR-022 without changing it.

1. Check in the complete 31-connection, nine-backup/restore, and DB-parent-creator inventory with a machine-checked registry.
2. Add red-green trusted-directory, private SQLite connection, read-only URI (including Windows path forms), and sidecar lifecycle primitives.
3. Secure default application data directories, remove or secure non-owner DB-parent creators, and preserve lexical custom database path selection.
4. Migrate core and backup-target connections before enabling the raw-construction guard, preserving memory, pragma, and exception contracts.
5. Migrate interop, UI maintenance, cookie, notification, sync, and widget owners; enable bypass-resistant raw-construction and literal-owner guards.
6. Centralize SQLite backups and restores with explicit verified source selection/identity and a tested live-connection quiescence contract.
7. Run the behavioral owner matrix, focused and full regressions, static/source gates, independent security review, and Backlog closeout.

Detailed plan: Docs/superpowers/plans/2026-07-23-private-sqlite-owner-lifecycle.md
<!-- SECTION:PLAN:END -->
