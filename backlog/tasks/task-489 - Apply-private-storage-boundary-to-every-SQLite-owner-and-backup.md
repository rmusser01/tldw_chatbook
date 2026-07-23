---
id: TASK-489
title: Apply private storage boundary to every SQLite owner and backup
status: To Do
assignee: []
created_date: '2026-07-23 13:55'
updated_date: '2026-07-23 14:23'
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
