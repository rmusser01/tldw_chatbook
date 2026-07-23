---
id: TASK-491
title: Contain legacy Notes sync paths and preserve file modes
status: To Do
assignee: []
created_date: '2026-07-23 13:55'
labels:
  - security
  - privacy
  - notes
  - sync
dependencies:
  - TASK-488
priority: high
---

## Description

Keep legacy Notes synchronization inside its selected root and avoid widening
permissions when synchronizing database content back to disk.

## Acceptance Criteria

- [ ] The selected root is canonicalized once and descendant symlinks or junctions are rejected.
- [ ] Files whose resolved targets escape the canonical root are never read, imported, or written.
- [ ] Supported POSIX reads use no-follow semantics where available.
- [ ] Replacing an existing note preserves its permission bits.
- [ ] New notes written by sync default to owner-only permissions on POSIX.
- [ ] Containment failures are surfaced without aborting unrelated safe files.
- [ ] Behavioral tests cover outside-root symlinks, in-root aliases, directory links, replacement races, and mode preservation.

## Architecture

- [ADR-022: Local Private Data Boundary](../decisions/022-local-private-data-boundary.md)
- [Local Privacy Containment Design](../../Docs/superpowers/specs/2026-07-23-local-privacy-containment-design.md)
