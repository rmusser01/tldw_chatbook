---
id: TASK-488
title: Harden private artifact permissions and diagnostics
status: To Do
assignee: []
created_date: '2026-07-23 13:55'
labels:
  - security
  - privacy
  - storage
dependencies: []
priority: high
---

## Description

Ensure Chatbook-owned configuration, databases, SQLite sidecars, backups, and
logs do not remain accessible to unintended local users, and report honestly
when platform enforcement cannot be verified.

## Acceptance Criteria

- [ ] Fresh private artifacts use owner-only permissions on POSIX.
- [ ] Existing eligible private artifacts are automatically hardened on POSIX without following symlinks or changing unowned files.
- [ ] Insecure permission enforcement fails safely with actionable diagnostics.
- [ ] SQLite sidecars, backups, and rotated logs retain the private-file posture.
- [ ] Windows reports enforcement as unverified rather than claiming POSIX-style ACL security.
- [ ] Focused behavioral tests cover creation, migration, rotation, sidecars, ownership, and failure paths.

## Architecture

- [ADR-022: Local Private Data Boundary](../decisions/022-local-private-data-boundary.md)
- [Local Privacy Containment Design](../../Docs/superpowers/specs/2026-07-23-local-privacy-containment-design.md)
