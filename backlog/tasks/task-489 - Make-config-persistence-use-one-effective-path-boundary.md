---
id: TASK-489
title: Make config persistence use one effective-path boundary
status: To Do
assignee: []
created_date: '2026-07-23 13:55'
labels:
  - security
  - privacy
  - config
dependencies:
  - TASK-488
priority: high
---

## Description

Eliminate contradictory configuration targets and stale settings snapshots so
encryption and provider calls consistently use the active configuration
selected by the user.

## Acceptance Criteria

- [ ] All config creation, save, delete, encryption, disable, and password-change operations target the effective config path.
- [ ] Config persistence is atomic and owner-only on POSIX.
- [ ] Provider and UI consumers observe saved settings without a process restart.
- [ ] Production modules no longer import a mutable settings snapshot.
- [ ] Concurrent in-process config operations remain serialized.
- [ ] Behavioral tests cover override paths, encryption round-trips, live settings refresh, and direct-write boundary enforcement.

## Architecture

- [ADR-022: Local Private Data Boundary](../decisions/022-local-private-data-boundary.md)
- [Local Privacy Containment Design](../../Docs/superpowers/specs/2026-07-23-local-privacy-containment-design.md)
