---
id: TASK-491
title: Make config persistence use one effective-path and live-runtime boundary
status: To Do
assignee: []
created_date: '2026-07-23 13:55'
updated_date: '2026-07-23 14:23'
labels:
  - security
  - privacy
  - config
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
Route every configuration mutation, recovery, and credential-bearing backup through one effective-path owner while refreshing only runtime consumers whose existing contracts are live.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 First creation, save, delete, encryption enable/disable/password change, shutdown persistence, reset, raw-TOML display/replace/recovery, and config export all use config-module APIs and the effective config path.
- [ ] #2 Every config replacement is atomic and `0600` on POSIX, with file state and cache generation committed under one in-process serialization boundary.
- [ ] #3 Raw-TOML views and credential-bearing backups use the serialized encrypted representation when encryption is enabled; raw replacement preserves the encryption invariant, and backups are created privately.
- [ ] #4 Provider request boundaries and security/credential views observe the next successful save through a generation-aware immutable or defensive snapshot; production code does not import mutable settings or cache request-sensitive credentials at module scope.
- [ ] #5 ADR-004 storage defaults remain restart-bound, and ADR-006 Console session resolution continues to take precedence over persisted provider defaults.
- [ ] #6 The unrelated fallback config path and direct app/UI writes are removed; a production-source guard enforces the single persistence owner.
- [ ] #7 Behavioral tests cover every display/mutation/export path, override isolation, encryption round-trips and downgrade prevention, concurrent in-process reads/writes, provider refresh, and preserved restart/session boundaries.
<!-- AC:END -->
