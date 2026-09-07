---
id: TASK-491
title: Make config persistence use one effective-path and live-runtime boundary
status: Done
assignee:
  - '@codex'
created_date: '2026-07-23 13:55'
updated_date: '2026-07-24 14:24'
labels:
  - security
  - privacy
  - config
dependencies:
  - TASK-943
references:
  - backlog/decisions/029-local-private-data-boundary.md
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
- [x] #1 First creation, save, delete, encryption enable/disable/password change, shutdown persistence, reset, raw-TOML display/replace/recovery, and config export all use config-module APIs and the effective config path.
- [x] #2 Every config replacement is atomic and `0600` on POSIX, with file state and cache generation committed under one in-process serialization boundary.
- [x] #3 Raw-TOML views and credential-bearing backups use the serialized encrypted representation when encryption is enabled; raw replacement preserves the encryption invariant, and backups are created privately.
- [x] #4 Provider request boundaries and security/credential views observe the next successful save through a generation-aware immutable or defensive snapshot; production code does not import mutable settings or cache request-sensitive credentials at module scope.
- [x] #5 ADR-004 storage defaults remain restart-bound, and ADR-006 Console session resolution continues to take precedence over persisted provider defaults.
- [x] #6 The unrelated fallback config path and direct app/UI writes are removed; a production-source guard enforces the single persistence owner.
- [x] #7 Behavioral tests cover every display/mutation/export path, override isolation, encryption round-trips and downgrade prevention, concurrent in-process reads/writes, provider refresh, and preserved restart/session boundaries.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/029-local-private-data-boundary.md
Reason: Implements ADR-029's accepted single config persistence owner and live request-boundary snapshot while preserving ADR-004 and ADR-006.

1. Add failing ownership, effective-path, encryption/raw/export, concurrency, live-provider, and restart/session-boundary tests.
2. Complete the config-owned serialized/raw APIs using descriptor-anchored private atomic replacement and a generation-aware defensive runtime snapshot.
3. Route encryption, shutdown, startup/reset, advanced raw editor/recovery, and export through those APIs; remove direct app/UI config writes and the unrelated fallback path.
4. Update request-sensitive provider and security consumers to resolve current snapshots without mutable settings imports or module-scope credentials.
5. Add production-source guards for the persistence owner and mutable/request-sensitive snapshots.
6. Run focused and broad verification, encrypted sentinel probes, self-review, and task closeout.

Detailed plan: Docs/superpowers/plans/2026-07-24-exclusive-config-persistence-runtime.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-029 config ownership and runtime publication. config.py now owns effective-path bootstrap, private atomic replacement, serialized read/backup/export, encryption-preserving raw replacement, shutdown persistence, and generation-aware defensive runtime snapshots under one lock. App startup/shutdown and Settings advanced TOML/recovery route through that owner; provider/local-provider/sidebar consumers resolve current snapshots and no production module imports mutable settings. ADR-004 restart-bound storage and ADR-006 Console session precedence remain unchanged.

Verification: 43 config regressions; 248 config-owner/runtime/Settings tests; 202 provider/Console tests (2 skipped); 22 targeted provider/live-console tests; changed-file Ruff (documented baseline ignores F401/F602/F821/F841), compileall, and git diff --check. A canonical /private/tmp encrypted sentinel probe passed all 14 assertions: override-only persistence, 0700 parent, 0600 target/backup, encrypted target, plaintext sentinel absence, defensive/generation refresh, downgrade rejection without mutation, and shutdown persistence.

ADR: backlog/decisions/029-local-private-data-boundary.md. Detailed plan: Docs/superpowers/plans/2026-07-24-exclusive-config-persistence-runtime.md.
<!-- SECTION:NOTES:END -->
