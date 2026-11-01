---
id: TASK-21240
title: >-
  An abandoned first-time setup leaves an empty notes-sync state DB that
  re-arms the start gate forever
status: Done
assignee: []
created_date: '2026-08-23'
labels:
  - bug
  - notes-sync
  - startup
  - performance
dependencies: []
priority: medium
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; the residual
recorded when TASK-21112 merged (PR #2009, `30c7e1fe9`).

TASK-21112 gated the notes-sync runtime start on real evidence of configuration — a
zero-profile boot now creates no state file at all — and backed the full-tree watcher off from
60 to **8 scans/min** when quiet. The gate predicate (`app.py`, `start_evidence`) is
`legacy_sync_directory_configured(settings) or state_path.exists()`, deliberately side-effect
free: the obvious probe would have created the store it was testing for, which is the trap
that implementation hit and recorded in the lessons.

The residual: `state_path.exists()` is satisfied by an **empty** state database. A user who
opens first-time setup — which force-starts the runtime on demand and therefore creates the
file — and then abandons it without configuring any sync directory is left with an empty store
on disk. Every subsequent boot sees the file, arms the gate, and pays the full start the gate
exists to avoid, permanently, for a feature that was never configured.

## Acceptance Criteria

- [ ] A boot following an abandoned first-time setup does not start the notes-sync runtime
- [ ] The gate predicate remains side-effect free — evaluating it never creates, opens for write, or migrates the state store
- [ ] A genuinely configured profile, and the one-time legacy `[notes]` sync-directory migration path, still start exactly as they do today
- [ ] A test covers the abandoned-setup boot and fails if the runtime starts

## Implementation Plan

1. Replace the gate's `state_path.exists()` clause with a read-only roots
   probe: a state DB counts as evidence only when `notes_sync_roots` has a
   row (a configured root).
2. Keep the predicate side-effect free: read-only URI open, SELECT only,
   never create/write/migrate; exceptions propagate so callers fail open
   (same safe direction as the runtime's own `_start_once` probe handling).
3. Share the refined predicate between the runtime's `start_evidence` and
   the new boot-construction gate (TASK-21247 lands the second caller).
4. Tests: absent-file, abandoned-empty-store, missing-table,
   configured-root (with byte-identical side-effect pin), legacy key
   unchanged, corrupt -> raises (fail-open preserved).

## Implementation Notes (2026-10-11, dev `a3c47c6c72`)

- **Fix**: `_notes_sync_state_db_has_configured_roots()` in
  `app_service_wiring.py`; both the `start_evidence` lambda and the new
  `_notes_sync_boot_start_evidence()` helper use it.
- **Evidence**: `Tests/App/test_notes_sync_boot_gate.py` -- abandoned-setup
  store returns False (the exact AC scenario: the following boot does not
  start the runtime), configured root returns True, absent file returns
  False AND creates nothing, the probe leaves the file byte-identical, the
  legacy `[notes].sync_directory` path is unchanged, and a corrupt DB
  raises so callers fail open. All green in that file.
- Boot-level corroboration: the `_ui_ready` census (warm, subprocess) shows
  the notes-sync construction chain absent on a zero-profile boot -- see
  TASK-21247 notes for the numbers.
