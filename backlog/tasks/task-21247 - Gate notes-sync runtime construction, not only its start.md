---
id: TASK-21247
title: >-
  Gate notes-sync runtime construction, not only its start
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - notes-sync
  - startup
  - performance
dependencies: []
priority: medium
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; finding from the
TASK-21108 review.

TASK-21112 gated the notes-sync runtime's `start()` on real evidence of configuration. Its own
evidence shows the remaining cost is in **construction**, not start.

On dev `b2b1e2e0d`, `TldwCli.__init__` builds the owner unconditionally
(`app.py:6032` `build_notes_sync_runtime_owner(...)`, with the module-level import at `:379`),
so the ~15-module chain is paid at boot by every user — including one with zero profiles
configured, the exact case TASK-21112 was written for.

On the in-flight TASK-21108 branch (`origin/fix/task-21108-wave5`, **not merged into dev** at
close-out) that construction moves behind a lazy property, but `on_mount` reads the property
unconditionally — so the chain is relocated to just before first paint rather than removed.

Either way, the same `start_evidence` predicate that already decides whether to start could
decide whether to construct. Re-confirm the exact call shape against dev after TASK-21108
merges.

## Acceptance Criteria

- [ ] A boot with no notes-sync configuration constructs no notes-sync runtime owner and imports none of its dependency chain
- [ ] A configured profile, and the one-time legacy `[notes]` sync-directory migration path, behave exactly as they do today
- [ ] The gate predicate stays side-effect free — evaluating it never creates or opens the state store
- [ ] An import-closure guard names the notes-sync chain and fails if a zero-profile boot pulls it in
