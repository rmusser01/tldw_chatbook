---
id: TASK-21112
title: >-
  Notes-sync runtime - gate the unconditional start, bound the legacy scan, and back off the 1Hz full-tree watcher
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - notes-sync
  - startup
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21112).

The lasting notes-sync runtime starts unconditionally at app.py:10297-10303
(`cutover_admitted=True` hardcoded at app.py:5986). Zero-profile boots still create the state
DB, run >=3 schema-censused transactions, and the first boot runs two unbounded SELECTs over
chachanotes.db (notes_sync_legacy.py:603-628). With >=1 active root, the watcher performs a
full recursive stat walk of every root every 1 second forever
(notes_sync_watcher.py:18,74-77; discovery bounds 10k entries - an over-bounds root pays the
full scan every tick before bailing). Library already falls back to `InertLastingSyncRuntime`
(library_screen.py:3212-3214).

## Acceptance Criteria

- [ ] `start()` is gated on actual configuration (non-empty root summaries, plus the legacy `notes.sync_directory` key for one-time migration); a zero-profile boot creates no notes-sync state DB and runs no legacy scans
- [ ] The watcher backs off when consecutive polls see no change (1 s -> 5-15 s with jitter, or native FS events with polling fallback); the interval is configurable
- [ ] The legacy first-boot SELECTs are bounded/paginated
- [ ] Existing notes-sync tests stay green; a probe with an active 5k-file root shows the reduced steady-state stat traffic
