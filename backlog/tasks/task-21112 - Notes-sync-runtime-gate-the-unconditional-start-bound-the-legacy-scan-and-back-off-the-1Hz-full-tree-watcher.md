---
id: TASK-21112
title: >-
  Notes-sync runtime - gate the unconditional start, bound the legacy scan, and
  back off the 1Hz full-tree watcher
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-23 10:45'
labels:
  - performance
  - notes-sync
  - startup
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21112).

The lasting notes-sync runtime starts unconditionally at app.py:10297-10303
(`cutover_admitted=True` hardcoded at app.py:5986). Zero-profile boots still create the state
DB, run >=3 schema-censused transactions, and the first boot runs two unbounded SELECTs over
chachanotes.db (notes_sync_legacy.py:603-628). With >=1 active root, the watcher performs a
full recursive stat walk of every root every 1 second forever
(notes_sync_watcher.py:18,74-77; discovery bounds 10k entries - an over-bounds root pays the
full scan every tick before bailing). Library already falls back to `InertLastingSyncRuntime`
(library_screen.py:3212-3214).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `start()` is gated on actual configuration (non-empty root summaries, plus the legacy `notes.sync_directory` key for one-time migration); a zero-profile boot creates no notes-sync state DB and runs no legacy scans
- [x] #2 The watcher backs off when consecutive polls see no change (1 s -> 5-15 s with jitter, or native FS events with polling fallback); the interval is configurable
- [x] #3 The legacy first-boot SELECTs are bounded/paginated
- [ ] #4 Existing notes-sync tests stay green; a probe with an active 5k-file root shows the reduced steady-state stat traffic
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline: run Tests/Notes notes-sync suites + ProductionApp lifecycle + library sync UI tests on base 8e949873e (teed).\n2. Gate: add an injectable start_evidence callable to NotesSyncRuntimeOwner/build_notes_sync_runtime_owner; when it reports no configuration, _start_once defers inert (new 'not_configured' status) without touching the store; app.py wires it as legacy notes.sync_directory key presence OR state-DB file presence (Path.exists probe - never opens/creates the DB).\n3. Live bring-up: start(force=True) re-arms a deferred start; review_setup forces the start so activating the first root at runtime creates the machinery on demand; Library controller treats 'not_configured' as setup-available.\n4. Watcher backoff: PollingNotesSyncWatcher doubles its sleep on consecutive no-change polls up to a jittered max (default 1s -> ~5-15s), resets to base on any detected change; base+max configurable via [notes] keys read in config.py and passed through the builder.\n5. Legacy scan: LIMIT-bound the two first-boot SELECTs in notes_sync_legacy.py with an explicit overflow error instead of unbounded fetchall.\n6. Tests red-first: zero-profile boot creates no state DB; legacy-key gate starts; deferred owner unit contracts; watcher backoff shape with fake clock/sleep; legacy bound; then full Tests/Notes + touched suites green; A/B pre-existing reds vs base.
<!-- SECTION:PLAN:END -->
