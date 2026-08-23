---
id: TASK-21100
title: >-
  First-boot migration wall - move the v46 messages_fts rebuild out of the version-bump transaction
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - database
  - migrations
  - startup
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1` (2026-08-22). Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21100).

First boot after an upgrade replays schema v34->v46 (12 migrations) inside `TldwCli.__init__`
(app.py:5808 thread-pool join -> `_init_notes_service`), before anything paints.
`chachanotes_v45_to_v46_sync_log_retention.sql` (merged 2026-08-22, PR #1974) unconditionally
rebuilds the entire `messages_fts` index (`delete-all` + reinsert of every non-deleted message)
plus nine full-`sync_log` purge scans, all in ONE transaction (runner
`ChaChaNotes_DB.py:5936-6021`). On a large profile on a slow disk this is tens of seconds to
minutes of silent pre-paint hang, with the WAL ballooning to roughly index size. This is very
plausibly the literal "it got slow after updating" complaint. The migration's privacy goal must
be preserved; the delivery mechanism is the problem.

## Acceptance Criteria

- [ ] The v45->v46 FTS rebuild runs as a resumable, chunked backfill outside the schema-version-bump transaction (in-repo exemplar: `Subscriptions/fts_backfill.py`), so an interrupted upgrade neither bricks the DB nor loses the version bump
- [ ] A visible "upgrading database..." state (splash or equivalent) is shown while any pending migration chain runs, instead of a silent hang
- [ ] A timed probe on a seeded scratch DB (stamped v45, >=10k messages) demonstrates the first-paint path no longer blocks on the full FTS rewrite, with before/after numbers recorded in the task
- [ ] Existing migration tests stay green; the v46 privacy semantics (sync_log retention, deleted-guards) are unchanged
