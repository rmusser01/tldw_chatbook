---
id: TASK-22200
title: >-
  Pace the ChaChaNotes messages_fts backfill and yield to foreground writes
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - performance
  - database
  - console
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22200).

`DB/chachanotes_fts_backfill.py:82-97` drives `backfill_messages_fts` in a tight loop with
no sleep, yield, or time budget between chunks; every chunk is a `BEGIN IMMEDIATE`
tokenize+write transaction on the shared CharactersRAGDB (`DB/ChaChaNotes_DB.py:16173`).
Every UI write is also `BEGIN IMMEDIATE` (15 s busy timeout), and a chunk commit kills any
concurrent DEFERRED read-then-write writer instantly (snapshot-upgrade SQLITE_BUSY bypasses
busy_timeout — the recorded wave-1 lesson from task-21100's own review). On the first boot
after upgrading across v46 the loop runs to completion over the user's entire message
history, concurrently with the screen pre-importer, the subscriptions FTS backfill, and
actor-pack recovery (boot workers went 4 -> 7 since `35d4bf3a1`). This is the single most
plausible mechanism behind "the app recently got slower": every upgrading user gets a
first session that contends with a whole-history rebuild. The backfill itself is the fix
for a worse defect (the v46 inline freeze) — the residue is the unpaced contention window.

## Acceptance Criteria

- [ ] Inter-chunk pacing exists (sleep/yield and/or contention-aware backoff) and its effect is measured: a UI write issued mid-backfill completes within a stated bound in a probe that runs a concurrent writer against an in-flight backfill
- [ ] Total backfill duration for a large history is measured before/after and reported in the notes (pacing may lengthen it — state the trade explicitly)
- [ ] Resumability is preserved: killing the process mid-backfill still leaves a consistent, resumable index (existing state = messages_fts_docsize membership)
- [ ] The every-boot no-op probe stays cheap (one indexed scan)
