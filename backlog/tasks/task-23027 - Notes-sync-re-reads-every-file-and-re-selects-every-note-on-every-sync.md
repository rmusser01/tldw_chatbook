---
id: TASK-23027
title: >-
  Notes sync re-reads every file and re-selects every note on every sync
status: To Do
assignee: []
created_date: '2026-08-27'
labels:
  - performance
  - notes-sync
  - database
priority: medium
---

## Description

Three compounding inefficiencies in the notes-sync path, all pre-existing and in neither prior review:

1. **`observe_root` re-reads everything per sync.** At N=1000 with **nothing changed**: 350-370 ms,
   1,000 SELECTs, 1,017 file opens, worst event-loop stall 36-48 ms. It already computes a
   `_discovery_signature` (`notes_sync_runtime.py:651`) that is never used to skip. It runs at boot.
2. **`to_thread(lambda: asyncio.run(coro))` 9 times per synced note** - 684 us each against 0.1 us
   for a plain `await`; ~6-8 ms of the 16.5 ms per note, across 22 sites.
3. **~1 abandoned SQLite connection per synced note** - at N=1000, 1,010 opened, 0 closed.

## Acceptance Criteria

- [ ] A sync with no filesystem changes does not re-read every file or re-select every note - use the signature that is already computed
- [ ] The nested `asyncio.run`-in-a-thread pattern is removed from the per-note path
- [ ] Connections opened per sync are bounded; measured before and after at N=1000
- [ ] Worst event-loop stall during a no-op sync is measured and reported
- [ ] Sync correctness is unchanged - this area has a data-loss history, so a faster path that risks a lost or mis-ordered write is not acceptable

## Evidence

Counted live at N=1000. The loop-stall figures come from a probe with no rendering competing, so they
are a **floor**, not a ceiling.

Source: `Docs/Design/2026-08-27-holistic-perf-review.md`.
