# Task 6 — close-out

Final task of the phase 1 delivery. Full test sweep, two backlog tasks filed
(programme tracking + a review-found follow-up), spec status flipped, one
commit.

## Sweep

`Tests/Subscriptions/ Tests/Scheduling/ Tests/Watchlists/ Tests/UI/ -k watchlist`:

```
2 failed, 662 passed, 6335 deselected in 525.80s (0:08:45)
```

Failures, both matching the documented tree-chevron baseline (not ours,
pre-existing, verified against git history predating this branch's commits):

- `Tests/UI/test_destination_visual_parity_correction.py::test_watchlists_tree_chevron_shares_a_row_with_its_watchlist[size0]`
- `Tests/UI/test_destination_visual_parity_correction.py::test_watchlists_tree_chevron_shares_a_row_with_its_watchlist[size1]`

Both fail on the same assertion: the expanded source row's indent no longer
sorts after the watchlist row's chevron column in the painted strip
comparison (`assert 4 > 5`). Unrelated to briefings; not touched by tasks 1-5.

No other failures. The documented focus-race flake (TASK-1345) did not
reproduce this run. The documented `test_chat_shell_bar.py` collection error
also did not reproduce — that file collects cleanly on its own (15 items) and
no collection-error line appeared in the sweep's summary; nothing to action
either way since the brief only says these baselines "may" appear.

`Tests/UI/test_watchlists_inspector.py` unfiltered: 34 passed, including
`test_the_queue_write_runs_off_the_event_loop_thread` (the Task 5 review-round
pattern task-1541 asks a future fix to replicate).

No failures outside the two documented baselines. Nothing BLOCKED.

## Backlog tasks filed

Verified neither filename existed in this worktree's `backlog/tasks/` before
writing (`ls`, not git); highest existing id in this worktree was 1494, so
both new files land clean at the controller-assigned 1540/1541 with no local
collision. Frontmatter/section structure copied from the newest existing file,
`task-1494 - The-readers-full-page-and-previous-snapshot-affordances-were-never-built.md`.

- **`backlog/tasks/task-1540 - Watchlists-briefings-spec-2-programme-tracking.md`**
  — programme tracker for spec #2, pointing at
  `Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md`. AC #1
  (phase 1: tables, id-watermark selection, `chat_api_call` pipeline,
  Artifacts section, queue affordance, preset-less) checked `[x]`; AC #2-4
  (presets/scripts/audio, exports+feed directory, TASK-1383 scheduling)
  unchecked.

- **`backlog/tasks/task-1541 - Watchlists-screen-item-status-writes-never-leave-the-event-loop.md`**
  — the screen-wide version of the bug Task 5's review found and fixed only
  for the new queue-toggle write. Verified directly against source before
  filing: `WatchlistsBackendController._maybe_await` really is at
  `tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py:29`
  (the dispatch's stated `Widgets/...` path is stale — I corrected it to the
  real path in the task body) and has no `to_thread`; `_update_item_status`
  (`watchlists_collections_screen.py:3923`) routes through it and is invoked
  via bare `run_worker(coroutine, exclusive=True)` from Ingest, Ignore, the
  unread toggle, and the silent mark-read-on-open path; `Subscriptions_DB.py`
  has no `busy_timeout` pragma (grepped, zero hits). AC #1 asks for a
  thread-identity test shaped like
  `test_the_queue_write_runs_off_the_event_loop_thread`; AC #2 forbids adding
  `exclusive=True` cancellation across different items' writes; AC #3 pins
  the existing item-action tests as a regression guard.

## Spec status

`Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md` line 4:
`**Status:** proposed` → `**Status:** phase 1 implemented (2026-07-30);
phases 2-4 pending`. Diff confirmed as the sole change to that file (`git
diff --stat`: 1 file, 1 line). Spec #1
(`2026-07-25-watchlists-console-rebuild-design.md`) untouched — not in the
diff at all.

## Commit

`bd883151d6d2ee8b1cf22b0d0c8f2446ff4a0edf` —
`docs(briefings): phase 1 close-out — spec status, tracking task, event-loop follow-up`
(3 files changed: the spec edit + the two new task files).
