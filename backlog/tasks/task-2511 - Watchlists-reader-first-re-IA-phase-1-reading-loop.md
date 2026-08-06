---
id: TASK-2511
title: 'Watchlists reader-first re-IA, phase 1: reading loop'
status: Done
assignee: []
created_date: '2026-08-05 22:26'
updated_date: '2026-08-06 21:02'
labels:
  - watchlists
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the Watchlists Read tab a daily-driver feed-reader loop per Docs/superpowers/specs/2026-08-05-watchlists-reader-first-design.md (ADR-042). Scope-plumbed items, per-feed unread badges, mark-all-read + undo, next-unread, Read-first landing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Picking any rail node scopes the items list
- [x] #2 Per-feed unread badges render in the tree
- [x] #3 Mark-all-read is one key and undoable
- [x] #4 Next-unread and read/unread toggle keys work
- [x] #5 Read is the landing tab
- [x] #6 Tests/Watchlists green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-08-05-watchlists-reader-first-phase-1-reading-loop.md

ADR required: no (already exists)
ADR path: backlog/decisions/042-watchlists-reader-first-ia.md
Reason: ADR-042 was written at plan time and covers the re-IA; phase 1 is a direct implementation of it.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Phase 1 (reading loop) implemented per the plan's 11 tasks, one conventional
commit each on `feat/watchlists-reader-first` (`b8e34f918` docs through
`2c46b710c` verbs), TDD throughout (failing test first, then implementation).

What shipped, in plan order: scope filters + per-source unread counts in
`Subscriptions_DB` and the service layer (`e4f682ab8`, `c0526aff9`); scoped
bulk `mark_all_read` / `restore_items_new` with a `status='reviewed'` guard
so undo never resurrects rows the user has since ingested/ignored
(`3eefd4161`); FEEDS region removed, Read-first tab order and landing
(`9e65f861c`); post-review fixes (`e29b58096`); inspector collapsed for new
users (`7bbde091d`); rail tree scope drives the items list, with tree moves
reloading scope (`023c0ab17`); per-source unread badges in the tree
(`f1af96d97`); collapsed-rail "N unread" header suffix (`22967c992`); and
the reading-loop verbs (`2c46b710c`).

Key decisions and deviations from the plan's pseudo-code:

- Mark-all-read is a two-key loop (`a` catches the scope up, `u` restores
  the exact batch) rather than a confirm modal — a daily reader verb must
  cost one key, and the undo batch makes it safe.
- `space` (next unread) is bound on `ItemsPane`, not the screen, so it can
  only fire from the items region; `Input` still consumes it as text.
- `m` dispatches with `refresh=False, patch_item=item` (mark-read-on-open's
  proven contract) so a second press flips back from live state instead of
  re-deriving from a stale dict; the tree-counts refresh is requested
  explicitly alongside.
- The undo batch matches on the raw `item["item_id"]` ints; the plan's
  pseudo-code `int(item["id"])` would have raised, since normalized ids look
  like `local:watchlist_item:5`.

Two real bugs found by the new tests and fixed in the verbs commit:

- Typing in the items search box lost focus after the first character:
  `search_query` is `reactive(..., recompose=True)`, so every keystroke
  destroyed the focused input. `ItemsPane.recompose()` now captures focus
  before the teardown and restores it (caret at end) to the fresh input,
  using the guarded `self.screen.focused` accessor (`App.focused` raises
  `ScreenStackError` during stack transitions), and the input gets
  `select_on_focus=False` so the refocused query is caret-appended instead
  of selected-all-and-replaced.
- Follow-up noted, not in scope: `SourcesPane` has the same
  recompose-on-keystroke pattern on its own search box; its `recompose()`
  override (TASK-1035/1345) only re-homes create-form focus, so the same
  first-character-only bug likely exists there.

Verification:

- `Tests/Watchlists/`: 428 passed (was 415 before the verbs).
- `Tests/UI/test_destination_shells.py` +
  `test_destination_visual_parity_correction.py`: all watchlists
  parametrizations pass. The 4 `schedules` parametrization failures in the
  parity file are pre-existing/environmental (`ValueError: Local media DB is
  required` inside `schedules_workbench` loading): the same tests fail
  identically on the main checkout without the branch (7 schedules failures
  there), and the branch's diff against its merge-base (`06bf63a62`)
  contains no scheduling files. Outside watchlists/subscriptions/docs the
  branch touches only the parity test file, `Tests/conftest.py`, and two
  CSS files (all part of the re-IA), none of them near the schedules
  workbench.
- Repo sweep (`Tests/Subscriptions`, the three `Tests/DB/test_subscriptions_*`
  files, `Tests/Utils/test_subscriptions_dependency_gate.py`,
  `Tests/tldw_api/test_watchlists_client.py`): 757 passed, 1 failed --
  `test_briefing_selection.py::test_overflow_and_watermark_stay_exact_...`,
  a pre-existing TIME BOMB, not a branch regression: it seeds items at
  fixed `2026-07-29` timestamps with no injected clock, while
  `select_briefing_items` floors the first briefing's window at
  `now - FIRST_WINDOW_DAYS` (7 days). As of 2026-08-06 the seeds fall
  outside the window, so 0 items are selected. Neither the test nor
  `briefing_selection.py` is touched by this branch.
- Watchlists UI pilots (`Tests/UI/test_watchlists_destination_shell.py`,
  `test_watchlists_content_pane.py`, `test_watchlists_overview_loading_state.py`,
  `test_watchlists_source_vocabulary.py`): 134 passed.
- Manual TUI smoke: NOT performed (no interactive terminal in the
  implementation environment); coverage is via Textual `run_test` pilots at
  180x50 exercising the real screen.
<!-- SECTION:NOTES:END -->
