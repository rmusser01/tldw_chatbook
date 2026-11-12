---
id: TASK-40333
title: "test_a_rail_reopened_after_a_discarded_build_shows_the_landed_tree fails with NoMatches '#wl-tree' in the Dreams job's admission-sensitive step"
status: To Do
created_date: 2026-11-12 19:58
labels:
- follow-up
- tests
- watchlists
- flaky
- ci
priority: medium
references:
- Tests/Watchlists/test_watchlists_bulk_source_authoring.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-13 closeout of the 2026-10-23 burn-down, while reading CI after-times for TASK-39049 (read-only `gh`). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** The test runs in "Run Dreams admission-sensitive contracts" (`.github/workflows/derived-artifacts.yml:1159-1166`), a step of the Dreams boundary job, so when it flakes, an unrelated PR goes red. It failed in 2 of the 38 PR runs after round 12 merged (2026-11-10 ~15:40Z to 2026-11-12 17:03Z), each time as the step's only failure (`1 failed, 109 passed`):
- PR run 43142596949 (branch codex/send-feedback-dev-regressions, 2026-11-12 02:36Z, job 130937785349);
- PR run 43160188546 (branch codex/task-34569-streaming-badge, 2026-11-12 05:15Z, job 131064791688).

Both failed with `FAILED Tests/Watchlists/test_watchlists_bulk_source_authoring.py::test_a_rail_reopened_after_a_discarded_build_shows_the_landed_tree - textual.css.query.NoMatches: No nodes match '#wl-tree'`.

**Cause, by reading.**
- The test polls `_until(pilot, lambda: shown() == [watchlist["id"]])` (`:1193`). `shown()` calls `screen.query_one("#wl-tree", WatchlistTree)` (`:1190`), so while the reopened rail's tree is not mounted yet, the predicate raises instead of returning False.
- `_until` (`:663-667`) does not catch the error, so a late mount fails the test instead of being waited out.
- 6ee80bd822 ("Fix Console CI follow-up and scope Library outcome reads", merged with #3480 at 03:50Z on 2026-11-12) fixed the sibling `test_a_load_landing_while_a_rail_is_recreated_reaches_it` the same way: its poll now checks `bool(screen.query(pane_type)) and shown() == expected`. It did not touch this test. The second failure's merge ref already included that commit.

No open task names this test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test's poll returns False while the reopened rail's tree is not mounted, and then waits for the landed rows, so a late mount cannot raise; shown red first under a temporary delay of the tree's mount (or an equivalent forced late mount), then green
- [ ] #2 Every other poll in the file that queries a widget the test has just caused to mount gets the same guard
- [ ] #3 The file passes 20 of 20 runs of the admission-sensitive step's own command under load, with the load recorded
<!-- AC:END -->
