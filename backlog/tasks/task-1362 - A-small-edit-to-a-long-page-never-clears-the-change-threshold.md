---
id: TASK-1362
title: A small edit to a long page never clears the change threshold
status: Done
assignee: []
created_date: '2026-07-29 23:55'
labels:
  - watchlists
  - correctness
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`change_threshold` defaults to `0.1` and is compared against `calculate_change_percentage`, which is
whole-page character similarity via `difflib.SequenceMatcher`. On a long page, a genuinely important
small edit — a price, a version number, a single added paragraph — moves that ratio by far less than
0.1, so **no item is ever created and the user is never told**.

The failure is silent and indistinguishable from "nothing changed", which is the same class of
problem as the watchlists that never checked at all (TASK-1210): the machinery works, and the user
concludes the feature does nothing.

Found while implementing TASK-1343, which made the change body renderable and therefore made the
threshold's behaviour visible for the first time.

Worth considering together: a per-region or per-element comparison, an absolute floor alongside the
ratio (e.g. "N characters changed"), or surfacing the computed percentage in the UI so a user can see
why nothing fired and tune the threshold. `baseline_manager.py` already contains structural and
key-element comparison that would help here, but it is orphaned — see TASK-1360.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A small but meaningful edit to a long page produces an item under the default configuration
- [x] #2 The rule that decides "significant" is stated in the UI or docs, so a user can tell why a change did or did not fire
- [x] #3 A test pins a realistic long-page-small-edit case and fails under the old whole-page ratio alone
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Design: `Docs/superpowers/specs/2026-07-29-watchlists-noise-not-volume-design.md`. Eight-task plan
(`.superpowers/sdd/2026-07-29-watchlists-noise-not-volume/`): Tasks 1-4 landed the data-layer
default and migration; Tasks 5-6 made `ignore_selectors` a real, visible, editable control; Task 7
surfaced per-run dispositions; Task 8 (this) stated the threshold's role in the UI and closed the
stream out.

**The four-site 0.0 move.** `change_threshold`'s default moved to `0.0` at every place the spec
named: the DB column default (`DB/Subscriptions_DB.py:210`), `monitoring_engine.py`'s runtime
fallback, `site_config_manager.py:90`'s fallback, and the orphaned `UI/SiteConfigSettings.py:302`
input value. The engine's read explicitly coerces `None` (`monitoring_engine.py:1152`) rather than
relying on `.get(key, 0.0)`, because the column key exists once a row is read from the DB -- an
explicit NULL would otherwise reach `change_percentage < None` and `TypeError` inside a scheduled
fetch. `test_every_default_threshold_site_agrees_on_zero` pins all four so the default cannot drift
by path again.

**Fingerprint re-baseline.** Each `url_snapshots` row now stores a hash of the normalized (stripped,
deduplicated, sorted) selector list plus `extraction_method`, checked *before* the content-hash
comparison. Absence counts as a mismatch, which is what makes the migration self-healing: every
pre-migration source re-baselines exactly once instead of diffing extraction-noise as a phantom
change.

**Dispositions.** `URLMonitor.check_url` returns `(item, disposition)`; the four constants
(`monitoring_engine.py:487-494`) are `baseline_stored`, `unchanged`, `withheld_below_threshold`,
`changed`. Multi-URL sources aggregate per-run counts through `stats_json`, lifted onto the run
dict's own top level by `normalize_watchlist_run` (the actual single choke point both backends'
`list_runs`/`get_run` route through -- the plan's "mirror `found_count`" guidance was checked and
found to describe nothing real in the live pipeline). `RunsPane._stats_text` renders a
`Checks: N changed | N unchanged | N withheld | N baseline` line when a run carries the key, and is
unchanged (not blank) when it does not.

**Create-form field + Inspector editor.** `ignore_selectors` had no live UI before this
(`UI/SiteConfigSettings.py` is imported by nothing). It is now a `TextArea` on both the Sources
create form (prefilled with the shipped default set, so nothing is ever stripped invisibly) and the
Inspector -- the first edit affordance any source has had at all, wired through the real
`update_source` message path. Both carry the same label/help copy as border title/subtitle,
deliberately duplicated as literals across `SourcesPane`/`InspectorPane` (importing one from the
other would close an import cycle).

**Task 8 addition (AC#2).** The help copy now also states `change_threshold`'s role, added as one
clause rather than appended after the existing sentence: the field's bottom border is 91 columns
wide at 160x42, and bisection measurement found 87 characters to be the actual cutoff before
Textual's border-label renderer silently truncates with an ellipsis -- the pre-existing sentence
alone was already 75 of those. Shipped text: "Add a rule to silence noise; changes always report;
change_threshold limits volume." (83 chars), verified rendering in full at both 160x42 and 235x52.
`InspectorPane._IGNORE_SELECTORS_HELP` was deliberately left untouched: its rail measures a fixed
~30 columns regardless of window size, where even the *original*, shorter sentence already
truncates -- a separate, pre-existing defect this task did not introduce and did not fix.

**Critical: the plan's own atomicity premise was false.** Task 2's plan asserted the schema-gate
`ALTER` and its two data-migration `UPDATE`s "share one transaction... the write gates the marker
structurally," without the plan having read `_ensure_watchlists_schema`'s actual transaction scope.
Review disproved it with a probe: Python's `sqlite3` module opens an implicit transaction only
before DML, never before DDL, so a bare `ALTER TABLE` autocommits immediately regardless of caller
intent. An exception between the ALTER and the second UPDATE left the fingerprint column present
(the one-time gate durably spent) with `change_threshold` moved but `ignore_selectors` permanently
NULL -- and unrepairable, since a clean re-run sees the column and skips entirely. Fixed with an
explicit `BEGIN IMMEDIATE` / commit-or-rollback block wrapping the ALTER and both UPDATEs
(`DB/Subscriptions_DB.py:609-633`); `test_migration_rolls_back_atomically_on_mid_migration_failure`
pins the crash-and-recover sequence.

**The three-way-vacuous geometry guard.** Task 6's first version of "the Inspector editor fits on
screen" passed for three independent reasons unrelated to the rule it claimed to prove: it asserted
`region.height > 0` (catches total collapse only -- reproduced controls at y=28..40 on an 18-row
screen while still green); it ran on a harness with no `CSS_PATH`, so `styles.max_height` was `None`
and the shipped `max-height: 4` rule never applied; and the fixture carried too few selectors to
reach the cap at all. Any one of the three alone would have kept the check green through a real
regression. Fixed with on-screen placement assertions on the real-CSS
`_visual_destination_harness`, a 30-rule fixture that genuinely hits the cap, and a styling mutation
that reds at the small size.

**Pre-existing shared-baseline bug, fixed en route.** Found while making Task 3's disposition
counts add up correctly: the "previous snapshot" query for `url_list`/`sitemap` sources selected
only by `subscription_id`, so every URL of a multi-URL source diffed against whichever URL was
checked last -- every URL after the first looked "changed" on its own first check, and no URL was
ever `unchanged`. Fixed by adding `AND url = ?` (`monitoring_engine.py:1071`,
`baseline_manager.py:604`); old mixed baselines self-heal via the same fingerprint-absence
re-baseline, so no phantom items result.

**The corrected A/B on TASK-1345's focus race.** Task 5 initially reported clean `HEAD` as 5/5
green on the `content_pane -> create_form` ordering versus this branch's ~8/17 red. Review corrected
the baseline: clean `HEAD` is already 2/9 red (TASK-1345 is pre-existing), and this branch amplifies
it to roughly 55% because the new noise-selectors field lengthens the recompose window
`Widget.focus()`'s `call_later` callback has to land inside. This task's own close-out sweep
reproduced the same signature twice more: `Tests/UI/ -k watchlist` showed 3 failures (2 known
tree-chevron + `test_create_and_cancel_sit_side_by_side_like_the_dialog`) on one run and 2
(chevron-only) on an immediate re-run of the identical command -- the race landing on yet another
test in the same file, matching TASK-1345's own note not to quote a fixed test name as its
signature. No mitigation was shipped for it here, deliberately; TASK-1345 remains open.

**AC verification.**
- AC#1/#3: `test_a_small_edit_to_a_long_page_fires_under_the_default`
  (`Tests/Subscriptions/test_watchlist_noise_not_volume.py:399`) -- a one-sentence edit to a long
  page produces a `changed` item, with an explicit `change_percentage < 1.0` precondition proving
  the edit is far below the retired `0.1` default; its own mutation (restoring the `0.1` fallback)
  reds it on `withheld: 1`.
- AC#2: two independent surfaces now state the rule. `RunsPane._stats_text` (Task 7, `eeccc18a9`)
  prints the per-run `Checks: N changed | N unchanged | N withheld | N baseline` line. The Sources
  create form's noise field (Task 8, this commit) states `change_threshold`'s role directly in its
  help copy, verified rendering untruncated at both 160x42 and 235x52.

**Close-out sweep (Task 8).** `Tests/Subscriptions/ Tests/Scheduling/ Tests/Watchlists/` (full
directories): 557 passed. `Tests/DB/ -k subscription`: 47 passed. `Tests/UI/ -k watchlist`: 2
tree-chevron failures (pre-existing, unrelated to this task) plus the intermittent TASK-1345 race
described above; no other failures on either of two runs. Filed `TASK-1383` for the scheduled-check
path, which this task's Task 3 review found records no run row at all (`Scheduling/scheduler/
handlers/watchlist_check_handler.py` sinks only into a fixed-column daily aggregate nothing else
reads; the Runs pane reads exclusively from `local_watchlist_runs`, populated only by the manual
`launch_run` path) -- so the §4 dispositions this task built cannot reach scheduled checks by any
wiring that exists today.

**Files (Task 8, beyond Tasks 1-7 already committed):**
`tldw_chatbook/UI/Watchlists_Modules/sources_pane.py`,
`Docs/superpowers/specs/2026-07-29-watchlists-noise-not-volume-design.md`,
`backlog/tasks/task-1383 - Scheduled-watchlist-checks-never-create-a-run-row-so-the-Runs-pane-cannot-see-them.md`,
`backlog/tasks/task-1362 - A-small-edit-to-a-long-page-never-clears-the-change-threshold.md` (this file).
<!-- SECTION:NOTES:END -->
