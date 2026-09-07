---
id: TASK-1347
title: Watchlists Overview first-run tests assert a container, not its copy
status: Done
assignee: []
created_date: '2026-07-29 05:30'
labels:
  - watchlists
  - testing
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by mutation during Phase D Task 7: blanking the Overview pane's first-run title leaves **all
four** of its tests green. They assert that a container exists, not that it says anything, so the
first-run guidance could be emptied and CI would not notice.

The first-run affordance is what a brand-new user sees when they have no watchlists — the one
screen state where copy is the entire feature. This is the same shape as the ten-plus
green-for-the-wrong-reason tests found across the Phase D and TASK-1240 branches.

Also in this area: `Tests/UI/test_watchlists_content_pane.py`'s `_render_to_console` helper prints
the rendered article to stdout during the run (`console.print(renderable)` with `record=True`),
which is cosmetic noise in test output.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Overview first-run tests assert the actual guidance copy, and blanking that copy makes at least one of them fail
- [x] #2 The same check is applied to the tree's first-run affordance
- [x] #3 _render_to_console no longer writes to stdout during a normal test run
<!-- AC:END -->

## Implementation Plan

1. Locate the Overview pane's first-run copy (`OverviewPane._first_run_body`, two variants
   keyed on `watchlist_count`) and every test that mounts the pane in its EMPTY state, by
   mutating `_first_run_body` to `return ""` and running the suite to see which tests still
   pass (mutation-first, so the "four tests" are found empirically, not guessed).
2. Strengthen each test that asserts *presence* of `#overview-first-run` to also assert a
   distinctive substring of the real copy, read from the rendered widget the same way the
   file's other content assertions do (`str(widget.renderable)` / painted compositor text) --
   never by calling `_first_run_body()` directly.
3. Add a same-file unit test exercising the `watchlist_count > 0` variant directly, since
   nothing exercised it before.
4. Repeat the same mutate-and-look process for the tree/inspector's first-run hint
   (`InspectorPane`, `#inspector-first-run-hint`) and strengthen its one covering test.
5. Fix `_render_to_console` in `Tests/UI/test_watchlists_content_pane.py` to render into an
   `io.StringIO()` instead of real stdout, keeping `record=True` and the `(plain, ansi)`
   return contract unchanged.
6. Re-run the mutations to confirm each now reds, restore the source byte-exact, and run the
   full touched-file set plus `--collect-only` on `Tests/UI` and `Tests/Watchlists`.

## Implementation Notes

Strengthened the tests around both Watchlists first-run affordances so blanking the guidance
copy actually fails a test, and quieted a stdout-printing test helper. No production code
changed; `overview_pane.py` and `inspector_pane.py` were only touched transiently during the
required mutation checks and are back byte-exact (`git diff` on both is empty).

**AC#1 (Overview, `_first_run_body`).** Mutating `_first_run_body` to `return ""` and running
the suite showed exactly four tests interact with `#overview-first-run`, matching the task's "all
four" framing:
- `test_watchlists_first_run_replaces_empty_cards_with_guidance` and
  `test_the_overview_shows_a_loading_state_while_the_request_is_in_flight` assert *presence* of
  first-run copy -- both only checked the container (or a weak `"watchlist" in painted.lower()`,
  which happened to survive the blank-title mutation the task description names because the word
  "Watchlists" appears elsewhere on screen). Both now assert the distinctive substring
  `"a watchlist is a folder of feeds"` against the rendered pane (painted compositor text /
  `#overview-first-run-body.renderable` respectively).
- `test_watchlists_populated_overview_and_inspector_are_unchanged` and
  `test_a_user_with_sources_never_flashes_first_run_copy` assert *absence* of first-run copy on a
  populated profile; blanking the body text has no effect on them by construction, so they were
  left as-is -- narrowing honestly rather than padding them with an assertion that can't fail on
  this mutation.
- Added two new tests in `Tests/Watchlists/test_watchlists_overview_pane.py` that mount
  `OverviewPane` directly (matching that file's existing style) and assert on
  `#overview-first-run-body`'s rendered text for **both** variants
  (`watchlist_count == 0` -> "A watchlist is a folder of feeds...",
  `watchlist_count > 0` -> "Your watchlists have no sources yet...") -- the second variant had
  no coverage anywhere before this, so a regression confined to that branch would have passed
  every existing test even after strengthening the other four.
- Mutation re-verified: blanking `_first_run_body` reds exactly **4** tests total -- the two
  original first-run tests that assert the copy is PRESENT (now strengthened) plus the two new
  variant tests. The other two of the original four assert the first-run copy is ABSENT (loaded
  non-empty / still loading) and correctly stay green -- a copy check there would be vacuous, so
  they were left untouched. Reverting restores a byte-exact `overview_pane.py`.

**AC#2 (tree/inspector, `#inspector-first-run-hint`).** Only one test exercises this affordance,
`test_the_inspector_follows_the_same_three_states`
(`Tests/UI/test_watchlists_overview_loading_state.py`); it previously asserted only
`inspector.query("#inspector-first-run-hint")`. This copy has one variant (no `watchlist_count`
branch on the Inspector side), so it now also asserts the distinctive substring
`"start with new in the rail, then new source under sources"` against the hint's rendered text.
Mutation re-verified: blanking the `Static`'s text in `inspector_pane.py` reds this test;
reverting restores a byte-exact `inspector_pane.py`.

**AC#3 (`_render_to_console` stdout leak).** Added `file=io.StringIO()` to the `Console(...)`
constructor in `Tests/UI/test_watchlists_content_pane.py`; `record=True` still captures
everything printed to that buffer so `export_text()` -- and therefore the `(plain, ansi)` return
values every call site depends on -- is unaffected. Verified with `pytest -s -k
test_article_renders_title_source_and_body`: no rendered-article text reaches real stdout
(pytest's own logging/config chatter still does, which is unrelated and out of scope). Sibling
files `Tests/Watchlists/test_watchlists_artifacts_pane.py` and
`Tests/Watchlists/test_kept_briefings_modal.py` define their own separate local copies of
`_render_to_console` with the same `force_terminal=True`/no-`file=` shape; the task names only
`Tests/UI/test_watchlists_content_pane.py`, so those were left untouched.

**Modified files:**
- `Tests/UI/test_destination_visual_parity_correction.py` -- distinctive-copy assertion.
- `Tests/UI/test_watchlists_overview_loading_state.py` -- distinctive-copy assertions (Overview
  and Inspector).
- `Tests/Watchlists/test_watchlists_overview_pane.py` -- two new tests covering both
  `_first_run_body` variants.
- `Tests/UI/test_watchlists_content_pane.py` -- `_render_to_console` no longer writes to stdout.

**Verification:** `Tests/UI/test_watchlists_overview_loading_state.py` (4 passed),
`Tests/Watchlists/test_watchlists_overview_pane.py` (5 passed),
`Tests/UI/test_watchlists_content_pane.py` (40 passed),
`Tests/UI/test_watchlists_inspector.py` (35 passed),
`Tests/UI/test_destination_visual_parity_correction.py` first-run subset (2 passed); combined run
of all four touched/adjacent files: 84 passed. `pytest --collect-only Tests/UI Tests/Watchlists`:
8238 tests collected, no errors.
