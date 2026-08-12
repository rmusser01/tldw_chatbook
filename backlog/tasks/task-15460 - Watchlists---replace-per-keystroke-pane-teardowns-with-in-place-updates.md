---
id: TASK-15460
title: Watchlists: replace per-keystroke pane teardowns with in-place updates
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
labels:
  - perf
  - watchlists
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: `UI/Watchlists_Modules/article_list.py:188` declares `search_query = reactive("", recompose=True)` set directly in `on_input_changed` (`:355`) — every character typed tears down and rebuilds ~220 widgets (rows + day headers + toolbar), with a `recompose()` override (`:371`) existing solely to re-focus the destroyed search box; `status_filter`/`runtime_backend` share the blast radius. Same family: `items_pane.py:79/:295` (whole DataTable; its own docstring at `:311-314` admits the teardown-per-keystroke) and `sources_pane.py:131-135/:792` (toolbar + 8-Input create form + table). The downstream DB fetch is already debounced 0.3 s — only the recompose is per-keystroke.

Fix direction: plain reactives + in-place row repaint (surgical helpers already exist, e.g. `article_list._repaint_row`); filter via display toggles or diffing. Stability constraint: replace the re-focus hack with real focus preservation and pin it — focus must never leave the search box while typing. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Typing in Watchlists search/filter boxes causes no pane recompose (evidence)
- [x] #2 Filtering results identical, including the debounced DB reload path (tests)
- [x] #3 Focus stays in the input while typing (regression test replacing the re-focus override)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. PIN FIRST: characterisation tests for the current filtering behaviour of all
   three panes (typed query -> visible rows; status/type/active filters; tags
   filter; the open-item pin), asserting RENDERED results, green before any
   production change.
2. Born-red evidence tests: count pane recomposes per keystroke (expect 0), and
   assert the search `Input` widget identity + caret position survive a burst of
   keystrokes.
3. `ArticleListPane`: `items`/`search_query`/`status_filter`/`runtime_backend`
   become plain reactives. Rows for the whole item page are mounted once per
   data arrival (`watch_items` -> in-place ListView rebuild, toolbar untouched);
   filtering is a pure display/disabled toggle over the mounted rows plus the
   day headers. Delete the `recompose()` re-focus override -- with the toolbar
   never destroyed there is nothing to re-focus.
4. `ItemsPane`/`SourcesPane`: filter reactives become plain; a `DataTable`
   holds data rows, not widgets, so filtering re-populates the table in place
   (`clear()` + `add_row`) with zero widget mounts and no toolbar teardown.
   `SourcesPane.recompose()` keeps its create-form focus logic (still driven by
   `show_create_form`/`create_draft_source_type`).
5. Latency probe on a seeded 100-article pane, before vs after, isolated HOME.
6. Run the Watchlists UI suites + the panes' own suites; read pass counts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The three filtering panes no longer tear themselves down per keystroke.

**`ArticleListPane`** (the pane actually mounted in the Read section) now has
no `recompose=True` reactive at all. `_build_rows` mounts one `_ArticleRow`
for every item on the loaded page -- not just the ones the filter admits --
so filtering is a `display`/`disabled` toggle over rows that already exist
(`_apply_row_visibility`), including hiding a day header once its whole group
is hidden. `disabled` deliberately tracks `display`: ListView's cursor
movement skips disabled children and knows nothing about `display`, so a
hidden-but-enabled row would silently take the cursor and `j`/`k`. The
data-arrival path (`watch_items`) rebuilds the ListView's children in place,
touching nothing in the toolbar -- which matters because the screen's reload
is debounced 0.3 s behind the last keystroke and used to destroy the search
box the user was still typing into. With nothing destroying the input, the
`recompose()`/`_restore_search_focus` pair (TASK-3071) was deleted rather
than kept: the caret now stays where the user put it instead of being
restored to the end of the value. The empty state and the ListView are both
always mounted with `display` toggled, so "nothing matches" is not itself a
teardown.

**`ItemsPane`/`SourcesPane`** are `DataTable` panes, and a `DataTable`'s rows
are data rather than widgets -- so their filters became plain reactives that
re-populate the table in place (`_refresh_table_rows` -> `_populate_table`,
shared with `compose()`), which constructs no widget and leaves the toolbar,
an open create form, the focused `Input` and its caret alone. Their `items`/
`sources` reactives keep `recompose=True` (a data arrival, never triggered by
typing in `SourcesPane`, whose search is client-side only), and
`SourcesPane.recompose()` keeps its create-form focus logic, which is still
driven by `show_create_form`/`create_draft_source_type`. `runtime_backend`
was `recompose=True` on both panes while being read by neither `compose()` --
a free rebuild, now plain.

Measured on a seeded 100-article pane, isolated HOME, same probe before and
after (mean of 8 keystrokes, `pilot.press` + settle):

| | before | after |
|---|---|---|
| ms/keystroke | 312.7 | 142 |
| pane recomposes | 1.00/key | 0 |
| `_ArticleRow` constructions | 100/key | 0 |
| search `Input` replaced | every key | never |

The harness floor (same probe, 0 articles, after) is 122.5 ms, so the pane's
own per-keystroke work went from ~190 ms to ~20 ms. A narrowing query
(`"article 1"`, 100 -> 11 rows) measures 147-157 ms, also with zero
recomposes and zero row constructions. The no-op guard in `set_row_visible`
is load-bearing for that: a `display` write is a styles mutation and a
refresh even when the value is unchanged, and it runs once per row per
character.

Tests: `Tests/Watchlists/test_watchlists_pane_filter_in_place.py` -- ten
characterisation tests asserting RENDERED rows (they passed before the change
and still do) plus six evidence tests that were red before it (recompose
count per keystroke, `Input` identity, focus, and caret position including a
mid-string edit).

Modified: `tldw_chatbook/UI/Watchlists_Modules/article_list.py`,
`items_pane.py`, `sources_pane.py`. Added:
`Tests/Watchlists/test_watchlists_pane_filter_in_place.py`.
<!-- SECTION:NOTES:END -->
