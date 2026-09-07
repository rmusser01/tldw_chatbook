---
id: TASK-1035
title: >-
  The create-source form cannot be filled in — no field accepts typed text
status: Done
assignee: []
created_date: '2026-07-28 01:30'
updated_date: '2026-07-28 03:10'
labels:
  - watchlists
  - bug
  - ui
  - uat
priority: critical
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Opening the create-source form and typing puts text nowhere. Observed in the experienced-user UAT on `origin/dev` `13551ae8a`, clean profile, 235x52.

Steps, all of which failed to enter a single character into `Name` or `URL`:

1. Sources → `New Source` → type immediately. Nothing lands. **The form does not focus its first field**, which alone is a defect — the New-watchlist dialog *does* auto-focus, so the two creation flows behave differently.
2. Click directly on the `Name` input's own row (row 26, char column 40 — computed by character, not byte). Type. Nothing lands.
3. Press `Tab` five times, then type. Nothing lands.

**Control, run immediately afterwards in the same session:** clicking `Notifications` in the tab strip switched the section correctly. The app was alive and processing clicks throughout, so this is not a hung UI or a dead harness.

If this reproduces at the code level, **source creation through the form is unusable** and the only remaining path into the product is OPML import — which would mean the new-user path found in the previous UAT (task-995) is still blocked, just one step further along.

Recorded as observed-through-the-harness rather than proven in code. The first thing to establish is whether the form's `Input` widgets can take focus at all — check what `Screen.focused` is after the form mounts, and whether anything upstream is holding focus or the inputs are mounted outside the focus chain.

## Also seen on this form

`Create` and `Cancel` render stacked vertically with blank rows between them, unlike the New-watchlist dialog where they sit side by side. Cosmetic, but it makes the form look unfinished next to the dialog it is paired with.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Whether the form can be filled at all is established in code, and the answer recorded here
- [x] #2 If broken: a source can be created end to end through the form, verified live from a clean profile
- [x] #3 The form focuses its first field on open, matching the New-watchlist dialog
- [x] #4 `Tab` moves through the form's fields in visual order
- [x] #5 A test drives the form the way a user does — focus, type, submit — and fails against current code
- [x] #6 `Create` and `Cancel` are laid out consistently with the New-watchlist dialog
<!-- AC:END -->

## Implementation Notes

**Verdict: REAL, and narrower than reported.** Two of the three reported
symptoms reproduce in code and share one root cause. The third does not
reproduce and was a harness artifact.

### What was actually broken (AC#1)

`show_create_form` is `reactive(..., recompose=True)`, and `Widget.recompose`
removes and remounts *every* child of `SourcesPane`. Pressing `New Source`
therefore destroys the `Button` that was holding focus, and Textual does not
re-home focus when the focused widget is removed that way. Measured in the
full shell under the production stylesheet at 235x52:

    screen.focused = None            # immediately after the form opens
    tabs from cold to #sources-create-name = 37

So symptom 1 (open and type) and symptom 3 (open, `Tab` x5, type) are both
real: the form opened with nothing focused anywhere on the screen, and five
tabs from `None` restart at the head of the screen's focus chain and land on
`nav-personas`, nowhere near the form.

**Symptom 2 — clicking the `Name` input — does not reproduce.** A bordered
`Input` is three rows and only the middle one is content, so a border-row
click was the obvious suspicion. It is not the explanation: `get_widget_at`
returns the `Input` for all three of its rows and Textual focuses on
mouse-down, so a click at *any* row of the field focuses it and typing lands.
Pinned by `test_clicking_any_row_of_the_name_input_focuses_it`, which passes
both before and after this change, so the next UAT does not re-file it.

### Two more defects found on the same form while measuring

* The `Grid` had no CSS rule anywhere, so it took Textual's defaults — one
  column, rows sharing the container's height as `1fr` each. That spread
  seven controls over 23 rows with blank gaps (`Create` on row 40, `Cancel`
  on row 43 — the "stacked with blank rows" in the report) and starved
  `#sources-table` to a single row.
* The `Active` label was `width: 1fr` and claimed the whole 168-column row,
  pushing the `Switch` to `x=198` — one column past the pane's right edge at
  235x52. The only control for "is this source active" was neither visible
  nor clickable.

### The fix

* `SourcesPane.recompose()` re-homes focus into the create form after the
  rebuild: the first field when the form has just opened (matching
  `WatchlistNameDialog.on_mount`), or the field that had focus when some
  *other* recompose rebuilt the pane underneath the user. Focus is never
  taken from outside this pane's own create form.
* The form is a `Vertical` with `height: auto`, not a bare `Grid`; Type and
  the Active switch share one row; `Create`/`Cancel` sit in one
  `.dialog-buttons` row, the same class `WatchlistNameDialog` uses.
* Height matters as much as the spread: the Sources pane is 16 rows at
  160x42 and its toolbar takes two, so five full-height rows would put
  `Create`/`Cancel` below the fold — as unreachable as one that was never
  focusable. The form is 13 rows and every control is inside the pane at
  both 160x42 and 235x52.

### Verification

`Tests/UI/test_watchlists_source_create_form.py`, 7 tests x 2 sizes, all
against the production stylesheet in the full shell: 10 of 14 fail on the
pre-fix code (`assert '' == 'Morning'`, "nothing focused", "Create and Cancel
are on different rows"), 14 pass after. Regenerated the CSS bundle with
`build_css.py`. `Tests/Watchlists` (167), `test_destination_visual_parity_correction`
(116), `test_destination_shells` (103), `test_watchlists_inspector` (16) and
`test_watchlists_destination_shell` (48) all green.

Live from a clean throwaway profile at 235x52 (deleted afterwards): Sources →
`New Source` → typed `Morning AI Brief` with no click at all → `Tab` → typed
the URL → clicked `Create`. The Feeds region went to `Feeds in All sources
(1)` showing `Morning AI Brief  (rss)`.

One test needed updating rather than fixing:
`test_bracket_toggle_preserves_in_progress_create_form_draft` drove a
workbench rebuild with `pilot.press("[")` while the create form was open.
That only ever worked because nothing had focus. Now that the form focuses
its `Name` field, the bracket is typed into it (`'Draft Name['`) instead of
reaching the screen binding — correct behaviour for a focused Textual
`Input`, and what the Sources search box has always done. The test now moves
focus to a `Button` first.

### Found, not fixed — needs its own task

`_create_source` refreshes the local snapshot and the overview after a write
but never calls `_load_sources()`, so `#sources-table` keeps showing the
pre-create list until the section is re-entered. Confirmed live: after
creating, the Feeds region and the "Source created." toast both updated but
the Sources table stayed empty; leaving and re-entering Sources showed the
row. Out of scope here (creation itself works end to end and the user does
get feedback), but the next UAT will hit it.

### Modified files

* `tldw_chatbook/UI/Watchlists_Modules/sources_pane.py`
* `tldw_chatbook/css/features/_watchlists.tcss` (+ regenerated
  `tldw_chatbook/css/tldw_cli_modular.tcss`)
* `Tests/UI/test_watchlists_source_create_form.py` (new)
* `Tests/UI/test_watchlists_destination_shell.py`
