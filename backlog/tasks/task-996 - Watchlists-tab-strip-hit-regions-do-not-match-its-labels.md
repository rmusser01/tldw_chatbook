---
id: TASK-996
title: >-
  Watchlists tab strip activates the wrong section for a given click column
status: Won't Do
assignee: []
created_date: '2026-07-27 22:00'
labels:
  - watchlists
  - bug
  - ui
  - uat
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clicking the column where `Items` is drawn in the section tab strip activates `Runs` instead. Reproduced repeatedly at 235x52 on `origin/dev` `dbbb7de84`: computing the click column from `index($0, "Items")` and from `index($0, "    Items")+6` both landed on `Runs`, and `Items` was never reached by mouse.

So the tabs are mislabelled from the user's point of view — the thing you click is not the thing you get.

Very likely the same root as task-875's fix: `WatchlistsTabStrip` pins itself to `height: 1` while its `Button`s want three rows, so their layout boxes — and therefore their hit regions — do not line up with where their labels are painted. The label fix in `features/_watchlists.tcss` made the text visible; it may not have corrected the geometry.

The dialog `Create` button showing the same symptom (see `Docs/superpowers/qa/watchlists-uat-2026-07-27/notes.md`) suggests checking whether this is broader than the tab strip.

Evidence: `Docs/superpowers/qa/watchlists-uat-2026-07-27/notes.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Clicking any tab activates the section whose label is under the pointer
- [ ] #2 A test drives a real click at each tab's rendered label column and asserts the resulting `active_section`, proven to fail against current code
- [x] #3 The active tab stays visually distinguishable and the strip stays one row
- [x] #4 It is established whether the dialog `Create` button shares this cause, and the answer recorded here
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
**The defect does not reproduce. No application code was changed.** The tab
strip's hit regions match its painted labels exactly, and always did.

**Verified in the live app**, 235x52, tmux, SGR mouse clicks, on a scratch
profile — the same method the UAT used. Clicking the centre of each painted
label activates that label's own section, every time, including the
`Items` → `Runs` case that was reported as never reachable:

    click Overview       1-based col  37 -> active tab = Overview
    click Sources        1-based col  48 -> active tab = Sources
    click Items          1-based col  58 -> active tab = Items
    click Runs           1-based col  69 -> active tab = Runs
    click Rules          1-based col  78 -> active tab = Rules
    click Notifications  1-based col  92 -> active tab = Notifications
    click Items          1-based col  58 -> active tab = Items
    click Runs           1-based col  69 -> active tab = Runs

(Active tab read from `capture-pane -e` as the `\x1b[1;4m` bold-underline
segment, since plain `capture-pane` discards the only cue the strip has.)

**Verified in the harness** at both pinned viewports under the production
stylesheet: every column each label occupies — not just its centre — resolves
through `Screen.get_widget_at` to that tab's own button.

    overview       region=Region(x=30, width=12)  label painted at cols 32..39
    sources        region=Region(x=42, width=11)  label painted at cols 44..50
    items          region=Region(x=53, width=10)  label painted at cols 55..59
    runs           region=Region(x=63, width=10)  label painted at cols 66..69
    rules          region=Region(x=73, width=10)  label painted at cols 75..79
    notifications  region=Region(x=83, width=17)  label painted at cols 85..97

**Why the task-875 suspicion was wrong.** That defect was a *round border*
inside a one-row box eating the content line — a vertical clip. It cannot
move a label horizontally, and `.watchlists-tab-strip Button.watchlists-tab`
already carries `border: none` + `height: 1`, so the strip's buttons are
genuinely one row and their labels sit inside their own boxes. In Textual a
widget cannot paint outside its own region at all; the compositor clips it.
The reported symptom is therefore a UAT harness coordinate error, not an
app defect. The UAT's own note on the `Create` button ("possibly harness
coordinate error", recorded as needs-verification) was the right instinct.

**AC#4 — the dialog `Create` button does NOT share this cause; it works.**
Live, on the same run: opened the New-watchlist dialog from the rail, typed
a name, clicked `Create` at its painted column (0-based col 107, row 29).
The watchlist was created, the tree gained `Click Test WL  0` and the centre
heading followed to `Feeds in Click Test WL (0)`. Note the UAT recorded its
click at col 228 for a button painted near col 110 — consistent with a
coordinate error rather than a dead button.

**AC#2 is only half met, and the other half is unsatisfiable.**
`test_watchlists_tab_strip_hit_regions_match_its_painted_labels` is new: it
takes each label's column span from the compositor (not from the widget's
own region, which would beg the question), probes every column of it through
`get_widget_at`, then drives a real `pilot.click` there and asserts
`active_section`. It runs at 160x42 and 235x52. It cannot be "proven to fail
against current code" because the current code is correct. In place of that,
its sensitivity was demonstrated by injecting the reported behaviour — an
off-by-one in `WatchlistsTabStrip.on_button_pressed` mapping each tab to its
neighbour — which the test catches:

    AssertionError: clicking column 36 of row 9, where 'Overview' is
    painted, activated 'sources' instead of 'overview'

It was also run against the pre-TASK-995 tree to confirm this branch's
earlier commit did not incidentally fix it: green there too.

Modified: `Tests/UI/test_destination_visual_parity_correction.py` only.
<!-- SECTION:NOTES:END -->

## Resolution: invalid — harness error, not an app defect

**This task should not have been filed, and the fault is in how it was measured.**

The implementer could not reproduce it and said so rather than inventing a fix: verified live over tmux across all six tabs twice, and in-harness at both 160x42 and 235x52, every column of every label routes to its own button. The modal `Create` button works too — a watchlist was created by clicking it.

The controller then reproduced the *cause*. The UAT computed SGR click columns with `awk '{print index($0, LABEL)}'`, and `index()` counts **bytes**. Every line of this app carries three-byte box-drawing and arrow glyphs (`▊`, `▔`, `▼`), so the byte offset outruns the true column by three per glyph. Measured on one row:

```
New Source   char-col=169   byte-offset=181
Filters      char-col=186   byte-offset=198
```

A click computed at 185 as "inside New Source" lands on `Filters`. Clicking at the true character column 174 opened the create-source form correctly. The same arithmetic explains the original report — clicking "Items" reached "Runs" because the tab row's glyphs had pushed the computed column into the next button.

AC #2 asked for a test proven to fail against current code. That is unsatisfiable for a defect that does not exist, so it is left **unchecked**. The implementer instead proved the new tab-routing test's sensitivity by injecting an off-by-one into the tab handler — which is the right substitute.

Recorded in `backlog/docs/lessons-live-verification.md` so the next UAT does not repeat it.
