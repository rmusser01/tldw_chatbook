---
id: TASK-997
title: >-
  Watchlist tree row draws its expand chevron on its own line above the name
status: Done
assignee: []
created_date: '2026-07-27 22:00'
labels:
  - watchlists
  - bug
  - ui
  - uat
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A watchlist row in the tree renders its expand chevron on a separate, indented line above the name instead of beside it. Captured live at 235x52 on `origin/dev` `dbbb7de84` after creating one watchlist from a clean profile:

```
│ Unassigned  0            │
│       ▸                  │
│ Morning AI Brief  0      │
```

Expected: `▸ Morning AI Brief  0` on one row.

It costs a rail row per watchlist and reads as a stray glyph, which matters more as the tree fills up — the rail is 26 columns and the tree is the screen's primary navigation.

Evidence: `Docs/superpowers/qa/watchlists-uat-2026-07-27/notes.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The chevron and the watchlist name render on the same row
- [x] #2 One watchlist occupies one row in the collapsed state
- [x] #3 A test asserts the rendered row text against the production stylesheet, proven to fail against current code
- [x] #4 Expanding still shows the watchlist's sources indented beneath it
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Reproduced live before fixing (235x52, scratch profile, real app): creating a
watchlist produced exactly the filed rendering, chevron alone on an indented
row above the name.

**Two causes, both in one place.** `_watchlist_node` yielded the chevron
`Button` and the name `Button` as two separate children of the tree's
`Vertical`, so they stacked; and the chevron carried no width rule, so it
inherited Textual's `Button { min-width: 16 }` — `compact=True` removes the
border, not that floor — and was sixteen columns wide with the glyph centred
in them. That is why it painted seven columns in from the left edge of a
26-column rail and read as a stray mark rather than as a control.

The two buttons are now wrapped in a `.watchlist-tree-row` `Horizontal`
(`height: 1`, because `Horizontal` otherwise defaults to `height: 1fr` and
one watchlist would claim the rest of the tree) and the chevron is pinned to
`width: 3; min-width: 0`. Both keep their own ids and `Button.Pressed` still
bubbles to the tree's `on_button_pressed`, so expand, select and scope are
untouched.

Before / after, measured in the rail at 160x42:

    │ Unassigned  0            │      │ Unassigned  0            │
    │       ▸                  │  ->  │ ▸  Morning AI Brief  0   │
    │ Morning AI Brief  0      │      │ ▸  Security Digest  0    │

Expanded, sources still nest:

    │ ▾  Morning AI Brief  0   │
    │      ArXiv               │
    │ ▸  Security Digest  0    │

**Why `width: 3` and not the tighter `2`.** Textual's `Button` reserves a
column each side via `line-pad: 1`, so a 2-column button has zero content
columns and Rich raises `ValueError: range() arg 3 must not be zero` while
wrapping the glyph. The obvious `line-pad: 0` cannot be written in a
stylesheet at all: Textual's integer property parser rejects a literal `0`
for it (`_process_integer` errors on `value == 0`), so the sheet fails to
parse and every rule after it is silently lost. Both were hit and are
recorded in the CSS comment so the next person does not repeat them.

The source rows deliberately got no rule. Their two-space label indent plus
the existing `min-width: 16` centring already places them past the watchlist
name (name at column 5, source at column 7), so nesting reads without a
margin that would have to track the chevron's width.

**Test.** `test_watchlists_tree_chevron_shares_a_row_with_its_watchlist`, at
160x42 and 235x52 under the production stylesheet, asserts same-row geometry,
chevron-before-name ordering, the painted row text, the absence of a stray
chevron on the row above, and — after a real expand click — that the source
sits below and is painted further right than its watchlist. Proven red first:
"the chevron is on row 13 and its watchlist name on row 14".

Modified: `tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py`,
`tldw_chatbook/css/features/_watchlists.tcss`,
`tldw_chatbook/css/tldw_cli_modular.tcss` (generated),
`Tests/UI/test_destination_visual_parity_correction.py`.
<!-- SECTION:NOTES:END -->
