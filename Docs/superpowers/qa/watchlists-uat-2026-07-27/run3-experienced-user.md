# Watchlists UAT run 3 — experienced-user journey, 2026-07-28

`origin/dev` `e82ac1b18`, clean profile (`uat_wl_r3`, deleted after), 235x52,
click columns computed by character.

Third attempt. Runs 1 and 2 each stopped at a blocking defect before reaching
this journey; both are fixed, so this run finally walked it.

## Walked successfully

| Step | Result |
|---|---|
| Create two watchlists | Both appear; scope follows the new one |
| Create three sources through the form | All three created; table and Feeds both update |
| Switch scope to a watchlist | `Feeds in Morning AI Brief (0)`, staging line narrows with it |
| Add a source to a watchlist | `Feeds in Morning AI Brief (1)`, source listed |
| Rename a watchlist | Applied; dialog pre-fills the current name and selects it |
| Delete a watchlist holding a source | **Orphan prevention holds** — scope moved to `Feeds in Unassigned (3)` and all three sources survived |

The delete confirmation explains the consequence properly before acting: it
names the watchlist, says its sources are not deleted, and says where they go.

`task-895`'s AC#3 — deleting a watchlist must never orphan a source into
invisibility — is now verified in the running app, not just in tests.

## Found

- **task-1300 (high)** — `Escape` does not dismiss the dialogs; only clicking
  `Cancel` does. A keyboard user cannot back out, and because the dialog is
  modal the app appears to ignore every subsequent click. This bit the UAT
  itself: a `Delete` click was silently swallowed by a still-open Rename dialog.
- **task-1091 (medium)** — names are not trimmed, so `" Daily"` is stored with
  its leading space and the tree row is visibly mis-indented; and the delete
  confirmation reads "Its **1 source are** not deleted".

## A correction to run 2

Run 2 filed, and shipped, the claim that the rail and the centre disagree —
`All sources  0` beside `Feeds in All sources (1)`. **That was wrong.**
`get_watchlist_item_counts` returns *item* counts, not source counts. With
sources added but nothing scraped, `0` is correct. `task-1040` has been
corrected on the board; the fix it shipped is still right for the sources
table, but its AC #6 and #7 assert an equivalence that does not exist.

## Still not covered

- **Tag filtering** — no tags were set on any source, so the rail's tag filters
  were never exercised.
- **The Feeds → Items → Content reading path** — nothing has been scraped, and
  the sources point at `example.com` placeholders. Exercising it needs a real
  feed or seeded items, and it is the substance of Phase D.
