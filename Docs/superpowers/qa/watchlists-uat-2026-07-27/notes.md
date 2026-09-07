# Watchlists UAT — 2026-07-27

First UAT of the Watchlists screen since the Phase A/B/C rebuild. Run against
`origin/dev` at `dbbb7de84`, in a **clean profile** (`users_name = "uat_wl_clean"`,
data directory created from scratch and deleted afterwards), 235x52, driven
through tmux with SGR mouse clicks.

Prior QA for this screen was `task-14.6 "Screen QA: Watchlists"`, a screenshot
pass dated **2026-05-09** — before the rebuild — whose evidence archive was
deleted on 2026-06-11 (`6461279e6`). This is the first time the shipped screen
has been walked end to end.

## New-user journey

Land on Watchlists with nothing configured, create a watchlist, add a source.

**It stops at step three.**

1. The empty screen offers two entry points: `New` in the rail (a watchlist) and
   `Create source` in the centre. Reasonable, though it asks the user to grasp
   the watchlist-vs-source distinction immediately.
2. `New` opens a "New watchlist" dialog. Typing a name and pressing Enter works:
   the tree gains `Morning AI Brief  0` and the centre heading follows to
   `Feeds in Morning AI Brief (0)`. **Scope-follows-creation works as designed.**
3. `Create source` switches the centre to Sources — and that toolbar renders with
   **no visible controls at all**:

```
  Sources

  ▊▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
      Preview        Check now      Import OPML     Export OPML
  Name  Type  Status  Last scraped  Active
```

That single `▊▔▔▔` row is the strip holding the search input, the type / status /
active filters, **`New Source`**, and `Filters` (`sources_pane.py:132-158`). Only
its top border renders. A user who has just created a watchlist has no visible
way to add a source to it.

## Cause, isolated during the run

`.destination-filter-strip` is `height: 1` (`css/layout/_panes.tcss:31-36`). A
bordered `Input`/`Select` is three rows, so only the top border survives.

The Rules section, whose strip holds only `Button`s, renders correctly
(`Refresh  New Rule` both visible) — isolating this to strips containing Inputs
or Selects rather than to the strip class as a whole.

This is the **third** occurrence of the one-row-container / three-row-children
pattern here, after `WatchlistsTabStrip` and `LabModeStrip`.

## Other findings

- **Tab strip hit regions do not match label positions.** Clicking the column
  where `Items` is drawn activates `Runs`; repeated attempts never reached
  `Items`. Consistent with three-row tab buttons inside a one-row strip.
- **The watchlist row draws its expand chevron on its own line**, indented, above
  the name rather than beside it:

```
│ Unassigned  0            │
│       ▸                  │
│ Morning AI Brief  0      │
```

- **Overview is seven empty bordered cards** — the largest region on a new user's
  first screen, containing nothing.
- **The Inspector's empty state is a dead end for a new user**: "Select a source,
  run, item, rule, or notification to see actions." when by definition none exist.
- **The dialog's `Create` button did not respond to a click** at its rendered
  coordinates (row 29, col 228); `Enter` submitted correctly. Possibly the same
  hit-region issue as the tab strip, possibly harness coordinate error — recorded
  as needs-verification rather than asserted.

## What worked

Watchlist creation, scope following creation, tree and heading staying in sync,
section switching, the Rules and Runs toolbars, and the region chrome (one border
and one heading each) all behaved correctly.

## Not covered

The experienced-user journey — many sources, tag filtering, scope switching
across watchlists, Console staging, rename/delete — was not reached, because
adding a source is blocked. It should be re-run once the toolbar is fixed.
