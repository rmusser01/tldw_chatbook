---
id: TASK-998
title: >-
  Watchlists first run shows seven empty cards and dead-end Inspector guidance
status: Done
assignee: []
created_date: '2026-07-27 22:00'
labels:
  - watchlists
  - ux
  - uat
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two first-run problems seen together in the clean-profile UAT (`origin/dev` `dbbb7de84`, 235x52).

**The Overview region is seven empty bordered cards.** It is the largest region on the screen and the first thing a new user sees, and it contains nothing. This was recorded during the original design work as one of the screen's defects and has never been addressed.

**The Inspector's empty state is a dead end.** It reads "Select a source, run, item, rule, or notification to see actions." — but on first run none of those exist, so the guidance names five things the user cannot do. The right-hand rail is a third of the screen and spends it telling a new user to do something impossible.

Together these mean a first-time user's screen is mostly empty boxes and instructions that do not apply. The tree's `New` and the centre's `Create source` are the only real affordances, and neither is where the eye goes.

Worth treating as one piece of work: it is the same question — what should this screen say when there is nothing in it yet.

Evidence: `Docs/superpowers/qa/watchlists-uat-2026-07-27/notes.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Overview region shows real content, or is not shown, on a profile with no watchlists and no sources
- [x] #2 The Inspector's first-run text points at an action the user can actually take
- [x] #3 A first-run capture from a clean profile is attached, showing no empty bordered cards
- [x] #4 The populated states of both regions are unchanged
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
**The judgement made.** When Watchlists is empty, the screen should say what it
is for and name the two controls that do something, and it should stop
offering chrome around data that does not exist. When it has anything in it,
nothing changes. Both the Overview's seven cards and the Inspector's five-noun
guidance are answers to that one question, so both are branched on the same
predicate rather than softened into one string that fits neither state.

**Overview.** On an empty profile the card grid and the failed-runs table are
replaced — not blanked — by a titled panel. The copy has two variants, because
the UAT's journey ended exactly between them: telling a user who has just
created a watchlist to "create a watchlist" is the same dead end one step
along. `OverviewPane.watchlist_count` (screen-seeded, like every other
reactive on these panes) picks between them. Captured live, clean profile,
235x52:

    Nothing is being watched yet.

    A watchlist is a folder of feeds. Watchlists checks them on a schedule
    and collects whatever is new.

    1. Press New in the rail on the left to create a watchlist.
    2. Open Sources above and press New Source to add a feed to it, or
       Import OPML to bring a set of feeds over from another reader.

    Runs, items, rules and notifications fill in once a source has been
    checked.

and after creating a watchlist, on the same run:

    Your watchlists have no sources yet. Open Sources above and press New
    Source to add a feed, or Import OPML to bring a set of feeds over from
    another reader.

**Inspector.** "Select a source, run, item, rule, or notification" is correct
once those things can exist and a dead end before then, so it is kept verbatim
for the populated case and replaced on first run with "Nothing to inspect yet."
plus a line naming `New` and `New Source`. Captured live in the 34-column
rail, wrapping correctly.

Both are guarded on `total_sources` being PRESENT in `overview_data`, not
merely falsy. That dict is `{}` until a worker fills it, so a plain zero-check
would flash the first-run panel for a tick on every visit — including for
users with hundreds of sources.

**A second, larger defect found while doing this, and fixed.** The UAT read
"seven empty bordered cards" as a first-run problem. It is not: the cards were
empty in EVERY state and had been since the dashboard shipped. Measured at
160x42 with a source present:

    #overview-total-sources  region=Region(height=1)  content=Size(height=0)

`#watchlists-overview-grid` had no `height`, so it took `Grid`'s `1fr` default
and got six rows for three rows of cards plus two gutters; and `padding: 1`
inside `height: 4` with a `round` border leaves 4 - 2 - 2 = 0 content rows, so
even with its four rows the card could not have painted anything. Fixed with
`height: auto` on the grid and horizontal-only padding on the card. Live
confirmation after the fix: `Total sources / 1`, `Active sources / 1`,
`Total items / 0` all painting.

`Tests/Watchlists/test_watchlists_overview_pane.py` never caught this because
it asserts on `Static.renderable`, which is correct whether or not one cell of
it reaches the screen — the exact reason this branch's tests read the
compositor.

**On AC#4.** The first-run treatment does not leak into a populated profile,
which is what that criterion is guarding, and
`test_watchlists_populated_overview_and_inspector_are_unchanged` pins it. But
the populated Overview *rendering* was deliberately changed, because it was
broken; leaving it would have shipped a screen that works only when empty.

**Live end-to-end.** With TASK-995 and this in place, the path the UAT could
not finish now completes at the real surface: create watchlist -> Sources ->
New Source -> fill name and URL -> Create -> the source appears as
`ArXiv cs.AI  (rss)` in Feeds and the Overview flips to its populated cards.

Modified: `tldw_chatbook/UI/Watchlists_Modules/overview_pane.py`,
`tldw_chatbook/UI/Watchlists_Modules/inspector_pane.py`,
`tldw_chatbook/UI/Screens/watchlists_collections_screen.py`,
`tldw_chatbook/css/features/_watchlists.tcss`,
`tldw_chatbook/css/tldw_cli_modular.tcss` (generated),
`Tests/UI/test_destination_visual_parity_correction.py`.
<!-- SECTION:NOTES:END -->
