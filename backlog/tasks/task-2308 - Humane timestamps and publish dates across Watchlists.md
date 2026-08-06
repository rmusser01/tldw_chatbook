---
id: TASK-2308
title: Humane timestamps and publish dates across Watchlists
status: In Progress
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - ux
  - uat-2026-08-04
dependencies: []
priority: medium
---

## Description (the why)

UAT: Sources ("Last scraped"), Items ("Created") and Runs ("Started") show
raw ISO-8601 timestamps with microseconds in UTC — 32 characters of machine
format dominating row width, in the wrong timezone for the user. The Items
column additionally shows INGEST time (identical to the microsecond on every
row) when users need the item's PUBLISH time, which the reader byline proves
exists. Artifacts already uses a humane format ("2026-08-04 18:22:44"), so a
house style exists and is applied inconsistently.

UAT findings F20, F24, F41.

## Acceptance Criteria (the what)

- [ ] All Watchlists tables show local-time, human-scale timestamps (short
      format or relative), consistent with the Artifacts style.
- [ ] The Items list shows the item's publish date where available, falling
      back honestly when the feed omits it.
- [ ] Column widths return to proportionate sizes.

## Implementation Plan (the how)

1. Look for an existing humane formatter first. (Done: there is none shared --
   Artifacts' "2026-08-04 18:22:44" is simply the raw SQLite `CURRENT_TIMESTAMP`
   string, i.e. UTC that merely *looks* humane. So the house style has to be
   written down, once.)
2. New `UI/Watchlists_Modules/humane_time.py`: parse ISO-8601 (with or without
   microseconds, `Z` or offset, SQLite's space separator), treat a naive value
   as UTC (which is what every writer on this screen stores), convert to the
   system local zone, and render `Today HH:MM` / `Yesterday HH:MM` /
   `Mon DD HH:MM` / `YYYY-MM-DD`. Unparseable input passes through unchanged;
   empty input renders the column's dash.
3. Apply it at every Watchlists table timestamp: Sources "Last scraped", Runs
   "Started" (row + detail block), Notifications "Created", Artifacts
   briefings/scripts "Created" and both detail headers.
4. Items: the column becomes "Published" and reads `published_date`. When the
   feed omits one, the cell says `added <time>` -- the ingest time, explicitly
   labelled as such -- so it never shows ingest time under a publish heading.
5. Correct the one existing test that asserts the raw ISO string as a premise.
