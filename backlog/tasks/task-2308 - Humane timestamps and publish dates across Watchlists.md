---
id: TASK-2308
title: Humane timestamps and publish dates across Watchlists
status: To Do
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
