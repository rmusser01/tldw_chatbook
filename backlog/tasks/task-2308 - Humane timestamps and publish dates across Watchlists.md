---
id: TASK-2308
title: Humane timestamps and publish dates across Watchlists
status: Done
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

- [x] All Watchlists tables show local-time, human-scale timestamps (short
      format or relative), consistent with the Artifacts style.
- [x] The Items list shows the item's publish date where available, falling
      back honestly when the feed omits it.
- [x] Column widths return to proportionate sizes.

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

## Implementation Notes

Picked up mid-flight after a previous implementer was cut off by a rate
limit. The WIP commit had already landed `humane_time.py` (complete) and
wired it into Sources ("Last scraped"), Runs ("Started", row + detail),
Notifications ("Created") and Artifacts (both tables' "Created" columns plus
both detail headers). `items_pane.py` was barely touched: the column header
had been renamed to "Published" and `compose()` already called
`self.item_published_text(item)`, but that method did not exist anywhere in
the repo -- this session's core job was implementing it.

**AC#1/#3 (house style + column width) -- done, by the WIP for five of six
surfaces and confirmed working here; the sixth (Items) closed this session.**
Live-verified against a real feed check (hnrss.org/frontpage, 20 items):
Sources "Last scraped" reads "Today 01:19", four Runs rows all read "Started:
Today 01:19", and the Items table's twenty rows show a spread of distinct,
short, local-time values ("Today 00:34" through "Yesterday 09:05") -- the
column no longer dominates row width, and titles get the space back. No
separate width CSS was needed: `DataTable` auto-sizes to content, and a
15-character humane string versus a 32-character ISO one is the whole fix.

**AC#2 (publish, not ingest) -- done, this session.** Added
`ItemsPane.item_published_text` (`items_pane.py`): reads `published_date`
first (confirmed the correct field by tracing `get_new_items`'s `SELECT
i.*` through `normalize_watchlist_item`, which is the same field
`content_pane`'s reader byline already reads -- so the two can never
disagree again), and falls back to `f"added {humane_timestamp(created_at)}"`
-- ingest time, but explicitly labelled as ingest, never presented silently
under the "Published" heading -- when the feed supplied no publish date
(`monitoring_engine._parse_date` returns `None` rather than defaulting to
"now" for a feed that omits one). Live-verified: the UAT's own repro (reader
byline vs. table disagreeing) is gone -- opening "What I love about Django"
showed "HN Frontpage · Today 00:34" in the reader, matching the Items table's
"Today 00:34" in the same row, both sourced from the same field through the
same formatter.

**Test-suite fallout from the WIP's own wiring (fixed here, not new
regressions from this session's changes):** two pre-existing tests asserted
the raw ISO string as their premise -- exactly the step 5 this plan calls
for, but for TWO sibling surfaces, not the one originally anticipated.
`Tests/UI/test_watchlists_check_now_failure.py::test_source_row_cells_
render_the_normalizer_status_summary` (Sources) and `Tests/Watchlists/
test_watchlists_runs_pane.py::test_stats_text_without_dispositions_key_is_
unchanged` (Runs) both now assert via `humane_timestamp(...)` rather than a
hardcoded string, since the exact rendering depends on the machine's local
zone and the date the suite runs.

Tests: `Tests/Watchlists/test_humane_time.py` (new, 19 cases, `TZ=UTC`
pinned via `time.tzset()` so the Today/Yesterday/same-year branches are not
machine-dependent) plus 4 new cases in `Tests/Watchlists/
test_watchlists_items_pane.py` for `item_published_text` (publish
preferred, honest ingest fallback, dash when neither exists, wired
end-to-end through the mounted table). Mutation-verified: breaking the
Yesterday-boundary comparison and disabling the publish-date preference
both turned tests red; both reverted with an md5-verified byte-identical
restore.

Modified/added: `tldw_chatbook/UI/Watchlists_Modules/items_pane.py`
(`item_published_text`, import), `Tests/Watchlists/test_humane_time.py`
(new), `Tests/Watchlists/test_watchlists_items_pane.py` (4 tests),
`Tests/UI/test_watchlists_check_now_failure.py` and `Tests/Watchlists/
test_watchlists_runs_pane.py` (test corrections for the WIP's pre-existing
humane-timestamp wiring).
