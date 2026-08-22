---
id: TASK-19871
title: >-
  Watchlists day-header tests read the real clock and fail before 02:00 local
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - testing
  - flaky-test
  - watchlists
priority: medium
dependencies: []
---

## Description

Source: a residual red observed during **TASK-19559**'s work, initially taken
for a code regression. It is a timezone flake: the same commit passes under
`TZ=UTC` and fails at machine-local time. First seen failing at 00:38 PDT.
Re-derived at `3605bd52d`.

The tests stamp their fixture items relative to the **real** clock and then
assert on the day-bucket headers the pane renders:

- `Tests/Watchlists/test_watchlists_pane_filter_in_place.py:40-41` defines
  `_now()` as `datetime.now(timezone.utc)`, and `:52` stamps every item as
  `_now() - timedelta(hours=published_offset_hours)`
- `test_article_search_hides_a_day_header_whose_whole_group_is_filtered_out`
  (`:200`) uses offsets of 1h and 26h, then asserts
  `_visible_headers(pane) == ["Today", "Yesterday"]` (`:211`) and
  `== ["Today"]` (`:216`)

The headers come from `day_bucket()`
(`tldw_chatbook/Subscriptions/item_dates.py:140-160`), which buckets in the
**viewer's local zone** via `astimezone()`. So:

- the `now − 1h` item only buckets as "Today" once the local clock is at or
  past 01:00
- the `now − 26h` item only buckets as "Yesterday" once the local clock is at
  or past 02:00 — before that it lands two days back and renders as a full
  date

Net: the assertions are red for any run between local **00:00 and 01:59**, in
any timezone, and the boundary shifts by an hour across a DST transition. Both
assertions break in that window.

The same defect sits in four sibling tests in
`Tests/Watchlists/test_watchlists_article_list.py`, which copies the same
`_now()` helper (`:37-38`, `:51-52`) with no TZ pin:
`test_rows_group_under_day_headers_in_effective_date_order` (`:222`),
`test_future_dated_item_lands_under_today` (`:248`),
`test_displayed_items_excludes_headers_and_preserves_order` (`:262`) and
`test_headers_are_not_highlightable` (`:276`).

**The fix wanted is an injected clock, not a `TZ` pin.** Both `day_bucket` and
`relative_time` already accept an optional `now=` keyword; neither the
production pane (`UI/Watchlists_Modules/article_list.py:89`, `:465`) nor these
tests passes one. A `TZ=UTC` fixture — as used by
`Tests/Watchlists/test_humane_time.py:29-43` — makes the suite green while
leaving the tests unable to exercise the boundary behaviour that is the actual
subject of `day_bucket`, and leaves the next timezone-sensitive test to
rediscover this.

## Acceptance Criteria

- [ ] The affected tests produce the same result at every local wall-clock
      time and in every timezone
- [ ] The tests control the reference instant explicitly rather than depending
      on when they happen to run
- [ ] A test exercises the day-boundary transitions deliberately — an item just
      before and just after local midnight — which the current tests cannot do
- [ ] All five affected tests are covered (one in
      `test_watchlists_pane_filter_in_place.py`, four in
      `test_watchlists_article_list.py`)
- [ ] The fix is verified by running the affected tests with the process clock
      or `TZ` set to a value inside the previously-failing window and observing
      a pass
- [ ] Any remaining reliance on the ambient clock in `Tests/Watchlists/` is
      either removed or recorded with a reason

## Notes

The incident: this red was carried for part of a task as a suspected code
regression before anyone re-ran it under `TZ=UTC`. A flake that only fires in a
two-hour window each night is expensive precisely because it is usually green —
the cost is not the failure, it is the investigation it triggers in whichever
unrelated branch happens to run at 00:38.
