---
id: TASK-15764
title: 'Watchlists: URLMonitor extraction and diff work off the event loop'
status: To Do
assignee: []
created_date: '2026-08-13 12:31'
labels:
  - perf
  - watchlists
priority: medium
---

## Description

Follow-up named explicitly in task-15463's "Scope" section (input-latency
burn-down, `Docs/Design/2026-08-11-input-latency-audit.md`): task-15463 moved
the watchlist scheduler's sqlite work and feed-body parsing off the event
loop, but deliberately left `URLMonitor`'s own CPU work on the loop as "a
separate, larger change." For `url`/`url_list`/`sitemap` sources this covers:

- `ContentExtractor.extract_text_from_html` (BeautifulSoup over a page up to
  `MAX_FETCH_BYTES_PAGE`, reached from `monitoring_engine._fetch_url_content`)
- the difflib work in `check_url` (`_segment_for_diff` called twice,
  `build_change_diff`, `added_and_removed_text`, `classify_change_type`)

Both are pure CPU with no sqlite involvement, so `asyncio.to_thread` applies
without task-15463's in-memory-connection hazard (no `:memory:` guard is
needed here — this is not a DB call). A `url_list` source multiplies both
costs by its URL count, so a source that watches several URLs pays this
synchronously on the loop once per check, per URL.

## Acceptance Criteria

- [ ] `ContentExtractor.extract_text_from_html` runs off the event loop for
      `url`/`url_list`/`sitemap` checks (evidence: thread-identity test)
- [ ] `check_url`'s difflib work (`_segment_for_diff` ×2, `build_change_diff`,
      `added_and_removed_text`, `classify_change_type`) runs off the event loop
- [ ] A `url_list` source with multiple URLs shows the same off-loop behavior
      for each URL, not just the first
- [ ] Existing Watchlists/Subscriptions suites (including task-15463's
      `test_watchlists_db_instance_and_off_loop.py`) stay green; due-detection,
      run records, and change-classification semantics are unchanged
