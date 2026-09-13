---
id: TASK-31508
title: Review-set listing issues 2N+1 statements with unbounded item scans
status: To Do
assignee: []
created_date: '2026-09-04 19:30'
labels:
  - performance
  - library
dependencies: []
priority: low
---

## Description (the why)

`Library/review_set_service.py:174-198` (`list_review_sets`) fetches up to
200 set ids, then per id calls `_read_review_set` (`:402-420`), which
re-fetches the header row it just enumerated and runs an unbounded
`SELECT * FROM review_set_items ... ORDER BY position` -- 2N+1 statements and
N full item scans to render a picker that needs each set's name and a
progress summary. Off the event loop (isolate_in_worker), but scales with
accumulated sets x items. Evidence:
`Docs/Design/2026-09-04-holistic-perf-review.md` section 7.

## Acceptance Criteria (the what)

- [ ] Listing review sets for the picker runs a bounded number of statements independent of set count (join or aggregate), without loading full item rows when only counts/summary are needed
- [ ] Picker rows render identically for the existing fixtures (existing review-set tests stay green)
