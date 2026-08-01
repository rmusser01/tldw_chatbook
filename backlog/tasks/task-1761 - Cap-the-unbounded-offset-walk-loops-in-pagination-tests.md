---
id: TASK-1761
title: 'Cap the unbounded offset-walk loops in pagination tests'
status: To Do
assignee: []
created_date: '2026-08-01 20:05'
labels:
  - watchlists
  - briefings
  - testing
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four briefings pagination tests walk `offset` pages with an uncapped `while True` loop, copied
from the same precedent each time (first noted as a deferred minor at task-2 of the phase 3
close-out, `backlog/tasks/task-1540`'s programme):

- `Tests/Subscriptions/test_briefing_feed_query.py:253`
- `Tests/Subscriptions/test_briefing_audio_db.py:213`
- `Tests/Subscriptions/test_briefing_presets_db.py:133`
- `Tests/Subscriptions/test_briefing_presets_db.py:311`

Each follows the same shape:

```python
offset = 0
while True:
    page = db.list_...(..., limit=limit, offset=offset)
    if not page:
        break
    seen.extend(...)
    offset += limit
```

The loop's only exit is an empty page. If a regression made the underlying query **ignore**
`offset` entirely (e.g. always returning the first `limit` rows), the same non-empty page would
come back forever -- the loop never sees the empty page that is supposed to end it. Today that
spins each test until pytest's global timeout (300s) kills the whole run, rather than failing
fast with a message that points at the actual bug. A slow, timeout-shaped failure is a much worse
debugging experience than an immediate, named assertion failure, and it burns the entire test
run's time budget on one broken query.

This is not a live bug today -- all four queries correctly honor `offset` right now (each test
passes). This is purely about hardening the tests' failure mode for the day one of them
regresses.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the four `while True` offset-walk loops has a local iteration cap
- [ ] #2 Exceeding the cap fails the test immediately with a clear, specific message (naming which
      query/test hit it) rather than spinning to pytest's global timeout
- [ ] #3 All four tests still pass unchanged under normal (non-regressed) behavior
<!-- AC:END -->
