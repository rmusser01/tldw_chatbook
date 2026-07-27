---
id: TASK-1090
title: >-
  A failed watchlist fetch is swallowed into a debug log the user never sees
status: To Do
assignee: []
created_date: '2026-07-28 08:00'
labels:
  - watchlists
  - bug
  - observability
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`_check_now_source` wraps the whole fetch in `except Exception`, logs at **debug**, and shows a transient toast. Nothing durable records that a check failed, and `subscriptions.last_error` is only written by the service on paths that get that far.

**This is the swallow that hid TASK-1100.** Check now was raising `ValueError` on every single press — the entire feature was dead — and the only evidence was a debug line nobody reads and a toast that had vanished before anyone looked. Three UAT runs and a full test suite reported the screen as working while it fetched nothing.

The same shape appears throughout this screen: `except Exception: logger.opt(exception=True).debug(...)` around a service call whose failure the user needs to know about. A fetch is the one operation in Watchlists that *routinely* fails for ordinary reasons — the feed moved, the host is down, the XML is malformed, the network is out — so it is exactly the operation that must report.

AC #4 of TASK-1100 was left unchecked for this reason; it belongs here rather than folded into that fix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A failed check writes `subscriptions.last_error` and surfaces it in the Sources table's Status column
- [ ] #2 The failure is visible after the toast has gone — the user can find out why without repeating the action
- [ ] #3 An unexpected exception in the fetch path logs at `warning` or above, not `debug`
- [ ] #4 A run that fails is recorded in `local_watchlist_runs` with its error, not silently absent
- [ ] #5 A test makes the fetch raise and asserts the user-visible outcome, proven to fail against current code
- [ ] #6 The other `except Exception: ... .debug(...)` handlers on this screen are audited, and any that hide a user-facing failure are listed here or fixed
<!-- AC:END -->
