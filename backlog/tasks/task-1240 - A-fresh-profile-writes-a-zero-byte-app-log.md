---
id: TASK-1240
title: A fresh profile writes a zero-byte app log
status: To Do
assignee: []
created_date: '2026-07-28 10:20'
labels:
  - logging
  - observability
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Running the app under a fresh profile produces a `tldw_cli_app.log` of **zero bytes** — after a full
boot, a minute of real work (a scheduled watchlist check that fetched and persisted 5 items from a
live feed), and a clean `Ctrl+Q` shutdown. The path resolves correctly:
`get_cli_log_file_path()` returns `~/.local/share/tldw_cli/<user>/tldw_cli_app.log`, the file is
created, and nothing is ever written to it.

The long-lived `default_user` profile has an 8.4 MB log, so file logging works somewhere. Whatever
attaches the file handler does not happen for a new profile, or happens after the records it should
capture.

Reproduced with two different scratch configs, one of which set `[logging] log_filename` and
`file_log_level = "INFO"` explicitly.

**Why this matters beyond logging.** It is the second half of the failure TASK-1212 exists to fix.
Watchlist checks silently did nothing for the life of the feature because a working scheduler and an
unwired one were indistinguishable by observation. TASK-1212 adds structured startup reporting, but
a user on a fresh install has nowhere to read it: the file log is empty, and the in-app Logs screen
only starts buffering once its persistent handler is installed, which is after early startup work
has already logged.

Diagnosing the original scheduling defect required a runtime import trace and a seeded database
probe. It should have required reading a log line.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A fresh profile's tldw_cli_app.log contains startup records after a normal boot
- [ ] #2 Records emitted before the in-app Logs screen's persistent handler is installed still reach the file log
- [ ] #3 The behaviour is identical for a brand-new profile and a long-lived one
- [ ] #4 A test asserts the file handler is attached, by running a boot path and checking the log is non-empty rather than by asserting the handler exists
- [ ] #5 If file logging is intentionally deferred or disabled under some condition, that condition is documented where the log path is resolved
<!-- AC:END -->
