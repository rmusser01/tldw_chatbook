---
id: TASK-31503
title: Terminal ownership monitor scans the whole process table at 50 Hz
status: To Do
assignee: []
created_date: '2026-09-04 19:30'
labels:
  - performance
  - terminal
dependencies: []
priority: medium
---

## Description (the why)

`Terminal/posix_backend.py:968` (`_monitor_owned_processes`) wakes every
`_PROCESS_POLL_SECONDS * 2` = 20 ms and runs `_default_scan_locked`
(`:1128`), which enumerates `psutil.pids()` and calls `os.getsid` +
`os.getpgid` for EVERY pid on the system, plus `create_time()` per candidate.
Measured on the review machine (663 processes): 0.30 ms CPU per scan -> ~1.5%
of a core per open terminal session, continuously, ~30k syscalls/s, even at
an idle prompt; each session runs its own monitor thread, so open tabs
multiply linearly. Ownership bookkeeping does not need 50 Hz. Evidence:
`Docs/Design/2026-09-04-holistic-perf-review.md` section 4.

## Acceptance Criteria (the what)

- [ ] An idle terminal session's monitor consumes under 0.2% of a core (measured, not asserted from the constant)
- [ ] Ownership/reaping semantics are unchanged where they matter: process spawn/exit detection latency stays within the product's existing guarantees (existing Terminal ownership tests stay green)
- [ ] The chosen cadence/backoff is documented in the module with the measured cost that motivated it
