---
id: TASK-21113
title: >-
  Screen pre-importer niceness - yield between modules, priority order, low-core guard
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - startup
priority: low
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21113).

`_preimport_screens` (app.py:11762-11886) is a tight no-yield loop importing ~21 screen modules
(~117k lines + closures) on a GIL-holding daemon thread starting 0.2 s after first paint. A
live A/B on multi-core hardware measured NO first-interaction penalty (keystroke median 82 vs
99 ms, 30 s CPU 2.51 vs 2.75 s - within noise), so this is design hardening for 1-2-core
machines, not a measured regression: on such boxes the entire post-boot window has a CPU
competitor for every keystroke. Keep the `TLDW_SCREEN_PREIMPORT` overrides.

## Acceptance Criteria

- [ ] A short sleep between route imports bounds GIL contention to one module at a time
- [ ] Import order starts with the configured default tab, then the heavy trio
- [ ] The pre-import pass parks while a screen navigation is resolving, and is disabled (or heavily throttled) below 4 CPU cores
- [ ] The existing A/B env override still works both ways
