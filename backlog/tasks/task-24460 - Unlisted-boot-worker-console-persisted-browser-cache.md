---
id: TASK-24460
title: Unlisted boot worker console-persisted-browser-cache
status: Done
assignee: []
created_date: '2026-08-29'
labels:
  - performance
  - boot
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The boot worker census is red: a worker with group `console-persisted-browser-cache`
(started from `UI/Console_Modules/workspace.py:2553`) runs during boot without a row on the
reviewed allowlist.

First paint is the most contended moment in the application's life (finding 22215, GIL
contention). The census exists so that every worker riding the boot is a deliberate, reviewed
decision, and TASK-22215's stagger policy consumes this list.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `test_boot_worker_and_thread_starts_stay_within_the_allowlist` passes on a pristine checkout
- [x] #2 The worker is either deferred to first feature use, or added to the allowlist with its owning feature named
- [x] #3 If it is deferred, the persisted browser cache still populates correctly on first use of the feature that needs it
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Not a new worker and not a deferral: a NAMING regression. The allowlist already carried
`("_refresh_console_persisted_rows_cache", "console-persisted-browser-cache")`, but the census
recorded `("", "console-persisted-browser-cache")`.

Textual derives a worker's name as `name or getattr(work, "__name__", "") or ""`
(`worker_manager.py:112`), and this call site had been wrapped in `functools.partial` -- which has
no `__name__`. So the worker silently became anonymous, breaking its allowlist row and making it
unidentifiable in every worker diagnostic. Fixed by passing `name=` explicitly, which a partial
cannot erase.

Guard now green. 56 other `run_worker(partial(...))` sites exist repo-wide with the same latent
anonymity; none are boot workers, so they are out of this task's scope and left alone.

Modified: `tldw_chatbook/UI/Console_Modules/workspace.py`.
<!-- SECTION:NOTES:END -->
