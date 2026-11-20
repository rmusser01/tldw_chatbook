---
id: TASK-42138
title: "Retire the local receipt workarounds in the Console send tests once receipts are cleared between tests"
status: To Do
assignee: []
created_date: 2026-11-20 20:06
updated_date: 2026-11-20 20:06
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_send_without_mcp_store.py
- Tests/UI/test_console_runtime_ownership.py
- Tests/conftest.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the consoletests stream (TASK-42018, round 18). TASK-42018 (round-18 consoletests branch, d10411ca0b, not yet on dev) clears unseen Console receipt marks between bootstrap-profile tests in `Tests/conftest.py`. Before it, TASK-41900 and TASK-41904 added two local workarounds to `Tests/UI/test_console_send_without_mcp_store.py`:
- `_acknowledge_leaked_receipts` (line 105) marks every unseen receipt seen before a test mounts its app;
- `_wait_for_own_receipts_seen` (line 123) waits until the test's own receipts are seen, so a later test's app does not announce them.

`Tests/UI/test_console_runtime_ownership.py` imports both (lines 452-453; calls at 479 and 533). Once the conftest clear is on dev, the first is redundant and the second may be.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With TASK-42018's clear on dev, each helper is either removed with its call sites, or kept with a comment naming what it still guards that the clear does not
- [ ] #2 For each removed helper, the tests that called it pass in one process right after a test that leaves an unseen receipt (`-p no:xdist`), and the notes record that run
- [ ] #3 Both files' tests pass
<!-- AC:END -->
