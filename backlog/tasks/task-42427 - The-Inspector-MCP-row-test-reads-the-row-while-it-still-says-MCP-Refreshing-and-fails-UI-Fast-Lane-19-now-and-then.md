---
id: TASK-42427
title: "The Inspector MCP-row test reads the row while it still says 'MCP: Refreshing…' and fails UI Fast Lane 19 now and then"
status: To Do
assignee: []
created_date: 2026-11-22 22:10
updated_date: 2026-11-22 22:10
labels:
- follow-up
- ci
- console
- mcp
- tests
- flaky
priority: low
references:
- Tests/UI/test_console_send_without_mcp_store.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22), in the CI runs the round's filing read. Checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports); paths start at the repository root.

**Why.** `Tests/UI/test_console_send_without_mcp_store.py::test_inspector_mcp_row_says_off_and_why_when_the_store_is_unbound` (line 369, parametrized) failed the required "UI Fast Lane (19)" twice in a scan of the 140 Derived Artifacts PR runs created from 2026-11-20 00:00Z to about 21:50Z on 2026-11-22, each time on a different case, both reading the row while it still showed its transient text:
- `[non-default-data-dir]`, run 44760707115 (job start 2026-11-20 17:55Z): `assert 'MCP: Refreshing…' == 'MCP: off · C...settings file'`;
- `[bound]`, run 44974742439 (job start 2026-11-22 04:48Z): `assert 'MCP: Refreshing…' == None`.

All of the test's cases pass locally at the base. The test reads the row right after opening the Inspector (lines 398-401; for the `[bound]` case it expects no MCP text at all, `None`) and again after the reply, without waiting for the row's refresh to finish, so a row still refreshing is read as `MCP: Refreshing…`. Rare, but it reds the required lane for reasons outside the PR.

The same file's `test_unbound_live_work_mcp_row_fits_the_section_at_full_screen` failed that lane in 66 jobs up to 2026-11-21 23:11Z and not since; it is not part of this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Before each comparison, the test waits until the Inspector's MCP row has left its refreshing placeholder, with a load-tolerant deadline
- [ ] #2 With the row's first refresh delayed by a temporary edit (reverted), the old test fails with `MCP: Refreshing…` and the new one passes, for every case
- [ ] #3 The test passes in repeated runs of its file under parallel load (at least 20 runs, or a soak recorded in the notes)
<!-- AC:END -->
