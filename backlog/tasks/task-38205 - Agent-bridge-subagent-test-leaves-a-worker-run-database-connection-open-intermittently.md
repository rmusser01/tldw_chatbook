---
id: TASK-38205
title: Agent bridge sub-agent test leaves a worker run-database connection open intermittently
status: To Do
created_date: 2026-10-30 09:30
labels:
- agents
- flaky-test
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Chat/test_console_agent_bridge.py::test_tool_marker_with_brackets_renders_literally_not_escaped errors intermittently at teardown: the autouse _retire_original_run_databases fixture finds an AgentRunsDB maintenance participant still holding an admitted connection after the main-thread close ("Main-thread original close must not hide a live worker owner"). The test spawns a sub-agent, so a worker thread appears to still own a run-database connection after run_reply returns. It reproduced in 1 of 6 runs of the file on dev 03eb75814b (xdist -n 4) and on PR #3286's heads, so it is not caused by that PR. The rate is load-sensitive: 1 of 6 on dev in one session, but up to 10 of 11 whole-file runs on a loaded machine. Likely mechanism, from code reading: worker_guard (Agents/activation.py) closes a fleet child's thread-local AgentRunsDB handle in its finally, which runs only after run_child returns, i.e. after fleet.finish and the settle signal (agent_service.py, the child run path), so the primary can return and the fixture can close and assert while the child thread has not yet reached that finally. Either the fleet should signal settlement only after the child releases its connection (a real ownership gap), or the fixture should wait for the late release.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The open connection's owning thread and call site are identified, and the leak is classified as a product ownership bug or a test-timing assumption, with evidence.
- [ ] #2 The test file passes 20 consecutive runs under xdist on dev without weakening the fixture's no-live-worker-owner assertion unless the ownership contract changes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
