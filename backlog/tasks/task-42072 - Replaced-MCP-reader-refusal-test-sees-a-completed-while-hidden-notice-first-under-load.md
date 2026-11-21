---
id: TASK-42072
title: "Replaced-MCP-reader refusal test sees a 'completed while hidden' notice first under load"
status: To Do
assignee: []
created_date: 2026-11-20 01:10
labels:
- follow-up
- tests
- flaky
priority: medium
references:
- Tests/UI/test_console_send_without_mcp_store.py
- tldw_chatbook/Chat/console_runtime.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-42071 local review on 2026-11-19, at dev a6df4d8d48 plus that fix. The test `Tests/UI/test_console_send_without_mcp_store.py::test_unbound_store_with_a_replaced_mcp_reader_is_still_refused` was laned for the first time by the TASK-41630..41655 burn-down.

**Observed failures.** It failed in 2 of 3 local `-n 4` runs of its file, while a serial `Tests/Architecture` run loaded the machine. In those runs the first recorded notice was "A Console turn completed while hidden. Return to Console to review." rather than "Console preparation source unavailable; Send again." It passed alone, in 3 of 3 quiet runs on that branch and 3 of 3 on dev, and in the CI UI Fast Lane run that exposed TASK-42071.

**Where the notice comes from.** `ConsoleRuntime._notify_terminal_receipt` (`Chat/console_runtime.py`) sends that notice only for a COMPLETED terminal receipt that it does not count as on screen.

**Two possible causes, unresolved:**
- A stray receipt arrives first.
- Under load, the replaced-reader Send is not refused at all.

Telling them apart matters, because the second would be a product bug (the test's second assertion, `gateway.sent_messages == []`, never ran in those failures).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The source of the "completed while hidden" receipt in a failing run is identified, with the evidence recorded
- [ ] #2 A replaced MCP reader is shown to refuse the Send under load, or the product is fixed so that it does (red first)
- [ ] #3 The test passes in 10 of 10 loaded `-n 4` runs of its file
<!-- AC:END -->
