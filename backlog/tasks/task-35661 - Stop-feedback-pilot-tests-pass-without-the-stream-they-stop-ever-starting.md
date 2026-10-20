---
id: TASK-35661
title: Stop-feedback pilot tests pass without the stream they stop ever starting
status: To Do
assignee: []
created_date: '2026-10-19 13:20'
labels:
  - console
  - tests
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`_start_held_stream` in Tests/UI/test_console_stop_feedback.py presses Send, then polls for the first streamed chunk for about 3 s. It never asserts that the chunk arrived, so a Stop test can go on with no run to stop.

Found under TASK-35660, with a scratch harness that holds the config locks across `_select_llamacpp_console`. That harness was run with the shared helper as it was before TASK-34789, which leaves Send disabled ("Add API key in Settings > Providers & Models before sending."). The press was dropped and no stream started. test_stop_renders_stopped_state_even_when_direct_sync_coalesces still passed: it asserts that "Streaming…" is absent, which is just as true when nothing ever streamed. The shared helper now renders before it returns (TASK-34789), so this exposure needs another cause of a missed send. The test's guarantee should not depend on that.

Separately, test_stop_click_acknowledges_synchronously_and_records_stop_row fails locally on clean dev 4de437a5c5 (macOS, plain command) with "no explicit stopped-by-user transcript record". It fails with and without the harness. The cause has not been investigated.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each stop-feedback pilot test fails, naming how far the send got, when its held stream never starts
- [ ] #2 The local failure of test_stop_click_acknowledges_synchronously_and_records_stop_row on dev is attributed to a cause and recorded, and fixed if it is a test fault
<!-- AC:END -->
