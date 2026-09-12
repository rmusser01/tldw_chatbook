---
id: TASK-32376
title: dev red: test_fenced_nested_delivery_counts_exact_transformed_payload_before_mark mock lacks reasoning_replay kwarg
status: To Do
assignee: []
created_date: '2026-09-11 01:55'
labels:
  - tests
  - console
dependencies: []
priority: medium
---

## Description

Commit f043db12b77 on dev made `console_agent_bridge` pass `reasoning_replay=` to the counted-tokens seam, but the test's monkeypatched lambda in `Tests/Chat/test_console_agent_bridge.py` (`lambda *_: transformed_tokens`) does not accept keyword arguments, so `test_fenced_nested_delivery_counts_exact_transformed_payload_before_mark[90-done-3]` fails on dev and on every branch that merges it. Found while stacking the approval-card lanes; a one-line fix (`lambda *_, **__: transformed_tokens`).

Source: approval-card / MCP-permissions fix wave 2026-09-10/11 (plan `Docs/superpowers/plans/2026-09-10-approval-card-fix-wave.md`, review snapshot `.impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md`); rider recorded in the lane ledger, not fixed in the wave.


Also red at the same untouched baseline in `Tests/Chat/test_console_agent_bridge.py` (found while stacking lane A on dev 0e62de41d4): `test_successful_tool_payload_collisions_stay_success_live_and_resumed[tool call denied…]` and `test_run_reply_forwards_review_tool_calls_hook_to_agent_service`.

## Acceptance Criteria

- [ ] The test passes on dev without changing the source under test
- [ ] No other monkeypatched fake in that file rejects the `reasoning_replay=` kwarg
