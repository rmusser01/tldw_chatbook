---
id: task-2539
title: '"Server-source tools are display-only." message is unpinned at its raise site'
status: To Do
assignee: []
created_date: '2026-08-06 09:48'
labels:
  - mcp
  - honesty
  - tests
dependencies: []
priority: low
---

## Description

PR-T3 Task 3's refusal classifier (`mcp_workbench.py`, `_is_permission_refusal()`)
matches a specific `ValueError` by exact string —
`_SERVER_SOURCE_DISPLAY_ONLY_MESSAGE = "Server-source tools are display-only."` — to
render it as `Blocked · not run` instead of `Failed`. The string is raised at
`unified_control_plane_service.py:2235` (inside `execute_hub_tool()`, for a
server-source key).

Nothing pins that literal AT ITS RAISE SITE. The UI-side classifier has a test
asserting its own constant, but no test in the `unified_control_plane_service` /
`execute_hub_tool` suite asserts the exact text the production code actually raises.
A future reword of that message — even a small tidy-up unrelated to this PR — would
silently break the string-match in `_is_permission_refusal()`, and the refusal would
revert to rendering `Failed`. Nothing in the existing suite would catch it: both
sides currently pass independently, and neither test reads the other's string.

## Acceptance Criteria

- [ ] A test in the `execute_hub_tool` / `unified_control_plane_service` test suite
      asserts the exact message raised for a server-source tool execution attempt.
- [ ] Either that test asserts the same literal `_is_permission_refusal()` matches,
      or a single shared constant is introduced that both the raise site and the UI
      classifier import (preventing future drift structurally, not just by test
      coverage).
- [ ] No behavior change to the refusal path itself — this is a coverage/drift-proofing
      fix only.
