---
id: TASK-195
title: >-
  Fix pre-existing failure: console workspace-context active-conversation marker
  test
status: To Do
assignee: []
created_date: '2026-07-12 12:44'
labels:
  - tests
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_workspace_context_rail.py::test_console_workspace_context_syncs_active_conversation_marker fails on dev (asserts '> Planning thread' marker in the rail text; reproduced at 75e6987c and 89de6554 during the 2026-07 UAT remediation, masked by the intentionally-cancelled CI). Root-cause whether the marker sync or the test expectation drifted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test passes on dev for the right reason (behavior fixed or expectation honestly updated with rationale),No other workspace-context rail behavior regresses
<!-- AC:END -->
