---
id: TASK-2157
title: 'Console: dictionary send agent-branch test fails (KeyError: agent_messages)'
status: To Do
assignee: []
created_date: '2026-08-07 16:51'
updated_date: '2026-08-07 18:08'
labels:
  - console
  - tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_dictionary_send_integration.py::test_native_send_applies_conversation_dictionary_agent_branch fails deterministically: submit_draft returns accepted=True but the fake agent bridge's run_reply never captures agent_messages before the assertion (KeyError at test line 185); agent reply logs start/end during teardown unwind, suggesting the agent reply now completes asynchronously after submit_draft returns. Verified PRE-EXISTING: fails identically at 844966c5d (pre-TASK-2154-batches-4/5 baseline) and at 6dc8d41a8. Not caused by the Console UX remediation batches. Needs investigation of when the agent-reply dispatch stopped being synchronous with submit_draft (or a test-side await for bridge completion).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Duplicate of TASK-2155 (same test, same KeyError: 'agent_messages', same root cause -- agent bridge never invoked synchronously in harness). Re-verified during the 2154 batches-4/5 gate: fails identically at 844966c5d (pre-session baseline) and at 6dc8d41a8. Archiving as duplicate.
<!-- SECTION:NOTES:END -->
