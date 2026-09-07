---
id: TASK-2155
title: >-
  Agent-branch console send never invokes agent bridge (pre-existing dev
  failure)
status: To Do
assignee: []
created_date: '2026-08-06 17:09'
labels:
  - console
  - agent
  - test-failure
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
test_native_send_applies_conversation_dictionary_agent_branch fails on clean dev (KeyError: 'agent_messages'): the test's _fake_run_reply double is never invoked, so the agent branch of ConsoleChatController.submit_draft does not route to _agent_bridge.run_reply in the harness. Reproduced on dev @ ee3b4fae2. NOT caused by TASK-2154 batches 1-3. Either the agent-routing condition (_agent_runtime_enabled and _agent_bridge and not prefill and not force_plain) is not met under ConsoleHarness, or the fake's contract drifted from the real ConsoleAgentBridge.run_reply.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Test passes on dev without changing app behavior contracts,Root cause documented in task notes
<!-- AC:END -->
