---
id: TASK-2155
title: >-
  Agent-branch console send never invokes agent bridge (pre-existing dev
  failure)
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-06 17:09'
updated_date: '2026-09-12 06:55'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: test-harness repair preserving existing durable acceptance and dictionary contracts.
1. Record current failure: RuntimeError at ChatPersistenceService.commit_durable_turn because durable Console Library policy no longer matches acceptance. The manually bound conversation has no policy row; hydration alone does not fix it.
2. Insert the real Console Library policy using the current session candidate, then hydrate the holder before Send, matching the adjacent world-info harness. Preserve real durable commit and agent dispatch.
3. Verify substituted model payload and raw transcript through both dictionary send branches. Do not weaken production authority or persistence checks.
4. Run targeted static checks and independent review, record evidence and close.
<!-- SECTION:PLAN:END -->
