---
id: TASK-31382
title: Name the asking sub-agent on the ask_user card and its transcript marker
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-04 19:28'
updated_date: '2026-09-05 00:01'
labels:
  - console
  - agents
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PRD A11 says the question card names the asking agent. M2 (PR #2379) ships the attribution as a kind only -- the card title says 'A sub-agent has N questions for you' and the marker says 'Questions from a sub-agent' -- because a run carries no display label on dev: CurrentRunActor exposes kind, run_id and parent_run_id, and the transcript's own sub-agent markers are generic ('A sub-agent edited 3 files'). A user running a fleet cannot tell WHICH child is asking, which matters exactly when two children are working in parallel. Needs a run-id to display-label mapping (the fleet coordinator or AgentRuns_DB is the natural owner), threaded into the question payload's asked_by and read by the card title and the marker header.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The question card title names the asking sub-agent by its display label when one exists, falling back to the current generic copy
- [x] #2 The A14 transcript marker header carries the same label
- [x] #3 A primary-agent question is unchanged
- [x] #4 The label lookup never blocks the worker thread or raises into the round
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. CurrentRunActor gains an optional display label (default None) set by AgentService when it builds a sub-agent's actor: the named agent's name when the spawn named one, else a short form of the child's task.
2. request_user_questions copies actor.label into the card payload as asker_label; asked_by stays the kind.
3. ChatQuestionCard title and format_question_marker header name the label when present, falling back to the existing generic copy.
4. Tests: actor default, payload carries the label from the run actor context, card title, marker header; existing ask_user suites unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
CurrentRunActor gained an optional label (default None). AgentService sets it when it builds a sub-agent's actor in _run_one via subagent_display_label(agent_definition, task): the resolved named-agent name wins, else the task's first line cut to 40 chars. request_user_questions copies actor.label into the payload as asker_label; ChatQuestionCard's title reads "Sub-agent 'researcher' has N questions for you:" and format_question_marker's header reads "? Questions from sub-agent 'researcher' (N):", both falling back to the generic copy without a label; a primary agent is unchanged. The lookup is a field read on the run actor context, so it never blocks the worker thread or raises. Files: Agents/run_context.py, Agents/agent_service.py, Chat/console_chat_controller.py, Chat/console_agent_bridge.py, Widgets/Chat_Widgets/chat_question_card.py; tests in Tests/Agents/test_subagent_display_label.py, Tests/Chat/test_console_ask_user_round.py, Tests/UI/test_chat_question_card.py.
<!-- SECTION:NOTES:END -->
