---
id: TASK-322
title: Bound Console conversation history by tokens before dispatch
status: In Progress
assignee: []
created_date: '2026-07-20 18:45'
updated_date: '2026-07-22 14:03'
labels:
  - console
  - llm
dependencies:
  - task-320
  - task-321
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The native Console send path sends the entire conversation to the provider on every turn with no token or message cap. `console_chat_controller.py:1784` (`_provider_messages_for_session`) collects all session messages; the only budget applied in `_provider_message_payloads` (`console_chat_controller.py:1820`) is `max_history_images` — it counts images, not text tokens. Once a conversation exceeds the model window, cloud providers reject the request with a 400 (the conversation becomes un-continuable with no graceful degradation) and local servers silently front-truncate — dropping the prepended system prompt first — while token cost grows without bound. (The deprecated enhanced/legacy chat path shares this defect but is out of scope as it is being replaced by the Console.)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Console history is bounded by real tokens against the model input window (task-320/task-321) before dispatch
- [ ] #2 The system prompt and the latest user turn are always preserved; oldest turns are dropped in whole request-valid units (never splitting a tool_call from its tool_result)
- [ ] #3 The user is notified when earlier history was trimmed
- [ ] #4 A conversation that exceeds the model window remains continuable (no overflow-driven provider 400)
- [ ] #5 Tests cover the trim boundary and system-prompt preservation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-22-console-history-token-budget.md
<!-- SECTION:PLAN:END -->
