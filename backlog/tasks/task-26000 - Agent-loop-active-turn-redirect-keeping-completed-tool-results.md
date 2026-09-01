---
id: TASK-26000
title: 'Agent loop: active-turn redirect keeping completed tool results'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-31 15:43'
updated_date: '2026-09-01 15:29'
labels:
  - agents
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correcting a running agent costs the user every completed tool result. Verified on origin/dev: Stop is terminal - Chat/console_chat_controller.py:13048-13126 settles the stream as "Response stopped." and Agents/agent_runtime.py:1352-1362 returns RUN_CANCELLED, so a correction becomes a fresh turn and work already done in that turn is discarded. Hermes aborts only the in-flight model request, keeps completed messages and tool results, records displayed partial reasoning as assistant context, appends the correction as a real user message and re-runs the same turn. Distinct from task-25903 (steering), which injects guidance without cancelling the current model call; redirect is for when the current call is already wrong.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Cancelling with intent to redirect aborts only the in-flight model request; completed tool results in the same turn are retained
- [ ] #2 The correction is appended as a user-authored message and the turn re-runs with the retained context
- [ ] #3 Partial streamed text already shown to the user is preserved as assistant context rather than silently dropped
- [ ] #4 A redirect requested while a tool call is executing degrades to steering rather than corrupting the tool_calls/tool pairing
- [ ] #5 Plain Stop with no correction behaves exactly as today and remains terminal
- [ ] #6 Tests cover: redirect mid-stream retains prior tool results, redirect during tool execution degrades to steering, plain stop unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED loop tests (6, incl. fallback-stickiness enforcer)\n2. Loop: has_pending_redirect probe on LoopDeps + post-transport redirect branch (keep tool results, strip fences from partial, plain user correction, continue in-loop)\n3. Service: redirect_primary + per-run abort flag, drain clears flag atomically, on_primary_redirect_ready hook\n4. Bridge: stream_cut predicate (primary streams only, never LoopDeps.should_cancel) + on_redirect_ready plumb\n5. Console: /redirect command + Redirect button next to Stop (user's chosen entry points), controller redirect_active_run with steer-hook lifecycle\n6. Guide + pins
<!-- SECTION:PLAN:END -->
