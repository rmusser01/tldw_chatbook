---
id: TASK-28227
title: 'Agent loop: active-turn redirect keeping completed tool results'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:43'
updated_date: '2026-09-01 15:56'
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

## Renumbering provenance

Renumbered from `TASK-26000` to `TASK-28227` on 2026-09-02 under the
TASK-19601 older-arrival rule. The formatter characterization task was created on
2026-08-30 at 15:39; this redirect task was created on 2026-08-31 at 15:43, so the
older formatter task keeps `TASK-26000`. Historical commits may still cite the old
ID; current task, code, test, and documentation references use `TASK-28227`.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cancelling with intent to redirect aborts only the in-flight model request; completed tool results in the same turn are retained
- [x] #2 The correction is appended as a user-authored message and the turn re-runs with the retained context
- [x] #3 Partial streamed text already shown to the user is preserved as assistant context rather than silently dropped
- [x] #4 A redirect requested while a tool call is executing degrades to steering rather than corrupting the tool_calls/tool pairing
- [x] #5 Plain Stop with no correction behaves exactly as today and remains terminal
- [x] #6 Tests cover: redirect mid-stream retains prior tool results, redirect during tool execution degrades to steering, plain stop unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED loop tests (6, incl. fallback-stickiness enforcer)\n2. Loop: has_pending_redirect probe on LoopDeps + post-transport redirect branch (keep tool results, strip fences from partial, plain user correction, continue in-loop)\n3. Service: redirect_primary + per-run abort flag, drain clears flag atomically, on_primary_redirect_ready hook\n4. Bridge: stream_cut predicate (primary streams only, never LoopDeps.should_cancel) + on_redirect_ready plumb\n5. Console: /redirect command + Redirect button next to Stop (user's chosen entry points), controller redirect_active_run with steer-hook lifecycle\n6. Guide + pins
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Redirect ships in two commits: a26dfc92b (feature) + the review-fix round. Loop: LoopDeps.has_pending_redirect probe + a post-transport redirect branch (before continuation persistence) that keeps completed tool results, strips fences from the cut partial (incl. dangling opens), appends the correction as a PLAIN user message, and continues IN-LOOP so a sticky fallback switch survives (enforcer test). Service: redirect_primary shares the steering mailbox + a per-run abort flag raised/cleared under one lock; mid-continuation calls suppress the cut (F1 — cutting a checkpointed chain would RUN_ERROR); on_primary_redirect_ready hands (redirect_fn, abort_probe). Bridge: stream_cut cuts only primary streams; a cut prose turn gets a separator so the re-run doesn't glue (F2, mutation-verified). Console: /redirect command + Redirect button next to Stop (visibility synced with Stop; takes the composer draft), controller redirect_active_run with steer-hook lifecycle (4 pop sites paired). Redirect during tool execution degrades to plain steering; plain Stop byte-identical (probe-first guard keeps fleet stop-semantics poll counts). 16 new tests (8 loop + 8 service) + 3 bridge tests; Tests/Agents at the exact 7-name baseline, 2342 passing. Known v1 display gap: the re-run streams into the same transcript row (like multi-call runs today); separate settled-partial/user-correction rows need mid-run store insertion — follow-up material. Docs/User_Guide/console.md updated.
<!-- SECTION:NOTES:END -->
