---
id: TASK-457
title: >-
  Console first-feedback cluster follow-ups: cold-send optimistic echo, rail
  pressed state, inspector-during-run
status: To Do
assignee: []
created_date: '2026-07-22 02:20'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remaining pieces of the task-351 first-feedback finding (Console UX review j4-first-feedback-latency-cluster) not covered by the warm send-echo fix. Sub-symptom (a)-cold: on a cold provider the user's message still waits on the readiness probe before appearing — needs an optimistic user-append BEFORE resolve_for_send with an honest block/error row on failure (a real blocked-send behaviour change). Sub-symptom (b): rail conversation rows are plain Buttons with no pressed/loading feedback, so a slow/failed open reads as a dead click. Sub-symptom (c): the Inspector toggle during a run shows no immediate acknowledgment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 (a) Cold-provider first send echoes the user's message before the readiness probe resolves; a not-ready provider shows an honest block/error row instead of a silently-dropped message
- [ ] #2 (b) Rail conversation rows show a pressed/loading acknowledgment on click
- [x] #3 (c) Inspector toggle acknowledges immediately (within ~100ms) even during an active run
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
(c) DONE — no code change needed; already resolved by PR #745 (task-344/349,
the run-scoped workspace-context guard). The ~0.7s lag in the review (dev
cad9e271d, pre-#745) was the ~5x/sec workspace-context `sync_state` recompose
during a run blocking the event loop and delaying the click; #745's guard
(chat_screen.py ~6328: skip `sync_state` when `run_active and already_synced and
not state_changed`) removed that churn. The toggle handler itself
(`on_console_inspector_rail_open` → `_set_console_rail_preference(right_open)` →
`_sync_console_rail_visibility_if_changed`, a synchronous `styles.display` flip)
was always immediate. Served-app measured (llama live, mid-run): Inspector-open
latency 18–54ms during a run vs 6ms idle — well under the ~100ms AC. Regression
test (kept on one line so it copies cleanly):

```
Tests/UI/test_console_tick_gating.py::test_console_workspace_context_fresh_tray_still_synced_mid_run
```

(a) COLD ECHO — remaining, a real behaviour change: append the USER message in
`submit_draft` after validation but BEFORE `resolve_for_send`. On a not-ready
provider the USER row + SYSTEM block-row both persist. Open UX decision (needs
sign-off): the draft is currently kept on block (`should_clear_draft=False`,
task-340 draft-preservation), so echoing before resolve would make each blocked
retry echo a NEW duplicate USER row — the coherent model is clear-draft-on-block
+ retry-by-retype, which changes task-340 behaviour and every "blocked send →
no user message" test. Verify cold latency with a genuinely slow/cold provider.

(b) RAIL LOADING FEEDBACK — remaining: on a conversation-row click, mark the row
loading (spinner/dim) until `_resume_console_workspace_conversation` completes or
errors, so a slow/failed open no longer reads as a dead click. Needs load-state
tracking on the row + served-app verification.
<!-- SECTION:PLAN:END -->
