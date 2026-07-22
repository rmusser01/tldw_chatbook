---
id: TASK-457
title: >-
  Console first-feedback cluster follow-ups: cold-send optimistic echo, rail
  pressed state, inspector-during-run
status: In Progress
assignee:
  - '@claude'
created_date: '2026-07-22 02:20'
updated_date: '2026-07-22 14:35'
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
- [x] #1 (a) Cold-provider first send echoes the user's message before the readiness probe resolves; a not-ready provider shows an honest block/error row instead of a silently-dropped message
- [x] #2 (b) Rail conversation rows show a pressed/loading acknowledgment on click
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

(a) COLD ECHO — remaining. PROTOTYPED + reverted 2026-07-22 after finding a
core-model blocker; scoped precisely for a focused follow-up. Approach: append
the USER row in `submit_draft` after validation but BEFORE `resolve_for_send`,
keep the draft on block (retry model = keep-draft, echoed row persists next to
the block-row, retry re-attempts — an honest attempt history, minimal churn:
only `test_blocked_provider_wip_copy` needs its message-list expectation
updated). BLOCKER (empirically confirmed): the echoed-but-blocked USER row would
pollute the NEXT successful send's provider context, because
`_provider_messages_for_session` includes all USER rows and only drops
`status=="failed"` ones (`skip_failed`, controller ~2725). Excluding the blocked
row therefore needs it marked failed — but `store.mark_message_failed` is
assistant-stream-only (`_validate_can_mark_terminal` → `ValueError: Only
assistant messages can enter terminal stream states`), and `append_message`
returns a `_snapshot`, not the live object, so direct `.status` mutation is a
no-op. So (a) needs a NEW store method (e.g. `mark_message_send_blocked(id)`)
that sets `status="failed"` on a non-streaming row without the terminal-guard,
plus wiring in submit_draft's resolve-block path (`try/except` must catch the
right error). Trade-off: a `"failed"` USER row renders `hello [failed]` in the
transcript (honest, but consider a distinct `"blocked"` status if that reads
oddly). Value is marginal (resolve is 24ms warm; only a genuinely cold/slow
provider startup exposes the gap), so this touches the core send path for a
narrow win — do it as its own reviewed change, and verify cold latency against a
genuinely cold provider.

(b) RAIL LOADING FEEDBACK — remaining: on a conversation-row click, mark the row
loading (spinner/dim) until `_resume_console_workspace_conversation` completes or
errors, so a slow/failed open no longer reads as a dead click. Needs load-state
tracking on the row + served-app verification.
<!-- SECTION:PLAN:END -->


## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
(a) DONE (the fuller reviewed change the plan called for; the earlier prototype's
core-model blocker is resolved). `submit_draft` now appends the USER row AFTER
validation but BEFORE `resolve_for_send`, so a slow/cold readiness probe no
longer leaves the transcript blank while the composer clears. On a not-ready
provider the row persists next to the honest SYSTEM block-row (no longer silently
dropped) and the draft is kept (composer clears only on the accepted path), so
the user re-attempts. Staged attachments embed on the echoed row but only CLEAR
on the success path (a blocked attempt keeps them staged).

Core-model piece (the plan's blocker): the echoed-but-blocked row must NOT enter
the next send's provider context. New `ConsoleChatStore.mark_message_send_blocked`
flips a NEVER-STREAMED row to `status="failed"` without the assistant-stream
terminal guard (`mark_message_failed` raises for non-assistants, and
`append_message` returns a snapshot so direct mutation is a no-op); the resolve-
block path calls it, and `_provider_messages_for_session(skip_failed=True)`
already drops it. Two correctness gates so a `failed` USER row reads right: the
transcript stream-state suffix (`[failed]`) and the "retry" message action are
both now assistant-only (a USER row has no assistant response to regenerate).

Verified RED→GREEN: store-method test; controller `test_not_ready_provider_still_
echoes_the_user_message` (roles == [user, system], row failed, draft kept) +
existing `..._exclude_visible_recovery...` proves context exclusion; message-body
+ message-action tests for the clean rendering; 274 Chat-suite + full native-flow
(188) green. Value note: resolve is 24ms warm, so this mainly helps a genuinely
cold provider startup. (b) rail loading feedback remains.

Hardened per Qodo #777 review: (1) the readiness probe is now wrapped so a
probe that RAISES or is cancelled after the optimistic echo also marks the row
send-blocked (then re-raises) — otherwise a never-sent USER row would leak into
the next send's context; (2) `mark_message_send_blocked` now rejects non-USER
and mid-stream rows (USER echo only), so a mistaken caller cannot flip an
assistant/system row to `"failed"` and bypass the assistant terminal guards.
Both covered by new RED→GREEN tests (`..._rejects_non_user_rows`,
`test_probe_exception_after_optimistic_echo_marks_row_blocked`); 276 Chat-suite
green.

(b) DONE. A rail conversation-row click that opens a not-yet-loaded persisted
conversation awaits `_resume_console_workspace_conversation` inline; a slow or
failing open used to read as a dead click. The pressed row is now flagged
`loading` (Textual's built-in spinner overlay) for the duration of the resume
and always cleared afterwards. New screen helper
`_set_console_conversation_row_loading(conversation_id, loading)` matches the row
by its `conversation_id` attribute and no-ops when the row is gone (a successful
resume recomposes the rail, which already drops the flag; the `finally` covers
the not-resumable/error return so the row never stays stuck spinning). Only the
slow `session_id is None` resume branch is wrapped — the already-open
`switch_session` path is instant and needs no spinner.

Verified RED->GREEN in `Tests/UI/test_console_workspace_context_rail.py`:
`test_console_conversation_row_loading_toggles_on_matching_row` (helper toggles
`.loading` on the matching row; unknown id is a no-op) and
`test_console_conversation_row_click_shows_loading_until_resume_finishes` (a row
press flags the row loading before the awaited resume runs and clears it once the
resume settles). Full rail + scope-row + handoffs suites green.

PR map: (c) shipped via PR #745; (a) cold-send optimistic echo via PR #777;
(b) rail loading feedback via PR #779 — all three ACs met.
<!-- SECTION:NOTES:END -->
