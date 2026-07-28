---
id: TASK-1141
title: Duplicate park toast when a viewed run completes
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 18:05'
updated_date: '2026-07-28 01:53'
labels:
  - console
  - approvals
  - uat
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT (Docs/superpowers/qa/parallel-agents-uat-2026-07-27, F2): with session B parked (park toast already shown), completing a run in the VIEWED session A re-fires "Agent in B (workspace) needs approval." for B's unchanged round. The once-per-card toast guard does not survive the re-marshal/re-park performed by the viewed-run-completion sync. Repro: park B; run and complete a run in viewed A; second toast fires at A's completion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A parked round toasts exactly once across its lifetime, including viewed-run completions, visits, and re-derives.
- [x] #2 Regression test reproducing the viewed-completion re-toast path.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace every call site of park_pending_approval/ChatScreen._park_console_approval across ConsoleChatController and ChatScreen; determine whether the viewed-run terminal transition, switch_session/_remount_parked_*, or the marker/unvisited stamping path can invoke it twice for one round.
2. Write a failing repro test driving the real sequence: park B via the established seam, complete A's run via the real terminal transition (_set_run_state), then simulate the re-invocation of the shared park seam for B's unchanged round; assert exactly one toast (must fail on HEAD).
3. Add a round/request-id-keyed idempotency guard to _park_console_approval (the single shared seam for all three bridges) so a re-invocation for the SAME still-live round is absorbed, while a genuinely new round still toasts.
4. Sweep: confirm/verify the guard covers the skill-install and skill-script park paths (same shared seam) with dedicated regression tests.
5. Run the gate suite; update the task file with Implementation Notes documenting the root-cause narrative.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: `ChatScreen._park_console_approval(session_id)` (the single shared
UI-thread bridge target for all three worker-thread bridges --
`request_mcp_approvals`/`request_skill_install_confirm`/
`request_skill_script_confirm`'s park branches) had NO idempotency of its
own. Its "once per round" property relied entirely on the structural
assumption that each owning bridge invokes it exactly once per round --
true for a single, race-free bridge call, but nothing prevented a SECOND,
differently-triggered invocation for the same still-live round from
re-announcing it. Task 9's own "one-per-round toast" framing was a
documentation-only assumption, not an enforced guard.

Diagnosis: exhaustively traced every suspect named in the brief --
`_set_run_state`'s COMPLETED branch (the viewed session's own terminal
transition never reaches the non-active toast branch, since `target ==
active_session_id`), `_finalize_agent_*` (scoped entirely to the completing
run's OWN session_id, never touches a sibling session's state),
`switch_session`/`_remount_parked_skill_install`/`_remount_parked_skill_script`
(re-derive via `set_pending_approval`/`set_pending_skill_install`/
`set_pending_skill_script` only -- never call `park_pending_approval`), and
the unvisited-marker stamp (same non-active-only gate as the toast branch,
so it's skipped for the viewed session too). None of these invoke
`park_pending_approval` a second time for one round under single-threaded/
synchronous conditions. The exact live trigger (a `call_from_thread`
re-marshal race, or some other timing-sensitive path) could not be pinned
to a specific line via static tracing alone, but the underlying defect is
unambiguous regardless of the trigger: the callback itself carried zero
protection against a second invocation for an unchanged round.

Fix: `_park_console_approval` now keys its toast on the round/request
id(s) currently retained for `session_id` across all three bridges'
payload maps (`_parked_approval_payloads`/`_parked_skill_install_payloads`/
`_parked_skill_script_payloads` -- the SAME maps `switch_session` already
treats as the source of truth for "what round is this session's card
showing right now"), via a new `_current_park_round_ids` helper. A new
`ChatScreen._console_toasted_park_round_ids` set (never pruned; bounded by
total rounds ever parked) records which round/request ids have already
been announced. A re-invocation whose current round/request id(s) are
already in that set is silently absorbed (no toast); a genuinely NEW round
(different id, even for a session with an existing outstanding round)
still toasts and gets recorded. When none of the three maps carry an id
yet (the standalone test-seam usage several existing tests rely on, or a
caller racing ahead of the owning bridge's own payload write), the method
falls back to the pre-fix unconditional toast, preserving existing
direct-call test behavior. The callback's public single-arg
`Callable[[str], None]` contract is unchanged. The same guard shape covers
all three bridges automatically since they share this one seam.

Tests added (Tests/UI/test_console_parallel_runs.py):
- test_park_toast_survives_a_viewed_run_completion_re_invocation: parks B
  via the real threaded MCP bridge, drives A's real terminal transition via
  `_set_run_state`, then re-invokes the shared park seam for B's unchanged
  round; asserts exactly one toast total. Verified to FAIL on HEAD (2
  toasts) and pass with the fix.
- test_park_toast_fires_again_for_a_genuinely_new_round_same_session: a
  resolved round followed by a genuinely new round for the same session
  still toasts (guards against over-suppression).
- test_skill_install_park_toast_survives_a_re_invocation_for_the_same_round
  / test_skill_script_park_toast_survives_a_re_invocation_for_the_same_round:
  sweep confirming the same guard covers the other two bridges.

Files modified:
- tldw_chatbook/UI/Screens/chat_screen.py: `_console_toasted_park_round_ids`
  init, `_park_console_approval` guard, new `_current_park_round_ids`
  helper.
- Tests/UI/test_console_parallel_runs.py: four new tests (above).

Gate suite (Tests/UI/test_console_parallel_runs.py +
Tests/UI/test_console_mcp_approval.py +
Tests/UI/test_skill_install_concurrent_confirms.py): 76 passed, 2 failed
-- both pre-existing on HEAD (verified via git stash), matching the known
failures: test_batch_row_widgets_have_nonzero_geometry_and_do_not_overlap_under_bundled_css
(CSS-geometry batch-row) and
test_request_mcp_approvals_cancellation_records_denied_decision_to_execution_log
(mcp cancellation execution-log). No regressions introduced.

--- Review round 1 (approved-with-note) ---

Reviewer reproduced live on HEAD: the round-identity guard above only
consulted the three LIVE `_parked_*_payloads` maps via
`_current_park_round_ids`. A re-invocation of the shared park seam
arriving AFTER a round's own bridge `finally` teardown (which pops the
round from those maps once resolved) found everything empty and fell
through to the "no identity to key on" unconditional-toast fallback --
re-firing the toast for a round with no card left, despite its id
already sitting in the never-pruned `_console_toasted_park_round_ids`
set. Since the organic production trigger for the original F2 finding
was never definitively isolated, this post-teardown window is at least
as plausible an explanation as the still-live race the first fix
covered.

Fix: added `ChatScreen._console_last_parked_round_ids`, remembering the
most recent NON-empty `_current_park_round_ids` snapshot per session.
When the live lookup returns empty, `_park_console_approval` now falls
back to that remembered snapshot -- if every id in it is already in
`_console_toasted_park_round_ids`, the re-invocation is absorbed. Only a
session this screen has genuinely never seen a live round for (no
snapshot ever recorded) still falls through to the unconditional toast,
preserving the standalone test-seam behavior existing direct-call tests
rely on. A genuinely new round (different id) arriving after a full
teardown still toasts.

Tests added:
- test_park_toast_survives_a_post_teardown_re_invocation_for_the_same_round
  -- the reviewer's exact repro: park round-1, toast, tear down
  (discard_pending_round + pop payload, mirroring the bridge's own
  finally), re-invoke for the same session, assert no second toast.
  Verified to FAIL against the prior commit (e81bca5ff) and pass with
  this fix.
- test_park_toast_fires_once_for_a_new_round_arriving_after_teardown --
  a genuinely new round after full teardown still toasts once.

Gate suite (same three files, one foreground call): 78 passed, 2 failed
-- the same two pre-existing HEAD failures as before (CSS-geometry
batch-row, mcp cancellation execution-log). No regressions.

New HEAD: 5c6be2a33.
<!-- SECTION:NOTES:END -->
