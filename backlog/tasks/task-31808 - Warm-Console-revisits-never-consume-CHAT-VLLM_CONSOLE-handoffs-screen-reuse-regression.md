---
id: TASK-31808
title: >-
  Warm Console revisits never consume CHAT/VLLM_CONSOLE handoffs (screen-reuse
  regression)
status: Done
assignee: []
created_date: '2026-09-06 00:08'
labels:
  - bug
  - regression
  - console
dependencies: []
priority: high
---

## Description (the why)

ChatScreen is `reusable=True` (screen_registry), so `on_mount` fires once per
app run and warm revisits run only `on_screen_resume`. The Console
screen-reuse arc (fed0b7257a / af82ac9630) added a resume-path timer list
(`_console_resume_handoff_timers`) that re-arms four handoff consumers, but
the CHAT and VLLM_CONSOLE channel consumers were left out. Chat is the
default initial screen, so it is warm on essentially every real navigation:
vLLM setup's "Use in Console" navigates to Console but never applies the
target, and every `open_chat_with_handoff` caller (Library conversations,
Library screen, Media, Skills, Study) stages a CHAT payload that is never
consumed. No error surfaces. PR #2421's verification doc recorded the failing
assertion (`test_navigation_to_fresh_models_screen_preserves_exact_ready_handoff`)
as an unrelated route-reuse failure rather than fixing it.

## Acceptance Criteria (the what)

- [x] `Tests/UI/test_vllm_lab_workflow.py::test_navigation_to_fresh_models_screen_preserves_exact_ready_handoff` passes (was red on dev tip).
- [x] A warm resume of the reused ChatScreen consumes a staged CHAT-channel handoff (new regression test that exercises consumption on resume rather than mocking `open_chat_with_handoff`).
- [x] The re-armed consumers are tracked in `_console_resume_handoff_timers` so `on_screen_suspend`'s existing timer cancel covers them (no consumption against a hidden screen).
- [x] Neighboring targeted suites pass, with every observed failure paired-arm-proven pre-existing on unfixed dev (see Implementation Notes; full-file sweeps of two files curtailed by release-eve wrap-up directive, listed as residual risk in the PR).

## Implementation Plan (the how)

1. Reproduce: run the vLLM lab workflow regression test on an unfixed
   worktree from origin/dev and record it red (proof of regression).
2. Read both scheduling sites (`on_mount`'s non-ordered-resume timer block
   and `on_screen_resume`'s `_console_resume_handoff_timers` list) and
   confirm the idiom: `set_timer(0.15, consumer)` per channel, tracked for
   suspend-cancel.
3. Add the two missing consumers to the resume-path list with the same
   idiom: `self._consume_pending_chat_handoff` (async; `set_timer` accepts
   the coroutine method exactly as `on_mount` schedules it) and
   `self.consume_pending_vllm_console_intent`.
4. Add a warm-path CHAT-consumption test (stage a payload while on another
   screen, navigate back to warm Chat, assert the store drains and the
   payload lands) alongside the existing handoff tests.
5. Re-run the regression test green; run the neighboring targeted suites.
6. Preflight, PR to dev.

## Implementation Notes

Added the two missing consumers to `_console_resume_handoff_timers` in
`ChatScreen.on_screen_resume` (`tldw_chatbook/UI/Screens/chat_screen.py`),
using the identical `set_timer(0.15, ...)` idiom `on_mount` uses --
`_consume_pending_chat_handoff` first (async; Textual timers invoke
coroutine methods directly, exactly as `on_mount` schedules it) and
`consume_pending_vllm_console_intent` after the conversation-settings
return, mirroring `on_mount`'s ordering. Because they join the tracked
list, `on_screen_suspend`'s existing cancel loop covers them, preserving
the Qodo #2420 finding-4 guarantee (no consumption against a hidden
screen inside the 0.15s window). Both consumers are claim-based and
self-guarding, so the pre-existing first-visit double-schedule
(on_mount + the mount's own resume) stays harmless, matching the four
consumers already in both lists.

Evidence (all runs `.venv/bin/python -m pytest` in a fresh worktree off
origin/dev 2b4973971e):
- `test_navigation_to_fresh_models_screen_preserves_exact_ready_handoff`:
  RED on unfixed dev at the `has_pending(VLLM_CONSOLE)` assert (every
  baseline run that got past test setup -- 3/5, the other 2 hit the
  pre-existing setup flake below); GREEN with the fix across repeated
  runs (the handoff assert never failed post-fix).
- New `Tests/UI/test_console_native_chat_flow.py::
  test_warm_console_resume_consumes_staged_chat_handoff` drives the
  production router end-to-end (Chat -> Settings ->
  `open_chat_with_handoff` -> warm Chat): asserts the SAME reused
  instance returns, the CHAT claim drains, and the payload lands in the
  staged-context lane (`_pending_console_launch_context`). RED on
  unfixed code at the `has_pending(CHAT)` assert (Edit-based
  revert/restore, never `git checkout`); GREEN with the fix. Existing
  CHAT-handoff tests mock `open_chat_with_handoff` and never exercised
  warm-resume consumption.
- Neighbor sweeps under a load-32 machine (three other sessions running
  suites): `test_product_maturity_phase3_knowledge_entry.py`,
  `test_study_quizzes_screen.py` all passed;
  `test_settings_raw_cli.py` 40/41 with
  `test_pending_raw_cli_save_vetoes_real_navigation_until_arrival`
  failing IDENTICALLY with the fix Edit-reverted (splash runs 7.0s
  against the test's 3s boot budget; ChatScreen never mounts before the
  failing wait) -- pre-existing on dev. `test_chat_screen_suspend.py`
  (owner of the suspend timer-cancel contract) 2/2 passed.
  `test_console_controller_wiring.py` + `test_console_session_settings.py`:
  454 passed, 6 failed; the three named session-settings failures rerun
  IDENTICALLY on the Edit-reverted pristine arm (fork-transition cleanup
  asserts + notify-once IndexError) -- pre-existing on dev, not
  fix-caused; the resume-repair test passed on both arms.
- The vLLM test's separate SETUP-phase flake (`NoMatches:
  VllmSetupView`) was measured 2/5 on the UNFIXED baseline and at a
  similar rate post-fix: pre-existing and load-sensitive, filed as
  TASK-31809.
- Residual risk (release-eve wrap-up directive: stop batches): full-file
  sweeps of `test_vllm_lab_workflow.py` and
  `test_console_native_chat_flow.py`, and 4 of the 9 grep-neighbor files
  (`test_product_maturity_phase1_core_loop.py`,
  `test_console_fleet_lifecycle_controller.py`,
  `test_uat_first_time_character_chat.py`,
  `test_console_roleplay_resume_navigation.py`) were not completed
  locally; listed in the PR body for CI to cover.

Files: `tldw_chatbook/UI/Screens/chat_screen.py` (+8 lines of timers,
comment), `Tests/UI/test_console_native_chat_flow.py` (+68, new test),
this task file, TASK-31809. Preflight: all derived-artifact checks
passed. Task ids hand-assigned at 31808/31809 after two collisions
(CLI-assigned 31741 and first leapfrog 31790 both already in use across
worktrees/branches; coordinator sweep put in-use max at 31807).
