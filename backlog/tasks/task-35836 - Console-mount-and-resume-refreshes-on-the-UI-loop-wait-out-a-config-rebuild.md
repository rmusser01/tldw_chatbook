---
id: TASK-35836
title: Console mount and resume refreshes on the UI loop wait out a config rebuild
status: Done
created_date: 2026-10-20 19:16
dependencies:
- TASK-35645
labels:
- console
- performance
- flaky
priority: medium
updated_date: '2026-10-21 00:27'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
While a config rebuild holds the config locks, a Console refresh that runs at mount or on a screen resume and reads config on a cold cache waits for the rebuild on the UI loop. TASK-35645's pin caught one: the command popup's skill-candidate discovery (a console-skill-discovery worker started at mount and on some resumes) built the local skill trust service, which read config, while the pin held a rebuild (2 of 10 whole-file runs under 4-way parallel load). The mount path schedules about 25 refreshes and none has been listed for config reads. TASK-35645 kept the full UI sync, its transcript reconcile and seven post-mount refresh entry points from waiting, and its pin waits for mount-time skill discovery to finish before holding.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every refresh the Console mount and screen resume schedule is listed with whether it reads config on the UI loop, using the recording-lock diagnostic from TASK-35645
- [x] #2 No listed refresh waits for a config rebuild, shown by a test that holds the rebuild lock from the start of the Console mount
- [x] #3 Action reads stay live, as TASK-35645 decided
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run the TASK-35645 recording-lock diagnostic with its window opened at the start of ChatScreen.on_mount and held until the screen settles, then again across a real screen resume (push and pop a screen), listing every UI-thread config wait by site.
2. Record each refresh the mount and resume schedule with whether it read config on the UI loop.
3. Move each waiting read onto owned display data or a deferral, as TASK-35645 did; action reads stay live.
4. Pin it: a test that holds the rebuild lock from the start of on_mount and across a resume, failing on dev and when each fix is reverted.
5. Re-run the TASK-35645 pin without its skill-discovery settle, and drop that wait if it passes.
Repair plan (2026-10-31): rebase onto dev 198eca8b93, preserve the entire census prefix and append four new nodes; pin stale deferred arguments and the nonblocking native-lock lifetime, then apply the smallest correction. Retain skill discovery owner/source boundaries and run only response-rule, discovery, readiness, rail-width, census and size-ratchet controls.

ADR required: no
ADR path: N/A
Reason: Routine repair implementing TASK-35645's existing screen-by-screen deferral decision, retaining action reads and native ownership boundaries (ADR-230/231).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

2026-10-20/21 UTC. The TASK-35645 recording-lock diagnostic was rerun with its window opened at the start of ChatScreen.on_mount for 3 s, then again for 3 s across a real screen resume (a Screen pushed over the Console and popped). Caches were forced cold and non-blocking lock probes refused, as in TASK-35645.

AC #1, what the mount and a resume schedule, and what read config on the UI loop (dev 2d071d6d71: 111 waits at 19 call paths in the mount window, 0 in the resume window):
- on_mount schedules the attach reconcile. Its first full sync dispatches the roleplay refresh, which ran _sync_console_identity_surfaces: the core-state push, settings summary, rail system line and control bar, with live provider readiness, settings readiness and provider/model display reads. **Read config.**
- The mount's first layout fires Resize, and _adapt_console_workspace_to_width rebuilt the rail and inspector state live (4 builders). **Read config.**
- _start_console_view_after_reconciliation (after the attach reconcile) starts skill discovery. Its first trust-service build reads config. **Read config.** It also schedules the dictation availability sync, H3 image-edit reconcile, a full UI sync, workbench focus restores, collapsible-state restore, the task-resume sync, the credential and environment poll intervals, the fleet survivor tick, the character-navigation focus restore, and the watchlist follow-up. None of these read config in the window. Its eight 0.15 s handoff consumers (launch, chat handoff, conversation resume, prompt insert, settings return, vLLM and llama.cpp intents, fleet completion) found nothing pending. When one is pending, applying it is a user action, and action reads stay live.
- on_screen_resume schedules the hook refresh, workbench and character focus restores, character-view resume work, skill discovery (its trust service is built by then), the same handoff consumers, and the resume navigation startup (only when a target is pending). Its direct syncs are the guidance (TASK-35645's decorator), status-row position, background effects, cost chip, registry reconcile and roleplay dispatch. **None read config in the window.**
- The remaining 4 waits were the test fixture's own _ensure_console_chat_controller() call, an action read, live by design.

Fix:
- console_spend_projection.retry_after_config_rebuild, a decorator. While config.config_read_would_wait() (TASK-35645), the call runs 0.2 s later instead, one timer per refresh, and the latest call's arguments win. A retry on a screen that has started closing renders nothing. It is applied to ChatScreen._sync_console_identity_surfaces and _adapt_console_workspace_to_width (under @on(Resize)). The resize docstring was rewrapped, so chat_screen.py stays at its budget (24,330).
- ConsoleSkillController._refresh_console_skill_candidates, a worker, captures its discovery owner, then awaits asyncio.sleep(0.2) while config_read_would_wait(). Its first trust-service build never waits on the UI loop, and a result from before an attach change is still dropped.
After the fix the diagnostic lists only the fixture's action read.

AC #2: test_console_mount_refreshes_never_wait_for_a_config_rebuild[mount|view-start|roleplay]. A real rebuild starts as the mount reaches one step: on_mount, the view start, or the first roleplay refresh. It holds the config for about 0.75 s, and a 5 s watchdog records any wait. The hold lasts at least 0.75 s, and up to 2 s until the case's deferral is seen. The watchdog is 5 s because, from on_mount, the rest of startup runs inside the hold, and at 2 s it fired with no wait under 4-way load. Each case asserts its refresh deferred during the hold: the resize or identity retry is pending, or, for [view-start], discovery is waiting. In every case discovery fetches nothing during the hold. After release, every retry must run, discovery must fetch unless it already had before the hold began, and the width band must be set. That is deferred, not dropped: a mutation that drops instead of retrying fails [mount], [roleplay] and the unit test test_a_deferred_refresh_runs_once_with_its_latest_call. The unit test drives the decorator against a fake timer: defer, retry while waiting, run once with the latest arguments, nothing for a closing screen. A rebuild from the start of the mount defers the full sync, and the view start and roleplay refresh only follow its completion. So a hold from on_mount alone passed with two of the three fixes reverted, and each read got its own start. On dev 2d071d6d71 all three cases fail. Reverting one fix fails the case that reaches its read: the identity decorator [roleplay] (and [mount] in one of three runs), the resize decorator [mount], the skill wait [view-start] and [roleplay] (and [mount] in one of three).

AC #3: the retries cover only display refreshes and the popup's candidate list. A send still pushes the live selection through _ensure_console_chat_controller (TASK-35645). A settings save's identity-surfaces refresh defers only while another thread is mid-rebuild, and with free locks it reads live as before.

TASK-35645's pin no longer waits for skill discovery before holding.

Census: four node ids appended, the three mount cases (jobs 9, 10 and 1) and the unit test (job 2); floor 408 -> 412.

Evidence: on the final code: the whole response-rules file passes 8 of 8 runs for the new tests (4-way parallel; two runs failed only test_mapped_learning_phase_and_stop_follow_live_helper[size0], TASK-34789's known flake). Under a 4-way CPU load, an instrumented send left Stop hidden 2 of 20 times on dev and 1 of 20 here, and config_read_would_wait() never returned True during a send on either tree. The Console suites (119 files) pass whole, 719 of 719. Lint findings and both size ratchets match dev (chat_screen.py 24,330, its budget).

Files: tldw_chatbook/UI/Console_Modules/console_spend_projection.py, tldw_chatbook/UI/Console_Modules/skill.py, tldw_chatbook/UI/Screens/chat_screen.py, Tests/UI/test_console_response_rules.py, scripts/ui_pr_gate_census.txt, scripts/check_ui_pr_gate_census.py, backlog/docs/lessons-testing-evidence.md.
2026-10-31 repair: rebased onto dev `198eca8b93a04b9399ac8d26f310833a82a959a0`, retaining its complete census prefix and lessons. Four original nodes plus the review-discovered stock capture regression give 1,178 entries. The screen row now records the measured 24,336 lines / 737 methods under the user's explicit size-limit exception.

The retry retires stale queued arguments when a newer call renders, and retains both native config locks through the synchronous render. Deterministic tests first reproduced stale replay and rendering during a raced lock hold. Independent review identified a third race: a rebuild could invalidate stock skill preparation's captured cache and cause a synchronous fallback. Both config locks now surround only the synchronous source capture; contention retries asynchronously, and locks are released before awaiting native preparation. A real stock App/ChatScreen test reproduced the blocked fallback stack before the fix and passes after it; all seven skill discovery boundary controls also pass. The new ExitStack dependency is included in the original-source witness.

The old mount pin now observes original callback entries with local Python monitoring events instead of replacing source-checked methods. Broader mounted verification remains in progress. ADR required: no new ADR; this is a routine lock-lifetime repair preserving TASK-35645's decision and ADR-230/231 ownership and callback affinity.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

The Console mount's three config-reading refreshes (identity surfaces via the roleplay refresh, the width rail rules, skill discovery) now retry or wait asynchronously while a rebuild holds a cold config; a three-start pin fails on dev and per reverted fix.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
