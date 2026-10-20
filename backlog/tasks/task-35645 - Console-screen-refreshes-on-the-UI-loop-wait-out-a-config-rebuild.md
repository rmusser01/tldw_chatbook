---
id: TASK-35645
title: Console screen refreshes on the UI loop wait out a config rebuild
status: In Progress
created_date: 2026-10-20 00:23
dependencies:
- TASK-35444
updated_date: '2026-10-20 15:26'
labels:
- console
- performance
- flaky
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
While a config rebuild holds the config locks (after a save or reload invalidates the cache), any Console refresh on the UI loop that reads config on a cold cache waits for the rebuild to finish, and the whole app stops responding for that time. A real rebuild takes milliseconds, but the response-rules phase test holds one for seconds to prove the UI never waits, and it fails in CI whenever a background refresh reads config during that hold (TASK-35444: 7 of 31 runs on 2026-10-19). TASK-34415 made the control refresh defer instead of waiting, and TASK-35444 moved three more refresh reads onto that deferral. Found one at a time, more remain: the transcript's turn-file-cards setting (get_cli_setting in ConsoleTranscript._reconcile_rows) and the full UI sync's core-state push at its start. Each fix has exposed the next read. The general approach is an owner decision: defer screen by screen, as TASK-34415 did, or have config reads during a rebuild return the last loaded config instead of waiting, which also changes what action reads such as a send see during a rebuild.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Which approach to take (screen-by-screen deferral or reads that do not wait) is decided by the owner and recorded
- [ ] #2 No Console refresh on the UI loop waits for a config rebuild, shown by a test that holds the rebuild lock across a full UI sync including a transcript row reconcile
- [x] #3 Reads that a send or other action relies on still see the configuration the owner's decision requires
- [ ] #4 The response-rules phase test passes 20 consecutive pull-request runs
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the owner's decision (2026-10-20: defer screen by screen).
2. List every UI-thread config wait on the full UI sync in one run, not one per failure: a scratch test replaces the config lock accessors with a recording proxy and forces the caches cold, with the hold starting at the sync's entry or at its transcript step.
3. Move each listed read onto owned display data or the existing deferral; action reads stay live. Repeat the listing over the other UI-loop refresh entry points (screen resume, credential poll, control bar, mode bar, guidance callers).
4. Pin it: hold a real rebuild across the full sync from its start and from its transcript step, with seeded rows that reconcile under the hold; check each fix's revert fails the pin.
5. After merge, count 20 consecutive passes of the phase test in CI.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

2026-10-20 UTC. Owner decision (AC #1, 2026-10-20): defer screen by screen, as TASK-34415 did. Config reads keep their waiting semantics; only UI-loop refreshes change.

Finding: a scratch diagnostic (not committed) listed every UI-thread config wait in one run. It patched config._settings_rebuild_lock and _config_file_lock with a proxy that recorded every blocking acquire from the main thread and refused non-blocking probes as if another thread held them, and it forced _settings_cache_hit and _warm_config_cache_hit to miss for the window, so each read took the path a real hold blocks on. Three sites waited:
- the full sync's core-state push (_sync_console_chat_core_state -> _build_console_provider_selection -> _provider_readiness_app_config -> load_settings), in the sync and again in the whole-sync replay that an earlier deferral starts;
- the transcript's row reconcile (_reconcile_rows -> _turn_file_cards_enabled -> get_cli_setting), from both the sync's transcript step and the transcript's own refresh that set_messages queues;
- _sync_console_transcript_guidance -> _active_console_settings_readiness -> provider selection, from on_screen_resume (the attach reconcile calls it) and seven other callers: the first-send record, guidance dismissal, local-server discovery, the transcript sync, selection clearing, the control bar and the model switcher. The first diagnostic missed this one: the attach reconcile runs once, before its window opened. The branch's pin then caught it in 1 of 10 whole-file runs, and a second diagnostic that calls seven refresh entry points inside the window listed it (dev: 35 waits at 5 sites).
After the fix both diagnostics list 0 waits.

Fix:
- config_sync.sync_console_core_state: the full sync's push stays the live read it was, unless config.config_read_would_wait() (a cold settings or CLI config cache while another thread holds the config locks, probed without blocking). Then it hands the whole sync to the existing replay, and the rest of the pass still runs, so the session tabs publish first (the AC10 order the sync keeps). A first version rendered the push from the checked projection instead. That added a post-check retry to every full sync, and under a 4-way CPU load an instrumented mapped-learning send left Stop hidden in 1 of 20 runs (dev 0 of 20, with the same retries from other syncs), so the push went back to its live read.
- ConsoleTranscript._turn_file_cards_enabled: when config.config_read_would_wait(), it keeps the value it last rendered with (the setting's default before the first read). It still reads through the module-level get_cli_setting, the patch point of test_console_turn_file_card_factory.py.
- ChatScreen._sync_console_transcript_guidance is decorated with spend.console_readiness_presentation, as _sync_console_mode_bar already is, so every caller renders from the checked projection; when it is cold the refresh is skipped and the projection's own refresh republishes through a full sync. Where the caller already renders under run_console_config_sync's native locks (its live fallback), the decorator now passes through to a live read, which cannot wait there and matches the rest of that pass; _sync_console_mode_bar gets the same. The transcript sync's TASK-35444 wrapper around that call became a plain call. chat_screen.py ends at 24,329 lines, under its budget of 24,330.

A fourth wait sits outside the syncs: the command popup's skill-candidate discovery (a console-skill-discovery worker that the mount and the attach reconcile's resume start) builds the local skill trust service the first time, and that build reads config on the UI loop. The pin caught it in 3 of 22 whole-file runs, when its hold began before that first discovery ended. The mount schedules about 25 refreshes that nobody has listed for config reads, so this is filed as TASK-35836 rather than fixed here, and AC #2 stays open until it lands.

AC #3: a send pushes the live selection itself. The prompt queue's controller accessor is screen._ensure_console_chat_controller() (wiring.py), which runs _sync_console_chat_core_state on every call, so deferring the full sync's push never changes what a send acts on. The earlier attempt at this push (reverted in TASK-35444) probed only the native locks, which ordinary guarded reads hold briefly all the time, so it deferred on most passes; this also needs a cold cache, which only a write or a rebuild leaves, and the local-probe test that attempt broke passes 5 of 5.

Pin (AC #2): test_console_syncs_never_wait_for_a_config_rebuild now has three cases. [recovery-and-resume] holds the rebuild, then runs the TASK-35444 recovery sync and on_screen_resume (one mount for both, to keep UI Fast Lane job 4's added time to one case). [full-ui-held] holds it before the whole sync starts. [full-ui] starts it at the transcript step, and asserts that rows reconciled during the hold. Both full-ui cases seed rows and clear the refresh key, give queued refreshes loop turns inside the hold, and after release require any whole-sync replay to finish and the rows to render. Before the hold, an ordinary deferral can end [full-ui]'s sync ahead of its transcript step. Dev's own copy of the pin failed that way in 1 of 8 whole-file runs, and so did this branch's, 2 of 8. So [full-ui] retries, up to 5 times, until its own sync reaches the step. Every case first waits for the app's own sync and replay, the attach reconcile and any skill discovery to end (TASK-35836). On dev 1bbc145ffc all three cases fail. On current dev 1ffc1ef0d6 too. Reverting any one fix fails at least one case, each blocked at that fix's read; which cases reach it varies with timing (the file-card revert failed both full-ui cases in one run and [full-ui] alone in another; the decorator revert failed two cases, then all three). Census comment updated; floor unchanged.

Evidence on dev: in 18 interleaved whole-file runs of the response-rules file, dev's phase test failed once ([rebuild]), blocked at the core-state push this fixes; dev's copy of the TASK-35444 pin failed once (the setup race above); test_workspace_command_opens_same_manager_in_registered_scope failed once (dev-flaky since TASK-35444). test_mapped_learning_phase_and_stop_follow_live_helper failed at its Stop-shown assertion (one pilot.pause() after the rule helper starts) in 1 of 30 whole-file dev runs and 4 of 64 here. An instrumented send under a 4-way CPU load left Stop hidden 0 of 40 times on dev and 2 of 40 here, each time behind three post-check retries of the checked display, which dev's sends also show. Neither difference is significant (Fisher's exact test, p > 0.4), the signature is dev's, and the test is TASK-34789's known flake. Watch it in CI. Here: on the final code, rebased onto dev 1ffc1ef0d6: the whole response-rules file passes 8 of 8 runs (4-way parallel); the Console suites (119 files) pass whole, 719 of 719; the local-probe test passes 5 of 5; the two turn-file-card tests pass under a private profile; the pin passes on every run and fails on dev. Lint findings and the size ratchets match dev (chat_screen.py 24,329 of 24,330; console_transcript.py unchanged at 8,652).

Open: AC #4 needs 20 consecutive PR runs after merge.

Files: tldw_chatbook/config.py, tldw_chatbook/UI/Console_Modules/config_sync.py, tldw_chatbook/UI/Console_Modules/console_spend_projection.py, tldw_chatbook/UI/Screens/chat_screen.py, tldw_chatbook/Widgets/Console/console_transcript.py, Tests/UI/test_console_response_rules.py, scripts/ui_pr_gate_census.txt, backlog/docs/lessons-testing-evidence.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
