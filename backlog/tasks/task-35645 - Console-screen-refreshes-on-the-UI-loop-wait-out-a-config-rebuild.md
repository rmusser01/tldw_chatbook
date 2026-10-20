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
- [x] #2 No Console refresh on the UI loop waits for a config rebuild, shown by a test that holds the rebuild lock across a full UI sync including a transcript row reconcile
- [x] #3 Reads that a send or other action relies on still see the configuration the owner's decision requires
- [ ] #4 The response-rules phase test passes 20 consecutive pull-request runs
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the owner's decision (2026-10-20: defer screen by screen).
2. List every UI-thread config wait on the full UI sync in one run, not one per failure: a scratch test replaces the config lock accessors with a recording proxy and forces the caches cold, with the hold starting at the sync's entry or at its transcript step.
3. Move each listed read onto owned display data or the existing deferral; action reads stay live.
4. Pin it: hold a real rebuild across the full sync from its start and from its transcript step, with seeded rows that reconcile under the hold; check each fix's revert fails the pin.
5. After merge, count 20 consecutive passes of the phase test in CI.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

2026-10-20 UTC. Owner decision (AC #1, 2026-10-20): defer screen by screen, as TASK-34415 did. Config reads keep their waiting semantics; only UI-loop refreshes change.

Finding: one diagnostic run listed every UI-thread wait on the full UI sync (scratch test, not committed). It patched config._settings_rebuild_lock and _config_file_lock with a proxy that recorded every blocking acquire from the main thread and refused non-blocking probes as if another thread held them. It also forced _settings_cache_hit and _warm_config_cache_hit to miss for the window, so each read took the path a real hold blocks on. With seeded rows and the hold from the sync's entry: 14 waits at 2 sites. With the hold from its transcript step: 7 waits, the same 2 sites. Both sites were the reads the description named:
- the full sync's core-state push (_sync_console_chat_core_state -> _build_console_provider_selection -> _provider_readiness_app_config -> load_settings), in this sync and again in the whole-sync replay that an earlier deferral starts;
- the transcript's row reconcile (_reconcile_rows -> _turn_file_cards_enabled -> get_cli_setting), from both the sync's transcript step and the transcript's own refresh that set_messages queues.
After the fix the same run listed 0 waits in both cases.

Fix:
- config_sync.sync_console_core_state: the full sync's push goes through run_console_presentation_sync (checked display config), like the rail-visibility pass; a deferral hands the whole sync to the existing replay and the rest of the pass still runs, so the session tabs publish first (the AC10 order the sync keeps). chat_screen.py changes by one line, net 0 (24,330, its budget).
- ConsoleTranscript._turn_file_cards_enabled: when config_sync.config_read_would_wait() (the CLI config cache is cold and another thread holds the config locks, probed without blocking), it keeps the value it last rendered with (the setting's default before the first read). It still reads through the module-level get_cli_setting, the patch point of test_console_turn_file_card_factory.py.

AC #3: a send pushes the live selection itself. The prompt queue's controller accessor is screen._ensure_console_chat_controller() (wiring.py), which runs _sync_console_chat_core_state on every call, so deferring the full sync's push never changes what a send acts on. The earlier attempt at this push (reverted in TASK-35444) probed the native locks, which ordinary guarded reads hold briefly all the time; this uses the checked display data, and the local-probe test that attempt broke passes 5 of 5.

Pin (AC #2): test_console_syncs_never_wait_for_a_config_rebuild gains [full-ui-held], with the rebuild held before the whole sync starts, and both full-ui cases seed rows and clear the refresh key so the rows reconcile under the hold ([full-ui] asserts a reconcile ran during it). Queued refreshes get loop turns inside the hold, and after release any whole-sync replay must finish and the seeded rows render. On dev 1bbc145ffc [full-ui] failed at the reconcile's get_cli_setting and [full-ui-held] at the core-state push; here 3 of 3 runs pass; reverting either fix fails both cases at that fix's read. Census comment updated; floor unchanged.

Evidence: phase test (all 4 variants) 15 of 15 runs; local-probe test 5 of 5; the two turn-file-card tests pass under a private profile (scratch wrapper) here and on dev; the Console suites (119 files, -n 8) pass whole here, 719 of 719; in the same comparison dev failed two of them (test_a_default_changed_after_start_chatting_names_the_model_in_use, test_resend_poll_survives_initial_hook_admission_read), which then passed 3 of 3 alone on dev. Lint findings are unchanged from dev.

Open: AC #4 needs 20 consecutive PR runs after merge.

Files: tldw_chatbook/UI/Console_Modules/config_sync.py, tldw_chatbook/UI/Screens/chat_screen.py, tldw_chatbook/Widgets/Console/console_transcript.py, Tests/UI/test_console_response_rules.py, scripts/ui_pr_gate_census.txt, backlog/docs/lessons-testing-evidence.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
