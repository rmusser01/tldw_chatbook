---
id: TASK-40321
title: "The shared UI test app factory gives the app a SimpleNamespace runtime policy, so skill discovery takes a different path than in the shipped app"
status: Done
assignee:
- '@claude'
created_date: 2026-11-12 19:36
updated_date: 2026-11-13 06:51
labels:
- follow-up
- tests
- console
- harness
priority: medium
references:
- Tests/UI/app_factory.py
- tldw_chatbook/runtime_policy/bootstrap.py
- tldw_chatbook/UI/Console_Modules/skill.py
dependencies:
- TASK-38948
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (send stream, TASK-38948: its 100 ms harness had to take a custom path because of this). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** Every app_factory-based UI test runs a skill-discovery path the shipped app does not. Off-loop fixes such as TASK-38948's skill discovery are therefore invisible to latency tests built on this factory, and a test can pass or fail for reasons that do not exist in production.

**Evidence.** `Tests/UI/app_factory.py:375-382` `fake_runtime_policy` builds `SimpleNamespace(state=RuntimeSourceState(...), persist=lambda: None)` and patches it over `tldw_chatbook.app.load_runtime_policy_for_app` (`:461-462`), from 913ec3101c (task-1458). The real type is `runtime_policy/bootstrap.py:122` `RuntimePolicyContext` (`__slots__`, constructor `(state, store, *, publish=None)`). Skill discovery's stock-source check needs `type(runtime_policy) is context_type` (`UI/Console_Modules/skill.py:482`, `_CONTEXT_CLASSES` at `:519-528`), so under the fake, discovery stays on the UI loop.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The shared factory gives the app a real RuntimePolicyContext backed by a sandboxed store, so skill discovery takes the shipped route
- [x] #2 A test shows discovery runs off the UI loop under the factory and fails if the factory goes back to a stand-in object
- [x] #3 The UI census files that build on the factory still pass (or each red is classified)
<!-- AC:END -->

## Implementation Plan

1. Pin the off-loop route first: a factory test that mounts Console under the eager task factory and records the threads where `LocalSkillsService.get_context` runs. It should be red with the stand-in.
2. Build a real `RuntimePolicyContext` in `fake_runtime_policy`, with the same state, a `RuntimeSourceStateStore` in the app's sandbox directory, and the app's projection publisher. Move the tests that assigned `runtime_policy.state` onto a committing helper.
3. Run a broad factory-based sample of Tests/UI and Tests/Chat at `-n 4` on base and head, diff the failure sets, and classify each difference.

## Implementation Notes

**Change.** `Tests/UI/app_factory.py`'s `fake_runtime_policy` now builds a real `RuntimePolicyContext`. It has the stand-in's state (`active_source="local"`, `server_configured=True`), a `RuntimeSourceStateStore` at `<app sandbox>/runtime_policy.json`, and `publish=app._publish_runtime_policy_projection`. Construction reads no file. A commit writes only inside the sandbox.
- The real `state` is read-only, so the 11 sites in 8 factory-based files that assigned `app.runtime_policy.state` now call the new `set_runtime_source_state(app, state)`. It commits through `commit_state`, so it also publishes the projection, as the app does; the six study sites published by hand right after.
- Product source switches (`set_authoritative_runtime_source`) on a factory app now commit to the sandbox store. On the stand-in they raised (no `snapshot()`) and were refused.

**AC#2.** New `Tests/UI/test_app_factory_cleanup.py::test_factory_console_skill_discovery_reads_off_the_ui_loop`. It mounts a factory app under the eager task factory (what `App.run_async` installs) and records, with `sys.monitoring`, the threads that run `LocalSkillsService.get_context`.
- Red with the stand-in: "the factory app read skills on the UI loop" (`8350506368 not in [8350506368]`). Green with the real context.
- The file is a whole-file target in the admission-sensitive PR lane (`derived-artifacts.yml`), so it needs no census edit. It passes 15 of 15.

**AC#3: broad sample, `-n 4`.** 95 targets, 1,961 tests:
- the 23 factory-based Tests/UI and Tests/Chat files that touch the runtime policy;
- 24 Console skill files;
- 40 random whole-file census entries (seed 40321);
- 13 runtime-policy nodes from the 5 files over 6,000 lines.

Results (load 33-87):
- Head arm (3e03be9be7, 2 h 06 min): 459 failed, 1,408 passed, 1 error.
- Base arm (26b2e677dd, 1 h 26 min, `--timeout-method=thread`): 457 failed, 1,409 passed, 1 error.
- Both arms stalled on the same pre-existing hang, `Tests/Chat/test_console_ordinary_tree.py::test_original_stop_hook_holds_same_ordinary_tree`. That worker made no progress for over an hour while the others drained; neither `--timeout=300` method ended it. Each run was interrupted (SIGINT) for its summary, so that worker's queue (about 90 tests) ran in neither arm.
- 452 failures are shared. Most are the local RecoveryRequired at `build_test_app_config` -> `load_settings` in plain tests; the Console popup, visit-dispatch and wiring ones are TASK-40323's.

**The 14 differing nodes,** each re-run 3 times on both trees:
- Red on both: `test_console_send_draft_snapshot.py::test_console_enter_schedules_visible_send_without_a_watchdog` (3 of 3 each, `assert None is not None` at :1370), `test_console_roleplay_resume_navigation.py::test_mounted_resume_never_focuses_setup_modal_before_final_opener` (3 of 3 head, 2 of 3 base; its private-profile child fails), `test_destination_shells.py::test_watchlists_collections_loading_times_out_to_recovery_copy` (3 of 3 each, RecoveryRequired), and `test_library_notes_move_never_false_conflicts.py::test_a_genuine_conflict_offers_a_working_refresh` (2 of 3 each, focus timeout).
- Pause-count navigation pins in `test_screen_navigation.py` (`150 x pilot.pause(0.02)` to land on Library or Study): `deep_link`, `media_route_round_trips`, `search_route_lands_on_library_rag_canvas`, `search_route_round_trips_to_the_library_rag_row`, `search_all_palette`, `study_screen_escape`. They fail 1-3 of 3 on both trees.
  - The rag-row round trip came out lopsided (9 of 9 head, 4 of 9 base), so it was re-run at head with the factory temporarily back on the stand-in: 5 of 6 failed, against 2 of 6 with the real context. The factory is not the cause.
- Green on both, 0 of 3 each, so one-off flakes in the long run: `test_console_command_origin_chat.py::...[skills]`, `test_console_mcp_approval.py::test_request_mcp_approvals_snapshot_covers_exactly_the_unique_names`, `test_library_ingest_tilde_path_and_enter_waits.py::test_a_screen_suspend_and_resume_drops_a_waiting_submit`, and `test_roleplay_hostile_names.py::...[[b]x]`. The second and third are census nodes.

No difference comes from the factory change. The popup and visit-dispatch reds that TASK-40323 exposed fail the same way with the stand-in.

**Files.** `Tests/UI/app_factory.py`, `Tests/UI/test_app_factory_cleanup.py`, `Tests/UI/test_study_quizzes_screen.py`, `test_study_flashcards_real_service_contract.py`, `test_settings_configuration_hub.py`, `test_study_dashboard.py`, `test_product_maturity_phase3_source_study_generation.py`, `test_study_flashcards_screen.py`, `test_study_screen.py`, `test_home_screen.py`.
