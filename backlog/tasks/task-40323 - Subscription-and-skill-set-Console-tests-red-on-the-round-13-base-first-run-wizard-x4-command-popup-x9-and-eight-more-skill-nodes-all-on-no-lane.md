---
id: TASK-40323
title: "Subscription- and skill-set Console tests red on the round-13 base: first-run wizard x4, command popup x9 and eight more skill nodes, all on no lane"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-12 19:36
updated_date: 2026-11-13 15:53
labels:
- follow-up
- tests
- console
- dev-red
priority: low
references:
- Tests/Wizards/test_first_run_subscription_readiness.py
- Tests/UI/test_console_command_popup.py
- Tests/Performance/test_console_skill_trust_stock_owned.py
dependencies:
- TASK-38948
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (send stream, TASK-38948's covering runs at `-n 4`, load 23-46). The notes record these at file level only; the nodes are not named.
- Subscription set: 4 in the first-run wizard file (most likely `Tests/Wizards/test_first_run_subscription_readiness.py`, 9 nodes) fail the same way on base ae9f9e8d21. (The 8 Personas nodes belong to TASK-40322.)
- Skill set: 17 fail on base too: `Tests/UI/test_console_command_popup.py` x9 (of 15 functions), visit dispatch x2, controller wiring x1, held pointer `[skill]`, managed received acceptance, and the stock-owned `[send_enter]` / `[send_button]` cases.

**Why.** None of these files is in the UI census or a PR job (`test.yml` runs Tests/UI only on main pushes), so they can stay red indefinitely, and nobody has established whether they are red on CI or only under the local RecoveryRequired trap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each failing node is named in the notes with its failure, and red or green is established on CI as well as locally
- [ ] #2 Each red is fixed or re-pinned with the commit that changed what it pinned; a product defect gets a test that is red before its fix
- [x] #3 Every node of these files that is green runs on a PR lane
<!-- AC:END -->

## Implementation Plan

1. Re-run every candidate file on base 26b2e677dd with the plain command at `-n 4`, and name each red node with its failure.
2. Clear RecoveryRequired where the file builds or reads config (a profile marker), and re-pin stale pins with the commit that moved what they pinned.
3. Diagnose what remains. Fix product defects that are in scope; report the rest.
4. Put every green node on a PR lane: Tests/UI nodes at the census tail, the others in the recovered-suites job.

## Implementation Notes

**Base 26b2e677dd, plain command, `-n 4`, load 55-87:** 29 failed and 65 passed over the seven files. Each node, with its failure and what happened to it:

- `Tests/Wizards/test_first_run_subscription_readiness.py`: 9 of 9 pass. Round 13's four reds (never named) are green here.
- `Tests/UI/test_console_command_popup.py`: 9 of 15 failed with `RecoveryRequired: raw_source_selection_changed`. `_build_test_app` -> `build_test_app_config` -> `load_settings` is refused by the per-test sandbox; CI's plain command takes the same path. The file is now `bootstrap_profile`, and 6 of the 9 then pass.
  - Re-pinned to 0520df15ac (TASK-32954 ships the built-in Character Creator skill): `test_slash_opens_popup_and_typing_filters` (an extra `/skills character-creator` row after the commands) and `test_enter_accepts_and_inserts_without_sending` (the arg-mode list is that skill, so the popup stays open).
  - Still red: `test_enter_with_popup_closed_sends_normally`, "Expected mock to have been awaited". The Send press never reaches `submit_draft`. It is red with the stand-in factory too.
- `Tests/UI/test_console_visit_dispatch_dedupe.py`: both nodes failed with RecoveryRequired (the same factory read). The file is now `bootstrap_profile`, and both are red on assertions:
  - `test_console_first_visit_refreshes_skills_and_task_resume_once`: "Console synced task-resume state 5x for one visit".
  - `test_console_later_resume_still_refreshes`: "Returning to Console must re-sync task-resume state".
- `Tests/UI/test_console_controller_wiring.py::test_character_controller_is_constructed_with_late_bound_screen_edges`: `AttributeError: ... '_character_console_session_defaults'`. Re-pinned to db515a71d3, which wires the character-chat defaults. Green; the file passes 48 of 48.
- `Tests/Performance/test_app_wiring_held_pointer_currentness.py[skill|collections]`: the observer's bare `assert frame.f_globals is vars(reads_source)`. Re-pinned to 064d5cb014 (TASK-39147 moved `run_preparation_read` into `Backup_Recovery.runtime_producer_lifetime`; `console_preparation_reads` re-exports it).
  - `[collections]` is green.
  - `[skill]` is still red at script :138: `source.runtime_reads` still holds a read right after the trust task returns; it is empty 0.5 s later.
- `Tests/Performance/test_console_managed_received_acceptance.py`: `KeyError` on `objects["hook_ready"]`. 077e5bfa09 (ADR-230 decision 3) routes the received snapshot through `HookPermissions.authority_read`. Re-pinned to that and to 064d5cb014.
  - It now reaches its end and fails its 10 s received-custody bound: 11.0 s measured at load 65-70. A trace-settlement thread was opening a ChaChaNotes connection through the private SQLite helper process.
- `Tests/Performance/test_console_skill_trust_stock_owned.py`: 17 of 17 failed. 14 failed at script :584 (the builder was never held): the 064d5cb014 globals pin swallowed in the monitor. Re-pinned. After that, four-wide:
  - 13 to 14 cases pass.
  - `[app_closing]`: its hang is TASK-40324's, now fixed.
  - `[send_enter]` and `[send_button]`: "Cold Send did not reach original configuration SQL". There is no received claim, no custody record and no notice, and the draft stays in the composer: the Send never started while the trust build was held.
  - `[mount]`: "Original stock skill trust setup blocked shared UI loop", the 0.35 s loop-progress gate, at load 64-77.
  - A product crash moves between cases from run to run (local_winner, app_closing, send_enter, cancel): `NoMatches: 'SelectOverlay' on PruneSafeSelect(id='recovery-scope')`. Fixed in review round 1; see below.

**CI.** No CI has run these files since 2026-10-03, when `test.yml`'s last main-push run was cancelled; they are on no PR lane. The lanes below make this PR's CI the first run. AC#1's CI half and AC#2 stay open until that run is read and the reds below are fixed.

**Lanes (AC#3).**
- UI census tail, 48 nodes: the popup's 14 green tests and the wiring file's 34 functions. `MINIMUM_FILES` is 2087; no re-deal. Serial lane times: popup 14 nodes 118.8 s at load 50-56; wiring 34 nodes 130.9 s at load 39-46.
- Recovered-suites job (`derived-artifacts.yml` plus `Tests/CI/test_ci_queue_pressure_contract.py`'s mirror): 16 targets. The wizard file goes in the sandboxed block; `[collections]` and the 14 green stock cases go last (bootstrap_profile; each a private subprocess).
  - Locally, serial, `--storage-lease-guard --timeout=180`: 24 passed in 8.0 min at load 38-44. Stock cases took 25-39 s each, `[collections]` 14.5 s.
  - That job took 5.7-7.3 min of its 20 in five recent PR runs, so it should land near 14-15 min.
- The Tests/CI contract files pass (193).

**Commits.** fde1a3a362 (the three Performance re-pins). This commit: the popup and visit markers, the popup and wiring re-pins, the lanes, and two `lessons-testing-evidence.md` entries.

**Review r14ti-rev1 fixes (2026-11-13).**
- **The `recovery-scope` crash (AC#2) is fixed.** The crash was not where my notes above first put it. Its real seam: once `App._exit` is set (`App.exit`, and first thing in `TldwCli._shutdown`), `Widget.mount_all` mounts nothing. A scope control that composes then gets no `SelectOverlay`, but `Select._on_mount` still runs `_setup_options_renderables`, and so does a later `set_confirmation` -> `set_options`.
  - A Console sync mounting a card during a stock case's `run_test` teardown hit the first of those.
  - `PruneSafeSelect` now treats `App._exit` as terminal, like `_pruning` and `_closing`: it skips the overlay rebuild and the DOM half of `_watch_value`, and `set_options` still stores the options. One gate covers both paths and every `PruneSafeSelect`. A guard inside `set_confirmation` alone would have missed the `_on_mount` crash.
  - New `Tests/UI/test_console_archive_recovery_cards.py::test_a_card_composed_after_app_exit_began_never_crashes_and_keeps_its_options`. It sets `_exit` as the scope control composes, checks no crash, calls `set_confirmation` and checks the options. Red at HEAD with the exact message ("No nodes match 'SelectOverlay' on PruneSafeSelect(id='recovery-scope')"); green after.
  - That file is a whole-file census entry, so no census edit. The 14 stock cases stay on the recovered-suites lane.
- **Per-entry serial times for the 48 census entries.** Measured in one serial run of all 59 entries this branch added (`-p no:xdist --timeout=180 --durations=0 --durations-min=0`, setup + call + teardown per entry, 73 passed in 167 s, load 19-26, 2026-11-13). The first entry in a process carries the shared import and fixture start-up.
  - Command popup (14):
    - `test_popup_show_highlight_accept_hide` 0.3 s
    - `test_slash_opens_popup_and_typing_filters` 4.5 s
    - `test_enter_accepts_and_inserts_without_sending` 5.1 s
    - `test_down_up_navigates_and_tab_accepts` 5.1 s
    - `test_escape_closes_popup_and_keeps_draft` 5.4 s
    - `test_skill_entries_and_skills_arg_mode` 5.7 s
    - `test_popup_does_not_reclaim_workspace_grid_rows` 3.9 s
    - `test_popup_anchors_above_composer_with_real_css` 4.1 s
    - `test_popup_anchor_clears_chips_above_composer_by_default` 4.3 s
    - `test_popup_anchor_clears_composer_with_chips_below` 4.5 s
    - `test_popup_orders_most_used_command_first` 4.6 s
    - `test_popup_order_is_registry_order_with_no_usage_data` 4.3 s
    - `test_command_dispatch_records_usage_token` 4.4 s
    - `test_unknown_command_dispatch_records_nothing` 4.2 s
  - Controller wiring (34):
    - `test_all_six_controllers_are_constructed_with_the_right_classes` 1.4 s
    - `test_all_eighteen_controllers_are_constructed_with_the_right_classes` 1.1 s
    - `test_terminal_controller_is_wired_to_late_bound_app_and_console_edges` 1.4 s
    - `test_selected_console_local_root_accepts_only_ready_status_forms` 0.8 s
    - `test_selected_console_local_root_rejects_every_non_ready_status` 1.5 s
    - `test_selected_console_local_root_rejects_wrong_workspace` 0.4 s
    - `test_selected_console_local_root_rejects_non_local_bindings` 2.2 s
    - `test_raw_cli_worker_adapter_uses_its_own_group` 1.4 s
    - `test_review_selection_controller_is_late_bound_without_sibling_objects` 8.0 s
    - `test_send_price_controller_is_constructed_with_late_bound_screen_edges` 1.5 s
    - `test_draft_spend_refresh_is_wired_with_late_bound_screen_edges` 1.5 s
    - `test_realtime_controller_is_wired_late_bound_with_empty_owned_state` 1.2 s
    - `test_realtime_outgoing_edges_target_controller_after_method_move` 1.2 s
    - `test_fleet_controller_is_constructed_with_late_bound_screen_edges` 1.3 s
    - `test_session_first_chat_edges_are_late_bound_and_presentation_only` 2.5 s
    - `test_session_first_chat_focus_ignores_opaque_token` 2.5 s
    - `test_retrieval_controller_is_constructed_with_late_bound_screen_edges` 1.3 s
    - `test_character_controller_is_constructed_with_late_bound_screen_edges` 1.3 s
    - `test_skill_controller_is_constructed_with_late_bound_screen_edges` 1.2 s
    - `test_controllers_are_built_in_the_documented_order` 1.5 s
    - `test_every_controller_holds_the_same_app_instance_as_the_screen` 1.3 s
    - `test_shared_chat_store_accessor_resolves_to_the_screens_own_store` 1.8 s
    - `test_shared_chat_store_accessor_is_late_bound` 1.3 s
    - `test_composer_accessor_is_late_bound` 5.5 s
    - `test_current_chat_store_accessor_reads_the_live_attribute` 3.9 s
    - `test_workspace_resolves_the_session_sibling_at_call_time` 1.6 s
    - `test_dictation_resolves_the_hands_free_sibling_at_call_time` 1.3 s
    - `test_dictation_session_late_binds_the_app_owned_service_factory` 1.3 s
    - `test_hands_free_reads_dictation_state_through_the_screen_at_call_time` 1.2 s
    - `test_hands_free_auto_speak_edges_are_late_bound_and_mount_safe` 1.2 s
    - `test_prompts_resolves_the_session_sibling_at_call_time` 1.2 s
    - `test_session_verified_adoption_edges_are_late_bound` 9.7 s
    - `test_context_spend_keeps_real_screen_proofs_and_live_module_callbacks` 6.8 s
    - `test_context_spend_re_resolves_runtime_framework_and_checked_timer_entry` 6.8 s

**Review r14ti-rev2 fixes (2026-11-13).**
- **The `_watch_value` half of the exit gate now has a test.** New `test_a_card_whose_scope_label_composes_after_app_exit_began_keeps_its_value`. It sets `_exit` as the scope control's `SelectCurrent` composes, so `#label` never mounts. It then checks there is no crash and that `set_confirmation(replace(confirmation(), selected_scope="chat_only"))` lands on `chat_only`. Red with only the `_watch_value` gate reverted (`NoMatches: '#label' on SelectCurrent`); green after.
- **The round-2 test now flips `_exit` back in a `finally`.**
- **`_retired()` no longer needs an active app.** Outside one it returns False, so the stock behaviour applies; `NoActiveAppError` is imported from `textual.message_pump`, as `chat_screen.py` does. New `test_a_scope_select_value_can_be_set_outside_any_app` is red with the guard reverted (`NoActiveAppError`) and green after. It lives in the archive-cards file, a whole-file census entry, because `Tests/Widgets/test_prune_safe_select.py` is on no lane.
- **Docstrings** (class, `_retired`, `_watch_value`) now name `App._exit`, say `Select.Changed` is not posted while retired, and say Textual never resets `_exit` (only tests clear it). I did not add `App._exit` to `_require_internals`: it is an instance attribute set in `App.__init__`, and a rename would raise `AttributeError` in `_retired` anyway.
- **Serial lane times** for the archive-cards file's three new tests: 0.1 s each, setup + call + teardown. The whole file took 8.05 s at load 28.
- **Runs at `-n 4`, load 25-39.**
  - Archive cards, `test_prune_safe_select.py` and the two Watchlists `PruneSafeSelect` users: 126 passed, 8 failed. The 8 are the Watchlists RecoveryRequired failures already seen with the change reverted.
  - Stock skill-trust file: 15 passed, 2 failed (`[send_enter]`, `[send_button]`, as before).
