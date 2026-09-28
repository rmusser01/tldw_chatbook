---
id: TASK-33002
title: 'Model config P2: one field table and honest copy'
status: Done
assignee: []
created_date: '2026-09-26 11:47'
labels:
  - model-config-redesign
  - phase-2
  - console
  - settings
  - ux
dependencies:
  - TASK-33001
references:
  - 'backlog/docs/spec-2026-09-26-model-config-redesign.md'
  - 'qa/model-config-ux-review-2026-09-26/judge-synthesis.md'
  - 'qa/model-config-ux-review-2026-09-26/verified-claims.md'
  - 'qa/model-config-ux-review-2026-09-26/report.md'
  - 'backlog/decisions/033-settings-commit-models-three-honestly-labeled.md'
  - 'backlog/decisions/095-conversation-owned-console-generation-settings.md'
  - 'Tests/Architecture/test_module_size_ratchet.py'
  - 'Tests/Architecture/test_screen_size_ratchet.py'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 2 of the model-configuration redesign (backlog/docs/spec-2026-09-26-model-config-redesign.md §8; qa/model-config-ux-review-2026-09-26/judge-synthesis.md §4). Ships as one PR. It changes copy and labels only; density and layout come later. It lands after phase 1, for two reasons:
- The field table carries phase 1's field-to-request-key definition.
- The scope copy describes the D1 convergence that phase 1 ships. Before that, the copy would be untrue.

What it closes:
- C1(a): Providers & Models save copy has no scope ('Provider settings saved.' at UI/Screens/settings_screen.py:30215, a toast at :30242-30244, and 'Shared with Console' at :9455).
- C3: the Provider Test result is one ' | '-joined dump of verdict prose and config-key spellings (settings_screen.py:15213-15335, joined at :15325/:15332).
- C7(d): chips print raw provider keys (UI/Screens/chat_screen.py:9826-9831, Chat/console_display_state.py:776), and several shipped keys have no display name (config.py:4120-4151).
- Label drift across four editors, for example 'Think budget' (settings_screen.py:17378, :18695) against 'Budget' (Widgets/Console/console_settings_modal.py:2125).

It keeps ADR-033's State badge and adds the unsaved count. Absorbs TASK-486 (custom-named credential query parameters in Test evidence). TASK-194 (popover display names) stays open: per the spec it closes when the popover rows are rebuilt.

Constraints:
- console_settings_modal.py has zero headroom (module-size ratchet 7,807).
- chat_screen.py must stay within its 25,363-line budget.
- ADR-097 ratchets never rise.
- ADR-066 legacy aliases stay selectable.
- Test copy for cloud providers stays a local readiness check (TASK-30011 AC#2). Changing what Test checks is out of scope.

Carried from phase 1 (TASK-33001 final review, riders with no code in phase 1):
- The field table's option lists: the Reasoning select that Settings now shows for llama.cpp and other local keys offers "minimal", which `build_local_thinking_payload_fields` drops with only a debug log (Chat/console_provider_support.py `_TEMPLATE_SAFE_EFFORTS`). TASK-33001.2 exposed this value-level silent drop.
- The Provider Test result rows (AC#2): the reachable-probe toast still states the generation fact twice in two phrasings ("Live generation has not been tested; ... generation not tested"), and the in-flight line hard-codes "generation not tested" even for an identity whose stored generation test succeeded (TASK-33001.3 review minors 1-2; TASK-33001 AC#3 is met for the literal string only).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every model-configuration field has one label and one help line wherever it is edited
- [x] #2 The Provider Test result reads as labelled rows that lead with the outcome, with no config-key spellings and no leaked secrets
- [x] #3 The Providers & Models save and the State line say what a save applies to
- [x] #4 The Settings State line keeps naming its save model and counts unsaved edits
- [x] #5 Console chips and notices name providers by display name, and one display-name map serves every surface
- [x] #6 No Settings copy refers to 'Console Defaults' or 'Override current Console model'
- [x] #7 Rendered captures at 211x44, plus one at 235x52, of every changed surface are attached to the PR
- [x] #8 The phase changes no layout, CSS, design token or geometry
- [x] #9 console_settings_modal.py does not grow, chat_screen.py stays within its budget, and no ADR-097 ratchet value rises
- [x] #10 TASK-486 is closed as Done
- [x] #11 Docs/User_Guide pages updated (settings.md, console.md); what was verified is recorded in the subtasks' Implementation Notes (was: "including their Verified-against stamps", superseded by dev's CLAUDE.md rule, TASK-33125)
- [x] #12 ./scripts/preflight.sh passes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Phase 2 ships one field table, labelled Test rows, scoped save copy, a State line that keeps its badge and counts unsaved fields, and one provider display-name catalog. It changes copy and labels only: no .tcss, token or geometry change.

**Subtasks and commits** (branch feat/model-config-p2-field-table, plan 4e23a5cb4f):
- TASK-33002.1 field table: 44b2df76a0, fix round 85ea4722f6.
- TASK-33002.2 Test rows (absorbs TASK-486, now Done): ccf34d7b9f, fix round cd6f9f1783.
- TASK-33002.3 save scope: 0c4ab0d890, fix round 780457707d.
- TASK-33002.4 State badge + unsaved count: 442b38422b.
- TASK-33002.5 provider display names: 174f4702fa, fix round 9085477f42.
- TASK-33002.6 renames: b31ceb696b, fix round 41aa49266d.
- Final fix wave (whole-branch review, .superpowers/sdd/plan-2026-09-27-model-config-p2/final-review.md): one `fix(model-config)` commit on top of 41aa49266d.

**Final fix wave.**
- C1: a llama.cpp profile saved with "minimal" crashed Revert and lost the value on an unrelated Save. One helper feeds compose, sync and Revert and keeps the saved level as "minimal (not supported here)". Two mounted regressions, red on the pre-fix code.
- I1: the pristine-convergence swap-notice pin is rewritten to display names.
- I2: Console Behavior's generation fallbacks get a focused-field guide and Control guide rows from the field table.
- I3: the Personas preview names providers through `provider_display_name`.
- I4: the nine branch-added "Verified against" paragraphs are deleted; their text moved into each subtask's notes; facts that lived only in a stamp moved into page content.
- Minors: Console validation names Thinking/Thinking budget from the table; the swap notice reads "not selected" for a provider-less default; the replay-override target line uses the display name; the Model inspector Purpose names the new-chat scope; an invalid draft keeps its badge and count (ruling in TASK-33002.4's notes); pairing comments on the two scope-copy forms; plan Task 6 AC#3 synced with the subtask.

**Delegations (ACs ticked on these terms).**
- AC#1: labels come from the table on all four editors; help lines render in the P&M inspector and Console Behavior's field guide. Rendering help lines inside the Conversation settings modal and the Alt+M popover is layout work the plan puts outside this phase.
- AC#5: the popover's rows stay with TASK-194 (per this task's description). Engine error-copy prefixes are TASK-33002.14. Settings' own custom-endpoint name, and its crash on a custom-endpoint default provider, are TASK-33002.12.
- AC#7: captures are committed under qa/model-config-p2-2026-09-27/task-1..6 and final-fix/, each with at least one 235x52 capture. Attaching them to the PR body is the PR step.
- AC#9: console_settings_modal.py is 7,806 lines against its 7,807 budget (this fixes a dev-red ratchet). The branch adds 0 lines to chat_screen.py (25,218 = budget here). After the rebase, dev's own +39 (task-33081) leaves chat_screen.py at 25,257 against 25,218: red on dev and on the rebase alike, not caused here. Ratchet reds at HEAD are a strict subset of the Phase 1 baseline (7 of 8; console_settings_modal.py now passes).
- AC#11: wording amended to dev's rule, which landed after this AC was written (TASK-33125). The human should confirm the amendment.

**Tests rewritten on purpose** (full lists in each subtask's notes):
- .1: old labels (8 hub tests, 2 modal tests, 1 context-memory test, 5 Console validation tests), plus `test_settings_console_behavior_inspector_explains_visible_controls` again in the final wave.
- .2: the pipe-form pins in the draft, subscription-readiness and hub files.
- .3: the "Provider settings saved." pins.
- .4: `test_state_banner_dirty_branch_keeps_priority`, `test_speech_tts_dirty_banner_names_leave_resolution` and the Video Gen banner pins.
- .5: the 'Provider: llama_cpp' chip pins, `test_stale_default_refresh_swap_is_visible`, and `test_provider_change_posts_the_swap_notice_and_a_model_change_does_not` (final wave).
- .6: the hub "Console Defaults" pins, and `test_reasoning_history_remembers_normalized_target_and_clears_override` (final wave).
- Mounted tests that were env-red under ADR-126 now run under `@private_profile_test`.

**Behaviour changes for the PR description.**
- Field labels unify on Max tokens, Thinking budget, Endpoint, Presence penalty, Frequency penalty, Budget strategy and When limit nears.
- Settings' Reasoning effort list drops levels that llama.cpp-family requests drop. A level saved earlier shows as "<level> (not supported here)".
- The Test result is five labelled rows that lead with the problem. Credential query parameters are never shown.
- The P&M save, toast and State line name the scope. The State line reads "State: <badge> · N unsaved | …", and an invalid draft reads "… | Needs correction: <message>".
- Providers show their catalog display names on the chip, the swap notice, the Settings picker, the First-run list and the Personas preview. Picker relabels: Google -> Google Gemini, MistralAI -> Mistral AI, Moonshot -> Moonshot AI, Custom OpenAI -> Custom OpenAI-compatible, and legacy aliases gain "(legacy alias)".
- Renames: 'Console Defaults' -> 'Console Behavior'; 'Override current Console model' -> 'Reasoning replay override'.

**Rulings needing the human** (review-flagged, not changed here):
- The "(change them in Console with Alt+M)" tail is plan- and spec-mandated. Recommend accepting it.
- chat_screen.py:11477 packs two kwargs on one line to hold the ratchet. `ruff format` would split it. Keep it only if dev pays its +39.
- The AC#11 amendment above.

**Pre-existing reds (not caused here).**
- ADR-126 `RecoveryRequired` env-reds in a clean worktree: 919 of the 941 failures in the covering run.
- `test_picker_caps_visible_providers_at_thirty` (dev b4674cf299 raised the cap to 40; TASK-33002.16).
- The order-flaky `test_provider_switch_syncs_generation_test_action_bidirectionally` (TASK-33002.15).
- The probe-endpoint network-policy tests, the F3/F8 log-key copy tests, `test_settings_pane_widths_are_owned_by_stylesheet_not_inline_python`, `test_settings_advanced_config_new_file_save_reports_no_backup`, two console_internals_decomposition tests and five console_session_settings modal tests. Each is red on HEAD 41aa49266d too.

**Riders filed:** TASK-33002.7 to TASK-33002.17.

**Verification (final wave, 2026-09-27).**
- Covering set: the 17 branch-touched test files, the pristine-convergence and reasoning-history files, test_settings_kimi_zai, test_personas_workbench(_state), test_console_provider_picker, test_settings_raw_draft and test_settings_web_search, run with xdist -n 6.
  - Result: 941 failed / 1,167 passed.
  - The same set on a HEAD export gave 941 failed / 1,159 passed.
  - Failure-name diff: 1 fixed (the I1 pin) and 1 new. The new one is the known order flake (TASK-33002.15), which passes 5 of 5 serially.
- Targeted files pass in full: test_model_config_field_labels (16), test_console_pristine_chat_convergence (19), test_settings_state_line_unsaved_count (7), test_settings_provider_save_scope (3), test_settings_console_reasoning_history (7).
- Mutation checks: the old Revert mapping fails the revert test with InvalidSelectValueError. HEAD's settings_screen.py fails both C1 tests. HEAD's session.py fails the "not selected" test.
- Live drive at 211x44 and 235x52 on a scratch profile (qa/model-config-p2-2026-09-27/final-fix/session-log.txt). The real config.toml SHA-1 is unchanged, and no real-profile file is newer than the launch.
- ruff: counts equal HEAD per file. Format drift is not increased.
- `PYTHON=<venv> ./scripts/preflight.sh`: rc=0.

**Capture follow-up (flags 1, 5, 6, 7 and 10 from qa/model-config-33002-captures).** Live captures found gaps against AC#1 and AC#5. One `fix(model-config)` commit closes them:
- Flag 1: Revert left the Test rows describing the discarded draft ("http://127.0.0.1:9 (draft) · model listing failed…"), and they survived a later Save. Revert now marks the Test result stale in `action_settings_revert_category`. Mounted regression: `test_revert_marks_the_discarded_drafts_test_rows_stale`.
- Flag 5: the field table gains `compaction_target_ratio` ("Reduce context to (%)", which fits the modal's 23-cell column; the old label wrapped there) and `compaction_carry_forward_mode` ("Keep after compaction"). A shared `CARRY_FORWARD_OPTIONS` ("Memory with recent turns / latest exchange", the memory-compaction spec's wording) feeds both editors. The popover's "Response max" row reads the table's "Max tokens" in the same 14-cell column. Console Behavior's Seed and Thinking budget placeholders read `MODEL_PROFILE_INPUT_PLACEHOLDERS`, the P&M copy. The other five Context-view labels already agree and stay literal; they join the table when one drifts.
- Flag 6: the modal's Context scope line names "F4 Settings > Console Behavior", a real category label.
- Flag 7: the Automatic refresh checkboxes and their search-index rows use `provider_display_name` (Mistral AI, Moonshot AI, Z.ai).
- Flag 10: Overview's "Last connection test" appends the Endpoint row to the leading row, so it says whether the endpoint was reached. A cloud Endpoint row with an empty field names the address the field shows ("https://api.openai.com/v1 (provider default)"). `_detail_row` renders plain text, so "[chat_defaults]" and "[console]" no longer vanish as markup ("Scope:  response fallbacks and  paste"); the same fix covers the other four bracketed inspector rows. The model-defaults and context-window "Saved as" rows use the model in the form, not "<model>".
- Ratchets: console_settings_modal.py 7,806 -> 7,802. chat_screen.py is untouched. No CSS, token or geometry change.
- Docs: settings.md covers the Overview Endpoint row, the cloud default address and Revert marking the Test result stale.
- Tests rewritten on purpose: `test_overview_shows_only_the_leading_test_row` became `test_overview_shows_the_leading_test_row_and_the_endpoint_row`. The "Console behavior" and "Reduce conversation to (%)" pins in `test_full_modal_has_stable_views_and_saves_conversation_policy` now read "Console Behavior" and "Reduce context to (%)".

**Verification (capture follow-up, 2026-09-27).**
- RED on the pre-fix surfaces, with the new table rows present so the test module imports: modal path test `{'Console behavior'}`; popover "Response max  unknown tokens…"; modal label "Reduce conversation to (%)"; P&M "Saved as" "...model_defaults.<model>.presence_penalty"; Console Behavior options "Memory + recent turns"; cloud Endpoint "provider default"; the Overview headline without the Endpoint row; after Revert, the Test result still held the draft rows. Assertions hidden behind an earlier failure were checked one mutation at a time on the fixed file: Automatic refresh "MistralAI: refresh" / "Moonshot: refresh" / "ZAI: refresh"; Seed placeholder "optional deterministic seed"; with `markup=False` removed, the Scope row "Scope:  response fallbacks and  paste display settings".
- GREEN: test_model_config_field_labels and test_settings_provider_test_draft pass, apart from the ADR-126 env-red `test_t_hotkey_does_not_run_test_while_input_focused`, which is red at HEAD too. `test_full_modal_has_stable_views_and_saves_conversation_policy` and `test_visual_representation_choices_enable_for_vision_model` pass under a temporary private-profile wrapper (unwrapped they hit `RecoveryRequired`, as at HEAD). The ratchet reds are the same 7 files as at HEAD.
- Covering set: 24 files, with the same list run on this tree and on a HEAD export under xdist -n 6. Fixed tree: 421 failed / 796 passed / 26 errors. HEAD: 419 / 795 / 26. Only two names are red here and not at HEAD. `test_console_settings_modal_saves_replaced_temperature_input` is a pre-existing flake ('0.6' == '0.7'): 2 of 8 wrapped runs fail at HEAD and 2 of 6 on this tree. `test_console_settings_modal_shows_current_catalog_model_provenance` passes 4 of 4 wrapped.
- ruff: per-file counts equal HEAD. Format drift is not increased (settings_screen.py 1,201 -> 1,190 diff lines).
- Live drive at 211x44 and 235x52 on a scratch profile (HOME, XDG_* and TLDW_CONFIG_PATH under the scratchpad; users_name verify_33002_drift5a1; model-listing stub on 127.0.0.1:9198). 24 captures overwrite or join qa/model-config-33002-captures/: the popover fields, the modal Context view and memory rows, P&M test-after-revert, save-result, save-toast, model-defaults and test-cloud, a new settings-pm-automatic-refresh pair, Overview last-connection-test, and the Console Behavior fallbacks and reasoning-replay-override views. The real config.toml SHA-1 is unchanged (194e7c5b82). No real-profile file is newer than the launch. tmux was killed and only the scratch data dir was deleted.
- Re-check before commit (2026-09-28): the new tests are RED again with HEAD's four surface files restored (8 named failures; `test_t_hotkey_does_not_run_test_while_input_focused` is the ADR-126 `RecoveryRequired` red at HEAD) and GREEN on the fix (104 passed). A 617-name failure set from a wider covering run was rerun on a HEAD export: every name is red there too except `test_console_settings_modal_provider_round_trip_ignores_none_model_sentinel`, which also fails 2 of 2 alone at HEAD (order-dependent, not this change).
- `PYTHON=<venv> ./scripts/preflight.sh`: rc=0.
<!-- SECTION:NOTES:END -->
