---
id: TASK-33008
title: 'Phase 8: First run connects in place and always returns to Console'
status: Done
assignee: []
created_date: 2026-09-26 11:47
labels:
- model-config-redesign
- phase-8
- console
- settings
- ux
- readiness
dependencies:
- TASK-33001
- TASK-33004
- TASK-33005
- TASK-33007
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Navigation/pending_handoff_store.py
- tldw_chatbook/UI/Navigation/conversation_settings_navigation.py
- tldw_chatbook/Widgets/Console/console_setup_modal.py
- tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py
- tldw_chatbook/UI/Screens/settings_screen.py
- Docs/User_Guide/console.md
- Docs/User_Guide/settings.md
- Docs/User_Guide/First_Run_Setup.md
- PRODUCT.md
- backlog/decisions/012-provider-credential-settings-boundary.md
- backlog/decisions/033-application-session-state-ownership.md
- backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md
- backlog/decisions/095-conversation-owned-console-generation-settings.md
- backlog/decisions/097-boot-budget-ratchets.md
- qa/model-config-ux-review-2026-09-26/judge-synthesis.md
- qa/model-config-ux-review-2026-09-26/verified-claims.md
priority: high
updated_date: 2026-10-17 17:23
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 8 of the model-configuration redesign (judge-synthesis.md §4 P8; spec §8). It ships as one PR.

Why: verified finding C1's remaining gap is a recovery dead end. When a Console chat needs a key, _open_console_provider_recovery (chat_screen.py:20533-20565) posts NavigateToScreen(TAB_SETTINGS) with the category, provider, model and field, but gives no way back. The user adds the key and then has to find Console again by hand.

A working pattern already exists. The Chat settings modal's own credential detour has a revisioned return: ConversationSettingsReturnIntent is staged on HandoffChannel.CONVERSATION_SETTINGS_RETURN (chat_screen.py:3556-3600), and Settings shows a return continuation (settings_screen.py:16980-17004, handler :28656). ADR-033 (033-application-session-state-ownership.md:13-23) requires handoffs to go through PendingHandoffStore.

First run also has three smaller problems:
- 'Use detected <server>' silently adopts the server's first model (chat_screen.py:15397-15479).
- The Get started card draws a snow field behind itself (ConsoleSetupBackdrop, console_setup_modal.py:113). PRODUCT.md:29 rules out 'gratuitous ASCII decoration'.
- The wizard's model step mounts failures such as 'Authentication failed — this API key was rejected' as disabled radio options (FirstRunSetupWizard.py:3836-3906).

Owner decisions:
- D4 keeps credentials in Settings. ADR-012 (012:10, :31) says recovery must open the exact credential control, so the cloud path is a round trip with a return, not inline key entry.
- D2's non-generating key check (phase 5) is how the user confirms the key before returning.
- D1's pristine-chat convergence (phase 1) makes the returning first-run chat use the new defaults.

The judge's budgets: first run with a local server takes 2 keys; with a cloud key it takes about 11 keys and ends back in Console.

Constraints:
- chat_screen.py has 32 lines of headroom (25,331 against 25,363, Tests/Architecture/test_screen_size_ratchet.py:85), so new logic goes in UI/Console_Modules.
- FirstRunSetupWizard.py already measures 10,462 lines against its 10,404 row (test_module_size_ratchet.py:81). That is a red on dev today, and this phase must not make it worse.
- ConsoleSetupModal stays the documented non-dismissible gate (ADR-031, task-16211 refinement).
- First-run copy may only teach keys that the footer shows (ADR-031 rule 5).

Absorbs TASK-32572 (Tab does not cycle the Get started card's buttons). Its 100x30 capture is replaced by the program's full-screen targets.

Depends on:
- Phase 1: pristine-chat convergence (D1).
- Phase 4: the switcher, with its pick and filtered modes and NEEDS SETUP rows.
- Phase 5: the key check and the in-place guidance for out-of-app failures.
- Phase 7: the Providers & Models card that hosts the return action.

After this phase, task-1379 re-runs the Settings critique.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 First run with a detected local server takes 2 key presses (Enter, Enter) from the Get started card to a ready composer, measured live at 211x44.
- [x] #2 First run with a cloud key takes no more than 12 key actions from the Get started card to a ready composer in the same chat, counting a paste as one. The count includes the key check and the return, and the user never navigates back to Console by hand. Measured live at 211x44.
- [x] #3 Every Console provider-recovery path that leaves for Settings for a provider a return can name (every built-in provider) offers a one-key return to the chat it came from. This closes C1's recovery dead end for those providers. A named custom endpoint (`custom-ep:<slug>`) is out of scope: no return can name one (controller ruling R37). Rescoped by the owner's decision of 2026-10-17 on PR #3144 (D1); TASK-35301 gives custom-endpoint recoveries their return.
- [x] #4 No Console surface accepts or displays an API key; keys are entered only in Settings ▸ Providers & Models (D4, ADR-012).
- [x] #5 chat_screen.py stays within its line and method budget, and FirstRunSetupWizard.py does not grow.
- [x] #6 No ADR-097 ratchet rises; boot CSS bytes go down with the backdrop's removal.
- [x] #7 Keyboard-only live captures of both journeys at 211x44 and 235x52 are attached, taken with the real stylesheet and a scratch TLDW_CONFIG_PATH.
- [x] #8 Docs/User_Guide pages updated where behaviour changed: console.md (Get started), settings.md (return action) and First_Run_Setup.md (model-step status lines); what was verified is recorded in Implementation Notes.
- [x] #9 ./scripts/preflight.sh passes.
- [x] #10 A 403 to a model listing or Test that sent no key still reads "Model listing unavailable" wherever that copy shows, and adds that the server may need a key, naming where that surface takes one: under Authentication on the wizard's Provider step, Back from its Model step, the API key field in Settings ▸ Providers & Models, and Settings from Chat settings' Test connection. A 403 to a request that sent a key, a 404, a 401 and a Claude subscription get no such hint (owner decision D2, 2026-10-17).
- [x] #11 The Console status row's cost chip fits the room the chips before it leave: it shows the longest form whose whole " · " figures fit, never a figure cut at the edge; "…" when no form fits; every figure while it has focus. It refits once, after the layout, when that room changes (a chip before it changes width, a pair applied from Switch model, the chip shown again after layout) (owner decision D4, 2026-10-17).
- [x] #12 A focused cost chip stays in view when its label grows or the row moves; a refit that keeps its label, a same-width relabel and a resize leave the user's own scroll of the row alone; and when focus leaves the chip the row scrolls back to its start (D4).
- [x] #13 Scrolling the Console does not lay out the whole screen to place the cost chip: a scroll keeps the strip's fast path (D4).
- [x] #14 After a save from a Console provider recovery that leaves new chats on the page's provider (a save of its key, Env var, Endpoint or Default model for new chats), the line above Return to Console names the pair new chats now use ("New chats use <Provider> · <model>."), so a key-only save from a Switch model NEEDS SETUP row no longer reads as touching only this chat; the save itself is unchanged (owner decision D5, 2026-10-17).
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Owner decisions of 2026-10-17 on PR #3144's five questions** (branch `feat/model-config-p8-decisions`, base origin/dev `111f8e48fc`):
- D1, AC#3: both, as recommended. (a) AC#3 is rescoped to the providers a return can name, and ticked: the acceptance check below found every built-in provider's recovery with its return. The card's Set up provider, the transcript's action and the prompts hook share `_open_console_provider_recovery`, and with a cloud Retry and the NEEDS SETUP rows they all go through `open_provider_setup`. Pinned in the UI Fast Lane by `Tests/UI/test_console_provider_return.py::test_recovery_round_trip_returns_to_the_chat_on_one_enter` (the card, 211x44, one Enter back to the same chat) and `::test_a_needs_setup_row_round_trip_reopens_switch_model_on_its_provider` (both cases). Outside a PR lane: `::test_a_route_that_names_no_field_arrives_on_the_return` (a Retry-style route) and `Tests/UI/test_console_native_chat_flow.py::test_console_add_api_key_recovery_targets_provider_settings_category`. (b) Filed TASK-35301 for custom-endpoint returns, with the tests and copy that pin today's one-way route.
- D2, a keyless 403 (my pick, as the owner left it): the wording stays "Model listing unavailable…" on every path, and a 403 to a check that sent no key adds a hint, phrased for where each surface takes a key (new AC#10). The Provider step's key field is below its line, not above it as the brief said, so its hint reads "If this server needs a key, add one under Authentication." (the step's own key-required wording). Model has no key field: "If this server needs a key, go Back to add one." Settings' Test result Endpoint row: "…; if this server needs a key, enter one in the API key field". Chat settings' Test connection takes no key (D4): " If this server needs a key, add it in Settings ▸ Providers & Models (F4)."
  - How: a 403 now keeps its cause. `_listing_not_permitted()` records `model_listing_unavailable` with category "forbidden" (it was "http_status", the 404's), and the four validators that pinned `{None, "http_status"}` share `LISTING_UNAVAILABLE_CATEGORIES` (Chat/provider_test_evidence.py, console_session_settings.py). Nothing reads that category to block or to word readiness: every blocker and reason that maps "forbidden" is gated on `unreachable`. `listing_refused_without_a_key` is the Settings and Chat settings rule (the evidence's identity sent no credential); the wizard's `_with_keyless_hint` (first_run_model_discovery.py) judges "sent" as its 401 copy already does: the draft for a listing, the Test identity for a Test. `_model_discovery_ui_outcome` returns "forbidden" as a 403 listing's category, and the Model step takes a Test's category from its evidence; `_handed_off_failure_category` returns a category only for a failure, as before.
  - Claude subscription: excluded. Its keyless listing is refused with a 401, which already reads as unavailable through `subscription_cannot_list`, and its user needs no key, so the hint would be false; `_signs_in_with_subscription` (split out of `subscription_cannot_list`) also excludes a subscription's 403. Settings never sends a keyless key check (a cloud check needs a key, and a subscription gets none), so only its local Test can show the hint.
  - Not changed: readiness words ("Ready · not tested") and Chat settings' summary tile ("Reachable — model listing unavailable") name no cause, and its toast stays short; the Switch model probe shows only readiness words.
  - Tests (each red without the change, the sent-a-key cases red with the key check removed, the subscription case red with its guard removed): `Tests/Wizards/test_first_run_setup_state.py::TestProviderTrustChain::test_the_model_step_hints_at_a_key_only_after_a_keyless_403` (new, PR Fast Lane), `::test_provider_tells_a_listing_that_cannot_list_as_model_does` and `::test_a_test_that_could_not_list_reads_as_the_listing_reads_it` (rewritten on purpose, already in the Fast Lane), `Tests/Wizards/test_first_run_setup_wizard.py::test_a_keyless_403_reads_as_unavailable_with_a_key_hint_on_both_steps` (new, both steps through the listing and through Test; admission step), `Tests/UI/test_settings_provider_test_draft.py::test_a_keyless_403_listing_suggests_a_key_on_the_endpoint_row` and `Tests/UI/test_console_session_settings.py::test_a_keyless_403_to_test_connection_suggests_a_key_in_settings` (new, UI census, floor 322). Each new lane entry passed 3 of 3 serial runs. The keyed 403's recorded category is rewritten on purpose in `test_key_check_listing_maps_to_bounded_evidence`, `test_a_listing_403_converts_to_the_non_blocking_listing_result` and the 403 case of `test_a_rejected_key_reads_rejected_on_the_key_row_too`; `test_a_403_to_a_listing_of_a_sent_key_does_not_reject_it` pins the sent key's unhinted line unchanged.
  - Docs: First_Run_Setup.md (the model-step status lines), settings.md (Test (t)) and console.md (Chat settings' Test connection).
- D3, Claude subscription: the owner confirms the shipped copy. A refused keyless listing reads "Model listing unavailable; you can enter the model ID on the next step." on Provider and "Model listing unavailable; enter the model ID used by this endpoint." on Model, and no failed check is recorded. No code change; D2's hint is not added there. Pinned by `Tests/Wizards/test_first_run_setup_wizard.py::test_a_claude_subscription_is_never_told_to_add_an_api_key` (admission step; passes at this round's head) and, for the hint, the subscription case of `test_the_model_step_hints_at_a_key_only_after_a_keyless_403`.
- D4, the cost chip: the ACs P8 lacked are added as AC#11 to AC#13, written from the cost-chip commits (`git log 1c1730f4b0..111f8e48fc -- '*cost*'`: a01ae059cc, ba63a6b2d6, b554ea6361, 14d23e1d9c, 2a3bc41c12, 02b3e525f8, 589f0edf4a, e2c6898873, 930050411e, c7c279a75b, 9db41ee1fe, 575a97adf3, d29ca254df) and `Widgets/Console/console_status_chips.py`. Each is ticked on named tests in `Tests/Chat/test_console_status_chips_cost.py`, a whole file in the PR Fast Lane; at this round's head the file passes 46 of 46 (serial), every case of each named test among them:
  - AC#11: `test_the_cost_chip_drops_whole_segments_to_fit_the_strip`, `test_the_cost_chip_shows_an_ellipsis_when_no_form_fits`, `test_the_cost_chip_refits_when_a_chip_before_it_changes`, `test_a_pair_applied_under_a_dialog_refits_the_cost_chip`, `test_the_cost_chip_fits_when_it_is_shown_after_layout`, `test_a_change_that_moves_the_cost_chip_refits_it_once`.
  - AC#12: `test_a_focused_cost_chip_relabelled_longer_stays_in_view`, `test_a_focused_cost_chip_in_view_stays_in_view_when_the_row_moves`, `test_a_refit_that_keeps_the_focused_chips_label_keeps_the_scroll`, `test_a_same_width_relabel_does_not_undo_a_later_swipe`, `test_a_capped_cost_chip_does_not_undo_a_swipe_on_resize`, `test_leaving_the_cost_chip_scrolls_the_strip_home`.
  - AC#13: `test_scrolling_beside_the_strip_keeps_the_scroll_fast_path`.
- D5, a key-only save from a NEEDS SETUP route: the copy, not the behaviour (new AC#14). Every Providers & Models save that writes a connection goes through `build_provider_setup_mutation`, which writes `[chat_defaults]` = the page's provider and model; measured on the real journey (Groq's row, a chat that holds work, key only): `chat_defaults` became groq · llama-3.1-8b-instant while the reopened Switch model read "Applies to: this chat only". It is not only a key (the owner asked whether Env keys or other things are affected): an Env var save (the same `CREDENTIAL_SAVED` outcome), an Endpoint save and a Default model for new chats save do it too; a Model defaults field, context window, API mode or Sign in with change alone writes no `[chat_defaults]`.
  - The strings: the switcher has no reopen notice; its scope line (`SCOPE_THIS_CHAT`) and Enter label speak of its own pick, which is true, so they stay. The line that reports the save is Settings' return continuation (`UI/Settings_Modules/provider_return.py::continuation_copy`): "Credential saved. Return to Console to continue in “Chat 1”; provider acceptance is not yet verified." It now names the pair, read the way a Ctrl+T chat reads it (`blank_console_session_settings`, `provider_model_pair`): "Credential saved. New chats use Groq · llama-3.1-8b-instant. Return to Console to continue in “Chat 1”; …". The sentence shows after any Console return's save that leaves new chats on the page's provider, so an Endpoint save says it too, and is left out when the default is another provider's (a Model defaults change for a provider that is not the default). Chat settings' two strings are unchanged. The save's own result row already says "new chats and open chats nobody has used yet take them" (`PROVIDER_SAVED_COPY`), without the pair.
  - Tests (red on the old copy first): `Tests/UI/test_console_provider_return.py::test_settings_copy_names_the_console_chat_and_keeps_the_modal_copy` (the copy's cases, including a default on another provider and one with no model) and `::test_a_needs_setup_row_round_trip_reopens_switch_model_on_its_provider[saves-a-key]` (the real journey: the saved `[chat_defaults]` and the line), both already in the UI census; `::test_recovery_round_trip_returns_to_the_chat_on_one_enter` (census) and `::test_a_key_check_after_the_save_updates_the_return_line` are rewritten on purpose to the new line. The four passed 3 of 3 serial runs together.
  - Docs: settings.md ("Coming back from a provider fix") and console.md ("After Return to Console", After a save).
- Also asked (Env keys): D2's "sent no key" counts a key from an env var as sent (identity source `environment`), so a 403 to it gets no hint, as for a saved or typed key.
- Follow-ups filed (each checked on origin/dev `111f8e48fc` first; none had a task): TASK-35301 (custom-endpoint returns), TASK-35302 (`python -m tldw_chatbook.app`), TASK-35303 (the project-instruction row's `NoMatches`), TASK-35304 (the Voice step's 403), TASK-35305 (a Fast Lane bootstrap guard; it would flag 4 of today's 92 sandboxed targets), TASK-35306 and TASK-35307 (two stale tests), TASK-35308 (the launch-cover return skips `notify_rail_opened`), TASK-35309 (four Library size rows).
- Verified 2026-10-17 on this branch's code (the D5 commit's tree), against base `111f8e48fc` in a detached worktree:
  - The 36 files that name the touched listing, evidence and wizard symbols (xdist, by file): head 542 failed, 2929 passed; base 541 failed, 2916 passed. 455 of head's are the local `RecoveryRequired` noise. The one node failing at head only, `Tests/Performance/test_console_keystroke_work_census.py::test_typing_never_snapshots_the_transcript` (a snapshot count under load), passes 2 of 2 serial runs at head and fails 1 of 2 at base. An earlier head run of the same files had `test_a_listing_403_converts_to_the_non_blocking_listing_result` as its only head-only node; it is rewritten on purpose above.
  - `Tests/UI/test_console_provider_return.py` whole (xdist, per test): 57 passed at head and at base.
  - Negative controls: with D2's source reverted its new tests fail (16 nodes, among them every hinted case); with the key check removed the six sent-a-key pins fail; with the subscription guard removed its case fails. D5's two pins fail on the old copy.
  - Size ratchets (the module, screen and Library files, serial): the same 11 rows fail at head and base, dev's 7 module rows and the 4 Library rows of TASK-35309. settings_screen.py measures 35,817 lines (row 35,818), first_run_model_step.py 997 (row 997), console_settings_modal.py 6,189 (row 6,193); first_run_provider_step.py and FirstRunSetupWizard.py are not touched.
  - `Tests/Performance/test_ui_ready_module_census.py` (serial): 4 passed. The four ADR-097 boot guards (import weight, UI-ready census, pre-import payload, boot CSS; serial): 19 passed.
  - Tests/CI (by file): 959 passed, 1 skipped. `./scripts/preflight.sh` under a scratch HOME, `XDG_*` and `TLDW_CONFIG_PATH`: exit 0. ruff on the touched Python files: 235 findings at head and at base, none new, and no file's formatting changes.
**Capture fixes (after `a57bc97f87`, Phase 8 full-screen captures).** Five capture-fix waves and their rounds (70 commits to `1c1730f4b0`, about 680 production lines) fixed what the live captures showed; `qa/model-config-33008-captures/README.md` records each retake. By area:
- Console status row: the cost chip fits the room left by the chips before it, drops whole figures, keeps a focused chip in view and refits after a model switch or a layout change. No subtask's acceptance criteria cover this work; it is recorded here, not in a subtask, and AC#11 to AC#13 describe it (owner decision D4, 2026-10-17).
- Footer: under the Get started card it names only the keys that act there (no F6), and under a dialog over the Console only Ctrl+P and Ctrl+Q (`CONSOLE_WORKBENCH_SHORTCUTS_COVERED`).
- Switch model: Enter's label follows the highlighted row and keeps its width, NOT RUNNING wraps and repacks on resize, MLX-LM it cannot check reads "not checked", Find counts in the singular, and no blank row after a probe shortens the list.
- Get started card: a long provider name is shortened so its readiness word stays on one line, and step 1 names the provider that word is about.
- Wizard (TASK-33008.5's area): a refusal of a check that sent no key asks for a key (key-required copy), the Authentication section drops "(optional)" and scrolls into view, the status box hugs its line (`#setup-model-choice.-status`), and the pinned notice keeps off the frame (`.setup-step-error` padding).
- Settings: a key check after the save redraws the return line ("the provider accepted the key"), and the unsaved-changes and Return without saving copy were reworded.

**Review fixes (2026-10-16).** The review of `1c1730f4b0`: the wizard's Test 401, Claude subscription and cleared-key cases and the status picker's Tab stop (see TASK-33008.5), the setup card's focus guard under a dialog, the return line's key-accepted copy for another provider, docs and docstrings, ratchet rows re-pinned to measured sizes, polling waits in the wizard's compose-crash tests, and the phase's new tests gated in the PR lanes.

**Review fixes, rounds 2 to 5 (2026-10-16).** Round 2 (`9b76adb752`..`f54e5fa863`, `06d` retaken in `2489f65118`): the subscription check reads Anthropic's table as readiness does; a key a check saw refused reads as checked, and Authentication drops "(optional)"; the Provider step names an unavailable listing as Model does; the rest of the phase's tests joined a PR lane (the wizard's bootstrap nodes in the admission step); polling waits replaced fixed pauses. Round 3 (review of `465eb63bf5..f54e5fa863`) fixed the Provider step's copy:
- A listing that cannot list reads "Model listing unavailable; you can enter the model ID on the next step." The step has no model ID field; Model keeps its own line.
- A Test's 401 to a key that was sent reads like the listing's: "<Provider> rejected this API key. Update it, or continue anyway." It keeps its ✗ and no longer uses the shared verdict's words.
- `failed_listing` returns the line and changes nothing; the listing handler marks the key.
- A new test pins the listing-path 401 title (admission lane). First_Run_Setup.md covers the staged key line's no-usable-list case. The Provider step's ratchet row is now 2,454. `06d` was retaken in `5c75fe49c8`; only its Test line changed.
- A 403 to the wizard's model listing no longer rejects the key (only a 401 does, owner ruling 2026-10-02): both steps read it as an unavailable listing, the key is not marked and its line reads as after a 403 to Test (unit case in the gated listing test, one admission-lane wizard node).
- Round 4 (review of `2489f65118..8c8f54773e`; `03cdc69dcc`..`95ab4ed876`): every failed check redraws the key line, so a 403 or a timeout after a 401 to the same key and address no longer leaves "the server rejected it"; a Test that got no usable list (a 403, a 404) reads "Model listing unavailable; you can enter the model ID on the next step." as the listing does (no capture shows one: `06b` and `06d` are 401s, so none was retaken); a keyless 403 listing is pinned as unavailable; the two discovery-on-return tests were made to wait for the mount's own resume (round 5: a product race, not test timing).
- Round 5 (review of `33391b94f7..fe230a6f96`; `79545922c3`..`63b763816c`): what round 4 waited out in the tests is a product race. The app pushes the first-run wizard (`call_after_refresh`) right after the first Console push, so the wizard suspends Console before its attach reconciliation finishes; on the return, view-start re-claims the one-shot mount token, and the deferred resume read as the mount's own and re-armed no discovery. A server started while the wizard was open (or under any screen pushed at launch) was not offered on that return. `on_screen_suspend` now marks the next resume to re-arm discovery (`_console_discovery_rearm_pending`); the token and task-15475's dedupe are unchanged. `test_a_card_covered_at_launch_runs_discovery_again_on_the_first_return` pins it, the two return tests' wait is gone, and the three run in the UI Fast Lane (census floor 320). A unit case pins a 403 and a 404 to Test reading as the listing does (PR Fast Lane).

Verified 2026-10-16 on `feat/model-config-p8-first-run` at `63b763816c` (the code; the next commit changes only this note), against base `fe230a6f96` in a detached worktree:
- Negative controls: without the `chat_screen.py` fix the new covered-at-launch test fails 3 of 3 serial and 3 of 3 xdist runs; the two return tests, without their wait, pass or fail depending on timing (one failed in 1 of 3 xdist runs at load average ~10; the round-6 review saw one fail serially too). Without the `elif` in `tested_evidence_detail`, both cases (403, 404) of the new unit test fail.
- `Tests/UI/test_console_local_server_discovery_card.py` (xdist, per test): 19 passed in 11 of 12 runs at head (load average 4-40). The other (the fifth, at load average ~40) failed `test_a_detected_legacy_alias_server_lists_its_models` before any suspend: a Console worker raised `NoMatches` for `#console-project-instruction-status-button` during mount; it passes 3 of 3 serial. Base: 18 passed in 3 of 3 runs. The covered-at-launch test and both return tests pass 3 of 3 serial runs together.
- The 30 Tests/UI files that name `on_screen_resume`, `on_screen_suspend` or `_console_mount_visit_refreshed` (xdist): head 2007 failed, 1265 passed; base 2012 failed, 1259 passed (the local `RecoveryRequired: raw_source_selection_changed` noise, 1,955 nodes at head). No node fails at head only; the 5 that fail at base only pass 2 of 2 serial runs at both.
- Tests/Architecture's 14 files that name chat_screen (serial): the same 24 nodes fail at both. The module-size ratchet fails only dev's 8 known rows; `chat_screen.py` measures 24,315 lines and 739 `ChatScreen` methods against its 24,369/740 row, which passes.
- `Tests/Wizards/test_first_run_setup_state.py`: 252 passed at head, 250 at base. Tests/CI: 959 passed, 1 skipped at both.
- `./scripts/preflight.sh` passes under a scratch HOME; ruff reports no new findings on the touched files.
- Round 4's run (at `95ab4ed876` against `33391b94f7`), not repeated: Tests/Wizards (xdist) showed base's 1 failure and 31 `RecoveryRequired` errors at both, plus one load flake at head that passes serially. Each test round 4 changed fails without its fix, except the keyless 403 listing case, which pins existing behaviour: it passes against `33391b94f7`'s `first_run_model_discovery.py`.

**Acceptance check (2026-10-16, head `8c8f54773e`).** Measured against origin/dev `edbf951ba7` and the merge base `14e13bb82c`. The key counts and the captures were taken before the branch's restack onto dev. `4f2bffa20f`..`7c260226ec` sat on base `5e661c1565`; `7c260226ec` is `6396f4896d` after the rebase, 988 dev commits later, among them the rest of Phase 7's Settings work. What later changed on each captured surface is named below, and the copy each capture shows is in head's code.
- AC#1: 2 key presses, measured live at 211x44 (README "Key counts": `02a`, `02b`, `02c`). At head, `Tests/UI/test_console_local_server_discovery_card.py::test_detected_server_enter_enter_saves_config_and_unlocks_card` passes: Enter, Enter at 211x44, the card unlocks, and the pair and the endpoint are saved. Changed on that path since: `b2e1914550`, where the card's focus guard reads its own screen; that matters only under a dialog.
- AC#2: 6 key actions against a budget of 12, measured live at 211x44 (`03a`-`03f`): Enter, paste, Esc, s, t, Enter, and back in "Chat 1" with no navigation by hand. At head two tests pass. `Tests/UI/test_console_provider_return.py::test_recovery_round_trip_returns_to_the_chat_on_one_enter` drives Enter on the card, the key, Esc, s and Enter at 211x44, and lands on the same chat id with the composer focused and `ready_to_send`. `test_a_route_that_names_no_field_arrives_on_the_return[saved-key]` drives t, then Enter back. Changed since:
  - `b0feaea19b`: "the provider accepted the key" is said only for the recovery's provider. `03e` checks the recovery's provider, so its line holds.
  - `e8917048fe`: the return labels are escaped, with no visible change.
- AC#3: **not met as written.** A named custom endpoint (`custom-ep:<slug>`) leaves for Settings with no return.
  - `open_provider_setup` (`UI/Console_Modules/model_switcher.py`) catches the `ValueError` that `ProviderSettingsNavigationTarget` raises for such an id, and posts the one-way context (R35). That covers both routes the tests drive: the card's Set up provider, and the endpoint's NEEDS SETUP row in Switch model.
  - At head, the custom-endpoint case of `test_a_route_whose_named_field_is_disabled_arrives_where_it_can_act` and both cases of `test_a_custom_endpoint_row_with_no_field_to_focus_arrives_where_it_can_act` pin it: all pass, and none shows a return.
  - The controller's R37 ruling (2026-10-10) accepted this, and the copy says so: the card's line reads "except for custom endpoints", and settings.md reads "Settings remembers no way back for a custom endpoint". The AC's "Every" was not amended.
  - Every other provider recovery that leaves the Console has its return. The card, the transcript's action, the prompts hook, a cloud Retry and the NEEDS SETUP rows go through `open_provider_setup`. Chat settings' credential detour returns to the modal. The Console's other two exits to Settings (Terminal's Privacy & Security, Hooks review) are not provider recoveries.
  - To close it, either the owner scopes the AC to the providers a return can name, or a custom endpoint's route gets a return.
  - Ruled 2026-10-17 (D1): both. The AC is rescoped and ticked, and TASK-35301 is filed for the custom endpoint's return.
- AC#4: at head no Console input takes a key. Every `Input` built under `Widgets/Console/`, `UI/Console_Modules/` and `chat_screen.py` was listed, and none is a password field or a key field. They are search, filter, name, path, prompt, base URL and sampling fields; Chat settings' rows edit only `MODEL_CONFIG_FIELDS`.
  - The Console's credential controls are buttons that open Settings: the card's Set up provider and Chat settings' Configure credential.
  - Switch model's only inputs are Find, Temperature and Max tokens, none a password. `Tests/UI/test_console_model_switcher.py::test_enter_on_needs_setup_opens_that_providers_settings_fix` pins that and passes at head.
  - A custom endpoint template copies only `api_key_env`, never a stored key.
  - Console surfaces show readiness words ("no key", "key rejected"), never a key. No Console capture shows a key field.
- AC#5: `chat_screen.py` measures 24,303 lines and 739 `ChatScreen` methods against its row of 24,369/740 (`Tests/Architecture/test_screen_size_ratchet.py`, unchanged from dev).
  - The merge base measures 24,369/740, so the branch nets -66 lines and -1 method. origin/dev now measures 24,390/741, over that same row.
  - `FirstRunSetupWizard.py` measures 3,136 lines against 3,144 at the merge base and on origin/dev. Its row is lowered 3,144 -> 3,136, and so are its steps' rows: `first_run_model_step.py` 1,012 -> 997 and `first_run_provider_step.py` 2,458 -> 2,454.
  - A serial run at head of both ratchet files fails only on dev's 8 known module rows.
  - `Tests/Architecture/test_console_review_selection_controller_boundary.py::test_chat_screen_remains_within_reviewed_projection_without_ratchet_raise` fails too: 24,303 lines against a 17,215 ceiling set 2026-08-27. It is red on dev (24,390 lines) and is not this file's budget.
- AC#6: the ADR-097 constants are the same at head and on origin/dev: `MAX_TLDW_MODULE_COUNT` 686, `MAX_TLDW_MODULES_AT_UI_READY` 1031, `MAX_BOOT_PARSED_CSS_BYTES` 608,090, and `MAX_PASS_ADDED_MODULES` / `MAX_PASS_ADDED_LOC` / `MAX_SINGLE_ROUTE_ADDED_LOC` 558 / 425,347 / 135,111.
  - A serial run of the four guards at head passes 14 tests. Import weight 679/686 modules; `_ui_ready` 1025/1031; pre-import 557/558 modules, 420,090/425,347 LOC and 128,371/135,111 LOC (library).
  - Boot-parsed CSS, the bytes of the five boot sources as the census reads them, is 604,860 B at head against 605,159 B on origin/dev and at the merge base: -299 B. `widget_defaults_self.tcss` drops 164 B (the `ConsoleSetupBackdrop` segment) and `tldw_cli_modular.tcss` 135 B. Dev's sheets reproduce from their sources (`check_bundle_sync.py` run on a `git archive` of origin/dev).
  - The snapshot `boot_css_bytes.json` reads 604,860 at head and 602,550 on dev. Dev's own sheets measure 605,159, so that +2,310 is dev's drift picked up by the refresh, not growth.
  - The ADR-150 hex ratchet (`_HEX_GRANDFATHERED`) is unchanged from dev, and its test passes at head.
- AC#7: the README's set, at 211x44 and 235x52: the local journey `02a`-`02c` (`02d` is at 211x44 only) and the cloud journey `03a`-`03f`. Each launch ran from this worktree with its own stylesheet and a scratch `HOME`, `XDG_*` and `TLDW_CONFIG_PATH`; keys went in by `tmux send-keys`, and the key by one bracketed paste (README "Procedure").
- AC#8: amended. It asked for a Verified-against stamp on each page, but CLAUDE.md now says "Do not add 'Verified against' paragraphs to User Guide pages" and asks for what was verified to be recorded in the task's Implementation Notes, so the AC now says that. The branch adds no stamp. Checked at head against the code and the captures:
  - console.md, Get started: the card's subtitle equals `CONSOLE_SETUP_CARD_SUBTITLE`. The footer under the card equals `01a`'s and the footer under a dialog `02b`'s. Also checked: Set up provider, then Esc and s, then Return to Console; the custom endpoint exception; Use detected's Enter, Enter; the scope line and the Enter labels (`console_model_popover.py`, `model_switcher.py`); and "Setup complete — …" (`console_onboarding_state.py`).
  - console.md, "After Return to Console": matches `05c`.
  - settings.md, "Coming back from a provider fix": the arrival line, the saved line and the accepted-key line are `ARRIVAL_COPY`, the saved copy and `KEY_ACCEPTED_COPY` in `UI/Settings_Modules/provider_return.py`. Return without saving's question and Keep editing are as in `04b`. A custom endpoint gets no return.
  - First_Run_Setup.md, the model-step status lines: every quoted line is in `first_run_model_step.py`, `first_run_setup_state.py` or `first_run_model_discovery.py`, including `8c8f54773e`'s reading of a 403.
- AC#9: `./scripts/preflight.sh` passes at head under a scratch `HOME`, `XDG_*` and `TLDW_CONFIG_PATH`: 12 checks, exit 0.
- Seen while checking, not fixed: `Tests/UI/test_console_local_server_discovery_card.py::test_a_card_resumed_with_nothing_detected_runs_discovery_again` failed in a 5-file run (213 others passed) and in 1 of 3 serial reruns, at load average 30-60. Its wait for the detected action is 60 × 0.05 s.
- Statuses: TASK-33008.1 to .5 are Done. This task stayed To Do until AC#3 was ruled on; the owner's decisions of 2026-10-17 (above) close it.
<!-- SECTION:NOTES:END -->
