---
id: TASK-33008
title: 'Phase 8: First run connects in place and always returns to Console'
status: To Do
assignee: []
created_date: '2026-09-26 11:47'
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
  - 'tldw_chatbook/UI/Screens/chat_screen.py'
  - 'tldw_chatbook/UI/Navigation/pending_handoff_store.py'
  - 'tldw_chatbook/UI/Navigation/conversation_settings_navigation.py'
  - 'tldw_chatbook/Widgets/Console/console_setup_modal.py'
  - 'tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py'
  - 'tldw_chatbook/UI/Screens/settings_screen.py'
  - 'Docs/User_Guide/console.md'
  - 'Docs/User_Guide/settings.md'
  - 'Docs/User_Guide/First_Run_Setup.md'
  - 'PRODUCT.md'
  - 'backlog/decisions/012-provider-credential-settings-boundary.md'
  - 'backlog/decisions/033-application-session-state-ownership.md'
  - 'backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md'
  - 'backlog/decisions/095-conversation-owned-console-generation-settings.md'
  - 'backlog/decisions/097-boot-budget-ratchets.md'
  - 'qa/model-config-ux-review-2026-09-26/judge-synthesis.md'
  - 'qa/model-config-ux-review-2026-09-26/verified-claims.md'
priority: high
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
- [ ] #3 Every Console provider-recovery path that leaves for Settings offers a one-key return to the chat it came from. This closes C1's recovery dead end.
- [x] #4 No Console surface accepts or displays an API key; keys are entered only in Settings ▸ Providers & Models (D4, ADR-012).
- [x] #5 chat_screen.py stays within its line and method budget, and FirstRunSetupWizard.py does not grow.
- [x] #6 No ADR-097 ratchet rises; boot CSS bytes go down with the backdrop's removal.
- [x] #7 Keyboard-only live captures of both journeys at 211x44 and 235x52 are attached, taken with the real stylesheet and a scratch TLDW_CONFIG_PATH.
- [x] #8 Docs/User_Guide pages updated where behaviour changed: console.md (Get started), settings.md (return action) and First_Run_Setup.md (model-step status lines); what was verified is recorded in Implementation Notes.
- [x] #9 ./scripts/preflight.sh passes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Capture fixes (after `a57bc97f87`, Phase 8 full-screen captures).** Five capture-fix waves and their rounds (70 commits to `1c1730f4b0`, about 680 production lines) fixed what the live captures showed; `qa/model-config-33008-captures/README.md` records each retake. By area:
- Console status row: the cost chip fits the room left by the chips before it, drops whole figures, keeps a focused chip in view and refits after a model switch or a layout change. No subtask's acceptance criteria cover this work; it is recorded here, not in a subtask.
- Footer: under the Get started card it names only the keys that act there (no F6), and under a dialog over the Console only Ctrl+P and Ctrl+Q (`CONSOLE_WORKBENCH_SHORTCUTS_COVERED`).
- Switch model: Enter's label follows the highlighted row and keeps its width, NOT RUNNING wraps and repacks on resize, MLX-LM it cannot check reads "not checked", Find counts in the singular, and no blank row after a probe shortens the list.
- Get started card: a long provider name is shortened so its readiness word stays on one line, and step 1 names the provider that word is about.
- Wizard (TASK-33008.5's area): a refusal of a check that sent no key asks for a key (key-required copy), the Authentication section drops "(optional)" and scrolls into view, the status box hugs its line (`#setup-model-choice.-status`), and the pinned notice keeps off the frame (`.setup-step-error` padding).
- Settings: a key check after the save redraws the return line ("the provider accepted the key"), and the unsaved-changes and Return without saving copy were reworded.

**Review fixes (2026-10-16).** The review of `1c1730f4b0`: the wizard's Test 401, Claude subscription and cleared-key cases and the status picker's Tab stop (see TASK-33008.5), the setup card's focus guard under a dialog, the return line's key-accepted copy for another provider, docs and docstrings, ratchet rows re-pinned to measured sizes, polling waits in the wizard's compose-crash tests, and the phase's new tests gated in the PR lanes.

**Review fixes, rounds 2 to 4 (2026-10-16).** Round 2 (`9b76adb752`..`f54e5fa863`, `06d` retaken in `2489f65118`): the subscription check reads Anthropic's table as readiness does; a key a check saw refused reads as checked, and Authentication drops "(optional)"; the Provider step names an unavailable listing as Model does; the rest of the phase's tests joined a PR lane (the wizard's bootstrap nodes in the admission step); polling waits replaced fixed pauses. Round 3 (review of `465eb63bf5..f54e5fa863`) fixed the Provider step's copy:
- A listing that cannot list reads "Model listing unavailable; you can enter the model ID on the next step." The step has no model ID field; Model keeps its own line.
- A Test's 401 to a key that was sent reads like the listing's: "<Provider> rejected this API key. Update it, or continue anyway." It keeps its ✗ and no longer uses the shared verdict's words.
- `failed_listing` returns the line and changes nothing; the listing handler marks the key.
- A new test pins the listing-path 401 title (admission lane). First_Run_Setup.md covers the staged key line's no-usable-list case. The Provider step's ratchet row is now 2,454. `06d` was retaken in `5c75fe49c8`; only its Test line changed.
- A 403 to the wizard's model listing no longer rejects the key (only a 401 does, owner ruling 2026-10-02): both steps read it as an unavailable listing, the key is not marked and its line reads as after a 403 to Test (unit case in the gated listing test, one admission-lane wizard node).
- Round 4 (review of `2489f65118..8c8f54773e`; `03cdc69dcc`..`95ab4ed876`): every failed check redraws the key line, so a 403 or a timeout after a 401 to the same key and address no longer leaves "the server rejected it"; a Test that got no usable list (a 403, a 404) reads "Model listing unavailable; you can enter the model ID on the next step." as the listing does (no capture shows one: `06b` and `06d` are 401s, so none was retaken); a keyless 403 listing is pinned as unavailable; the two discovery-on-return tests wait for the mount's own resume, which their return raced.

Verified 2026-10-16 on `feat/model-config-p8-first-run` at `95ab4ed876` (the code; the next commit changes only this note), against base `33391b94f7` in a detached worktree:
- Tests/Wizards (xdist, run at `0431f38f30`; `95ab4ed876` changes only a Tests/UI file): base 1 failed, 1209 passed, 31 `RecoveryRequired` errors. Head shows the same failure and errors, plus one load flake (`test_mounted_model_owner_timeout_fences_late_result_and_keeps_manual_retry`, load average 25-30) that passes serially, 3 of 3 at head and 1 of 1 at base.
- `Tests/UI/test_console_local_server_discovery_card.py` (xdist, per test): 18 passed in 6 of 6 runs at head at load average 40-70; at base one of its two return tests failed in 2 of 4 runs. Both return tests pass 3 of 3 serial runs at head.
- Tests/CI: 959 passed, 1 skipped at both.
- Module-size ratchet (serial): only dev's 8 known rows fail at both; no wizard row changed.
- `./scripts/preflight.sh` passes under a scratch HOME; ruff reports no new findings on the touched files.
- Each changed test fails without its fix.

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
- Statuses: TASK-33008.1 to .5 are Done. This task stays To Do until AC#3 is ruled on.
<!-- SECTION:NOTES:END -->
