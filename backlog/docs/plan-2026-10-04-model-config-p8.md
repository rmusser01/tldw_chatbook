# Plan: Phase 8: First run connects in place and always returns to Console (TASK-33008)

Spec: backlog/docs/spec-2026-09-26-model-config-redesign.md (binding authority; ADR-095 and ADR-012 amendments of 2026-09-26).
Evidence: qa/model-config-ux-review-2026-09-26/ (mockups-211x44.md, verified-claims.md).
Parent task: backlog/tasks/task-33008 - Phase-8-First-run-connects-in-place-and-always-returns-to-Console.md

## Phase goal (parent)

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

### Parent acceptance criteria

- [ ] #1 First run with a detected local server takes 2 key presses (Enter, Enter) from the Get started card to a ready composer, measured live at 211x44.
- [ ] #2 First run with a cloud key takes no more than 12 key actions from the Get started card to a ready composer in the same chat, counting a paste as one. The count includes the key check and the return, and the user never navigates back to Console by hand. Measured live at 211x44.
- [ ] #3 Every Console provider-recovery path that leaves for Settings offers a one-key return to the chat it came from. This closes C1's recovery dead end.
- [ ] #4 No Console surface accepts or displays an API key; keys are entered only in Settings ▸ Providers & Models (D4, ADR-012).
- [ ] #5 chat_screen.py stays within its line and method budget, and FirstRunSetupWizard.py does not grow.
- [ ] #6 No ADR-097 ratchet rises; boot CSS bytes go down with the backdrop's removal.
- [ ] #7 Keyboard-only live captures of both journeys at 211x44 and 235x52 are attached, taken with the real stylesheet and a scratch TLDW_CONFIG_PATH.
- [ ] #8 Docs/User_Guide pages updated, each with a Verified-against stamp: console.md (Get started), settings.md (return action) and First_Run_Setup.md (model-step status lines).
- [ ] #9 ./scripts/preflight.sh passes.

## Global Constraints

- Work ONLY in this worktree: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/model-config-p8. Start EVERY shell command with `cd <that path> &&`. Never touch the main checkout (another session's uncommitted work).
- Python: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python (the worktree has no venv); run pytest FROM the worktree cwd. Tests/Architecture needs -p no:xdist.
- NEVER write ~/.config/tldw_cli or ~/.local/share/tldw_cli. Never bypass test isolation (private_profile_test, TLDW_TEST_PRIVATE_PROFILE_NODE, HOME/XDG/TLDW_CONFIG_PATH). Live runs: scratch TLDW_CONFIG_PATH with a unique users_name, FULL SCREEN 211x44 (primary) and 235x52.
- ADR-126 RecoveryRequired in a clean worktree is environmental: compare failure-NAME sets against origin/dev.
- Size ratchets never rise (ADR-097); console_settings_modal.py net lines <= 0 against its current row.
- Geometry only through tokens in css/core/_variables.tcss (ADR-150/161); rebuild the CSS bundle with the repo script.
- ADR-031: never bind Ctrl+C/V/X/S/D/Z/A/R/W; footer hints must match working bindings.
- TDD; real-implementation tests for config/provider surfaces; rewrite pinned tests on purpose and name them.
- Commit per task `fix|feat(model-config): <summary> (TASK-33008.N)` + `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never push; NEVER merge origin/dev into the branch.
- Tick ACs, set Done, add Implementation Notes in each subtask file. Update Docs/User_Guide CONTENT; never add "Verified against" paragraphs (record verification in task notes).
- Lessons (backlog/docs/lessons-*.md): insert new entries MID-FILE near a related section, never appended at the end of the file (dev appends there constantly and every rebase conflicts).
- Evidence: commit captures (.txt/.ansi.txt) under qa/, but NEVER commit one-off driver or probe scripts (.sh/.py with machine-specific paths); describe the capture procedure in a short qa README instead. Captures are reviewed by AI reviewers, so they must not contradict the PR's claims.
- Scratch hygiene: every scratch directory carries a unique name with your task and sha (e.g. $SP/p<N>-t<M>-base-<sha>); never generic names (base, head, tree, cfgroot, devtree, neg); never rm -rf a directory you did not create; remove your own trees when done (git worktree remove for worktrees).
- Boot census: a NEW module only user actions need is imported inside the function that uses it; measure Tests/Performance/test_ui_ready_module_census.py with PYTHONPATH=<worktree> (the venv editable install otherwise measures the MAIN checkout); the limit must not rise.
- Before DONE: covering tests + `PYTHON=<venv> ./scripts/preflight.sh` (rc 0; never piped through tail).

## Owner speed rules (2026-10-04, binding; they OVERRIDE the Evidence and Before-DONE bullets above)

- Fix EVERY review finding, Minor included; fix rounds are not capped.
- Tests always run in parallel (xdist, `-n 6 --dist loadfile`); serial only for Tests/Architecture.
- Per task, run the test files that cover what you changed, once, at HEAD. The phase runs ONE base-vs-head comparison at its end, both trees at once.
- Evidence is the test names plus the command and its summary line in the report. Commit no per-task captures or evidence files; the Capture phase captures every changed surface once.
- Execution order: Tasks 3 and 5 run in parallel worktrees, then Task 4. Tasks 1 and 2 wait until Phase 7 (TASK-33007) is merged and this branch is rebased onto it.

## Task 1: Return to the originating Console chat after provider recovery in Settings (TASK-33008.1)

Task file: backlog/tasks/task-33008.1 - Return-to-the-originating-Console-chat-after-provider-recovery-in-Settings.md
Depends on: TASK-33007, TASK-33001

### Why

_open_console_provider_recovery (chat_screen.py:20533-20565) sends credential blockers to Settings. It passes category, provider, model and field context but carries no return, so the user is stranded after adding a key (the remaining part of C1).

Every recovery entry routes through this one function, so a fix there covers all of them:
- the setup card (Widgets/Console/console_setup_modal.py:550)
- the transcript's blocked-state action (Widgets/Console/console_transcript.py:2554)
- the prompts configure-provider hook (UI/Console_Modules/prompts.py:1265)

The Chat settings modal's credential detour already has a return. It stages a secret-free ConversationSettingsReturnIntent on HandoffChannel.CONVERSATION_SETTINGS_RETURN (chat_screen.py:3556-3600), and Settings shows a return continuation (settings_screen.py:16980-17004, handler :28656). ADR-033 (application-session-state:13-23) makes PendingHandoffStore the owner of such handoffs.

Settings' return copy and button are written only for that modal case (settings_screen.py:12228-12242, :16976-16990).

ADR-012:31 requires recovery to open the exact credential control. ADR-095 and its 2026-09-26 amendment decide what happens to the returning chat: an untouched chat converges to the saved defaults (D1, phase 1), and a chat that holds work keeps its own settings.

chat_screen.py has 32 lines of headroom (test_screen_size_ratchet.py:85).

Two tests pin today's context dict, which has no return:
- Tests/UI/test_console_native_chat_flow.py:6526, test_console_add_api_key_recovery_targets_provider_settings_category
- :6562, test_console_add_api_key_recovery_tolerates_missing_session_settings

### Acceptance criteria

- [ ] #1 From any provider-recovery entry that targets Settings, Settings ▸ Providers & Models opens with the named provider selected and focus in the exact control the blocker names, such as the API key field for a missing key (ADR-012:31).
- [ ] #2 When Settings is reached this way, it shows a 'Return to Console' action whose copy names the chat. After a successful save, focus lands on that action, so one Enter returns.
- [ ] #3 Returning lands on the chat that started the recovery, with the composer focused. If that chat was untouched it now uses the saved provider·model (D1); if it holds work it keeps its own settings (ADR-095).
- [ ] #4 'Return without saving' keeps today's discard confirmation and lands on the same chat unchanged.
- [ ] #5 The return request is a revisioned single-slot PendingHandoffStore handoff that carries no key, endpoint or prompt text, and a newer recovery supersedes it.
- [ ] #6 If the chat no longer exists or the handoff is stale, Console opens normally with a visible notice and applies nothing stale.
- [ ] #7 The Chat settings modal's existing credential round trip (return to the modal) behaves as before.
- [ ] #8 test_console_add_api_key_recovery_targets_provider_settings_category and test_console_add_api_key_recovery_tolerates_missing_session_settings are rewritten on purpose to assert the return-bearing route.
- [ ] #9 A real-implementation integration test runs the real Console store and the real Settings screen on a scratch TLDW_CONFIG_PATH, with no mocked navigation. It drives card → Settings → key entry → save → Enter and asserts the chat identity, composer focus and readiness.
- [ ] #10 chat_screen.py's row in Tests/Architecture/test_screen_size_ratchet.py:85 stays green.
- [ ] #11 Live captures at 211x44 show the Return to Console action in Settings and the Console after the return.

### References

- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/UI/Navigation/pending_handoff_store.py
- tldw_chatbook/UI/Navigation/conversation_settings_navigation.py
- tldw_chatbook/UI/Console_Modules/prompts.py
- tldw_chatbook/Widgets/Console/console_transcript.py
- Tests/UI/test_console_native_chat_flow.py
- Tests/Architecture/test_screen_size_ratchet.py
- backlog/decisions/033-application-session-state-ownership.md
- backlog/decisions/012-provider-credential-settings-boundary.md
- backlog/decisions/095-conversation-owned-console-generation-settings.md

## Task 2: Send the switcher's NEEDS SETUP rows through the same return route, and reopen the switcher afterwards (TASK-33008.2)

Task file: backlog/tasks/task-33008.2 - Send-the-switcher-s-NEEDS-SETUP-rows-through-the-same-return-route-and-reopen-the-switcher.md
Depends on: TASK-33008.1, TASK-33004, TASK-33005

### Why

Phase 4's switcher lists providers that cannot send yet under NEEDS SETUP, with the reason and the fix. Its Enter on such a row opens Settings one way, with no return. The judge's mockup row reads 'Not ready · no key — Enter: add key in Settings ↩'.

A row whose fix is a credential or an endpoint has to leave Console. Without P8.1's return route, that recreates the dead end. Rows whose fix lies outside the app (a stopped local server) already keep in-place guidance since phase 5 and are not changed here.

A chat that already holds work keeps its own settings after a Settings save (ADR-095:26), so after returning the user still has to pick a model from the newly usable provider. Reopening the switcher filtered to that provider turns that into one Enter. Credentials stay in Settings (D4, ADR-012).

### Acceptance criteria

- [ ] #1 Enter on a NEEDS SETUP row whose fix is a missing key or endpoint opens Settings at that provider's exact control through P8.1's return route, replacing phase 4's one-way route.
- [ ] #2 After the return, the switcher reopens filtered to that provider. Its models show the phase-5 readiness words, and one Enter applies the highlighted pair to this chat.
- [ ] #3 Returning without saving reopens the switcher unchanged.
- [ ] #4 An integration test on a scratch config drives a NEEDS SETUP row to Settings, saves a key, returns, and applies the pair. It asserts the chat's provider·model.
- [ ] #5 A live capture at 211x44 shows the reopened, filtered switcher.

### References

- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/Widgets/Console/console_model_popover.py
- tldw_chatbook/UI/Console_Modules/model_switcher.py
- backlog/decisions/095-conversation-owned-console-generation-settings.md
- backlog/decisions/012-provider-credential-settings-boundary.md
- qa/model-config-ux-review-2026-09-26/judge-synthesis.md

## Task 3: Open the switcher from the Get started card, filtered to a detected server (TASK-33008.3)

Task file: backlog/tasks/task-33008.3 - Open-the-switcher-from-the-Get-started-card-filtered-to-a-detected-server.md
Depends on: TASK-33004

### Why

'Use detected <provider>' calls _apply_detected_local_server (chat_screen.py:15397-15479). That adopts server.model_ids[0] without showing the choice, persists it and the endpoint as chat_defaults, and applies it to the chat. On a server with several models, the user never sees which one was picked. This path always persists the pair as the new-chat default today, which is right for first run and must now be said before Enter.

The card's 'Choose provider' and 'Choose model' actions open the full Chat settings modal: their recovery targets are 'console' in Widgets/Console/console_settings_summary.py:58-96, handled at chat_screen.py:20536-20541. After phase 4, though, a model is chosen as a provider·model pair in the switcher.

On first run there is no working new-chat default yet. A pick from the card that applied only to this chat would leave the next Ctrl+T chat blocked again.

task-16476 behaviour must survive: an endpoint that is already configured differently is kept, and the detected one is used for this session only (chat_screen.py:15424-15443).

Tests/UI/test_console_local_server_discovery_card.py:198 (test_pressing_detected_server_saves_config_and_unlocks_card) pins today's one-press adoption.

### Acceptance criteria

- [ ] #1 When a local server is detected, the card focuses 'Use <server>' first. Enter opens the switcher filtered to that provider, with its first model highlighted and every model the server lists visible.
- [ ] #2 Enter in that switcher applies the pair to this chat, makes it the default for new chats, and saves the detected endpoint. If a different endpoint is already configured, the saved endpoint is kept and the detected one is used for this chat only, with today's warning (task-16476).
- [ ] #3 Esc from that switcher returns to the card with nothing persisted.
- [ ] #4 The card's Choose provider and Choose model actions open the switcher instead of Chat settings; endpoint recovery still opens Chat settings.
- [ ] #5 A pair chosen through the card's Choose model action while no working new-chat default exists also becomes the new-chat default.
- [ ] #6 Whenever a card-opened switcher will also set the new-chat default, its scope line says so before Enter.
- [ ] #7 test_pressing_detected_server_saves_config_and_unlocks_card is rewritten on purpose for the two-key flow.
- [ ] #8 A real-implementation integration test on a scratch TLDW_CONFIG_PATH uses a stub loopback server that lists two models and picks the second. It asserts chat_defaults, the endpoint write and the chat's pair.
- [ ] #9 A live capture at 211x44 shows the filtered switcher opened from the card.

### References

- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/Widgets/Console/console_setup_modal.py
- tldw_chatbook/Widgets/Console/console_settings_summary.py
- Tests/UI/test_console_local_server_discovery_card.py
- backlog/decisions/095-conversation-owned-console-generation-settings.md

## Task 4: Give the Get started card a plain dim backdrop and actions that Tab reaches (TASK-33008.4)

Task file: backlog/tasks/task-33008.4 - Give-the-Get-started-card-a-plain-dim-backdrop-and-actions-that-Tab-reaches.md
Depends on: —

### Why

ConsoleSetupBackdrop paints a snow field behind the card (console_setup_modal.py:113; CSS at css/features/_console.tcss:2419). PRODUCT.md:29 rules out 'gratuitous ASCII decoration', and the review lists the backdrop as an anti-reference.

TASK-32572: Tab does not cycle the card's actions. The card shows up to three: Set up provider or Choose model, Write a note in Library, and Use detected <provider>. A keyboard-only user therefore cannot reach the notes action or the detected-server action.

The card must stay the documented non-dismissible gate (ADR-031, task-16211 refinement), and its copy may only teach keys that the footer shows (ADR-031 rule 5).

The opt-in Console background effects (Console Behavior ▸ Background effects, Utils/console_background_effects.py) are a separate feature and are not touched.

Tests that pin the snow field:
- Tests/UI/test_console_setup_backdrop_repaint_cost.py
- the backdrop tests in Tests/UI/test_console_reduced_motion.py (from :77)
- the setup-backdrop section of Tests/UI/test_console_rail_sections.py (from :616)

This subtask absorbs TASK-32572. That task's 100x30 capture is replaced by captures at 211x44 and 235x52, the program's full-screen targets.

### Acceptance criteria

- [ ] #1 Behind the card, the workbench is dimmed with a plain fill from existing tokens and no decorative glyphs, and the snow backdrop's code and CSS are removed.
- [ ] #2 Boot-parsed CSS bytes measure lower than before, MAX_BOOT_PARSED_CSS_BYTES (Tests/Performance/test_boot_css_byte_budget.py:117) is not raised, and the hex ratchet does not rise.
- [ ] #3 Tab and Shift+Tab cycle every action the card is showing, including the detected-server action when present, and the focus cue marks each stop (TASK-32572 AC#1, AC#2).
- [ ] #4 The card's copy says that cloud setup opens Settings and comes back here, and it names only keys the footer shows.
- [ ] #5 Esc still does not dismiss the card.
- [ ] #6 The tests that pin the snow field are deleted or rewritten on purpose and named in the PR, and Tests/UI/test_console_background_effects.py stays green.
- [ ] #7 Keyboard-only live captures at 211x44 and 235x52, on a scratch profile with no provider, show one capture per Tab stop.

### References

- tldw_chatbook/Widgets/Console/console_setup_modal.py
- tldw_chatbook/css/features/_console.tcss
- PRODUCT.md
- Tests/UI/test_console_setup_backdrop_repaint_cost.py
- Tests/UI/test_console_reduced_motion.py
- Tests/UI/test_console_rail_sections.py
- Tests/Performance/test_boot_css_byte_budget.py
- backlog/tasks/task-32572 - Console-Tab-does-not-cycle-the-Get-started-cards-buttons.md
- backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md

## Task 5: Show the wizard's model-step status messages as text, not radio options (TASK-33008.5)

Task file: backlog/tasks/task-33008.5 - Show-the-wizard-s-model-step-status-messages-as-text-not-radio-options.md
Depends on: —

### Why

The wizard's model step mounts its status messages as disabled SetupRadioButton options inside the model RadioSet (FirstRunSetupWizard.py:3836-3906). The messages are: listing unavailable; connection failed, including 'Authentication failed — this API key was rejected. Go Back to fix it…'; loading; pick a provider first; and no models found. A failure is therefore drawn like a choice (report, persona Jordan). In the wizard, the judge's surface map changes only these placeholders.

The TASK-21143 trust chain (tracker mark, Model-step confirm gate, Summary override) reads the classified outcome right after these mounts (:3911-3915). It must behave as before.

These tests in Tests/Wizards/test_first_run_setup_wizard.py assert the RadioButton type:
- :8817, in test_mounted_provider_handoff_preserves_typed_discovery_outcome (:8743)
- :9035, in test_model_connection_failure_shows_bounded_category_and_retry (:8982)
- :9243 and :9264, in test_mounted_model_owner_timeout_fences_late_result_and_keeps_manual_retry (:9173)

The file is already over its size row: 10,462 lines against 10,404 (Tests/Architecture/test_module_size_ratchet.py:81).

### Acceptance criteria

- [ ] #1 Each model-step status renders as a plain text line with its current copy and id. It cannot be focused or selected, and it shows no radio glyph.
- [ ] #2 Retry, manual model entry, and the TASK-21143 tracker, confirm gate and summary override behave as before.
- [ ] #3 The tests that assert the RadioButton type are rewritten on purpose to assert a non-selectable text line.
- [ ] #4 FirstRunSetupWizard.py does not grow (net lines ≤ 0), and its ratchet row is not raised.
- [ ] #5 A live capture at 211x44 shows the model step in the authentication-failed state.

### References

- tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py
- Tests/Wizards/test_first_run_setup_wizard.py
- Tests/Architecture/test_module_size_ratchet.py
- Docs/User_Guide/First_Run_Setup.md
