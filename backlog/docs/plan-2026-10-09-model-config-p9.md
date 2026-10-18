# Plan: Phase 9: Burn down every follow-up rider (TASK-33641)

Spec: backlog/docs/spec-2026-09-26-model-config-redesign.md (binding authority; ADR-095 and ADR-012 amendments of 2026-09-26).
Evidence: qa/model-config-ux-review-2026-09-26/ (mockups-211x44.md, verified-claims.md), plus each rider's own qa/ references.
Parent task: backlog/tasks/task-33641 - Model-config-Phase-9-burn-down-every-follow-up-rider.md

## Phase goal (parent)

Phase 9 is the last phase of the model-configuration redesign, added by owner decision on 2026-09-29. Phases 1-8 and the two hotfixes filed small follow-up riders as review, live-capture and Qodo findings surfaced: pre-existing defects, test-hygiene gaps, and copy or edge cases that fell outside their phase. This phase closes all of them, so the program ends with no open tail.

Scope: every rider that is not Done and is a subtask of TASK-33001 through TASK-33007 or of TASK-33641, plus TASK-33666 and TASK-33667. Phase 8's own tasks (TASK-33008.x) are in flight on the branch below this one and are out of scope.

This plan triaged 59 riders against this branch's HEAD (8b4a1ad23e, stacked on Phase 8's in-progress branch, which is stacked on Phase 7):
- 52 VALID (TASK-33003.13 joined them once the owner ruled), plus TASK-33641.7 (filed from the wave 1 integration) TASK-33641.8 (filed from a wave 2 incident) and TASK-33641.9 (filed from Phase 8's Task 2 review): grouped into the plan tasks below (Task 17 holds TASK-33641.10 and .11, filed from Task 9's review).
- 7 ALREADY FIXED and 0 OBSOLETE: listed under "Closed without code" with their evidence. They get no plan task; each is closed with that rationale in its own task file.
- 0 NEEDS OWNER: TASK-33003.13 was ruled on 2026-10-09 and is in Task 15.

It ships as one PR, or as a few PRs grouped by wave, through the program's usual SDD plus Qodo loop.

Size facts at HEAD (Tests/Architecture/test_module_size_ratchet.py and test_screen_size_ratchet.py):
- settings_screen.py 33,732 against its 33,732 row, console_settings_modal.py 6,193 against 6,193, and UI/Settings_Modules/providers_models_card.py 2,300 against 2,300: zero headroom. A fix there is net lines <= 0, or it moves code into a sibling module (UI/Settings_Modules/ for Settings, Widgets/Console/ for Chat settings).
- chat_screen.py 25,076 against its 25,204 screen budget: 128 lines of headroom shared by every task, so new Console logic goes in UI/Console_Modules.

### Parent acceptance criteria

- [ ] #1 Every rider listed in the description is Done with its ACs ticked, or closed with a written rationale the owner can review. None is silently dropped.
- [ ] #2 Every rider filed during Phases 3–8 that no later phase owns is likewise Done or closed with a rationale.
- [ ] #3 A final sweep of backlog/tasks for open TASK-33001.x through TASK-33008.x riders comes back empty, or lists only the closures named in AC#1 and AC#2.
- [ ] #4 Each fix carries a test that fails before it; config and provider fixes use real-implementation tests on a scratch profile.
- [ ] #5 ./scripts/preflight.sh passes, and each PR is rebased on dev, Qodo-clean and merged.

## Global Constraints

- Work ONLY in this worktree: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/model-config-p9. Start EVERY shell command with `cd <that path> &&`. Never touch the main checkout (another session's uncommitted work), and never read from or write to .claude/worktrees/model-config-p8 (Phase 8's workflow edits it).
- Python: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python (the worktree has no venv); run pytest FROM the worktree cwd. Tests/Architecture needs -p no:xdist.
- NEVER write ~/.config/tldw_cli or ~/.local/share/tldw_cli. Never bypass test isolation (private_profile_test, TLDW_TEST_PRIVATE_PROFILE_NODE, HOME/XDG/TLDW_CONFIG_PATH). Live runs: scratch TLDW_CONFIG_PATH with a unique users_name, FULL SCREEN 211x44 (primary) and 235x52.
- ADR-126 RecoveryRequired in a clean worktree is environmental: compare failure-NAME sets against origin/dev.
- Size ratchets never rise (ADR-097); console_settings_modal.py net lines <= 0 against its current row (see the size facts above for settings_screen.py and providers_models_card.py).
- Geometry only through tokens in css/core/_variables.tcss (ADR-150/161); rebuild the CSS bundle with the repo script.
- ADR-031: never bind Ctrl+C/V/X/S/D/Z/A/R/W; footer hints must match working bindings.
- TDD; real-implementation tests for config/provider surfaces; rewrite pinned tests on purpose and name them.
- Commit per rider `fix|feat|test(model-config): <summary> (<rider id>)`, using the rider's own id (for example TASK-33006.13 or TASK-33666), + `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never push; NEVER merge origin/dev into the branch.
- Tick ACs, set Done, add Implementation Notes in each rider's task file. Update Docs/User_Guide CONTENT; never add "Verified against" paragraphs (record verification in task notes).
- Lessons (backlog/docs/lessons-*.md): insert new entries MID-FILE near a related section, never appended at the end of the file (dev appends there constantly and every rebase conflicts).
- Evidence: commit captures (.txt/.ansi.txt) under qa/, but NEVER commit one-off driver or probe scripts (.sh/.py with machine-specific paths); describe the capture procedure in a short qa README instead. Captures are reviewed by AI reviewers, so they must not contradict the PR's claims.
- Scratch hygiene: every scratch directory carries a unique name with your task and sha (e.g. $SP/p9-t<M>-base-<sha>); never generic names (base, head, tree, cfgroot, devtree, neg); never rm -rf a directory you did not create; remove your own trees when done (git worktree remove for worktrees).
- Boot census: a NEW module only user actions need is imported inside the function that uses it; measure Tests/Performance/test_ui_ready_module_census.py with PYTHONPATH=<worktree> (the venv editable install otherwise measures the MAIN checkout); the limit must not rise.
- Before DONE: covering tests + `PYTHON=<venv> ./scripts/preflight.sh` (rc 0; never piped through tail).
- Riders' line numbers are stale: locate code by symbol. A rider whose premise turns out false at HEAD is closed with the evidence in its notes, not forced.

## Owner speed rules (2026-10-04, binding; they OVERRIDE the Evidence and Before-DONE bullets above)

- Fix EVERY review finding, Minor included; fix rounds are not capped.
- Tests always run in parallel (xdist, `-n 6 --dist loadfile`); serial only for Tests/Architecture.
- Per task, run the test files that cover what you changed, once, at HEAD. The phase runs ONE base-vs-head comparison at its end, both trees at once.
- Evidence is the test names plus the command and its summary line in the report. Commit no per-task captures or evidence files; the Capture phase captures every changed surface once.
- Execution order: six waves, run in order: Wave 1 = Tasks 1, 3, 4, 5; Wave 2 = Tasks 9, 10, 11, 13; then the rebase onto Phase 8's final head; Wave 3 = Task 2 alone; Wave 4 = Tasks 6, 7, 8; Wave 5 = Tasks 12, 14, 17; Wave 6 = Tasks 15, 16. Waves 1 and 2 hold only files Phase 8 does not edit. console_settings_modal.py goes 11, then 6, then 15; settings_screen.py goes 2, then 7, then 12; chat_screen.py goes 2, then 8. Inside a wave, the tasks run in parallel git worktrees and are cherry-picked together, so their files are disjoint (each task's References list what it touches, and a 'Must not edit' line names the shared files it leaves alone). Each wave holds at most one task on each of settings_screen.py, chat_screen.py and console_settings_modal.py. Before Wave 3, rebase this branch onto Phase 8's final head; it is stacked on Phase 8's in-progress branch, which is stacked on Phase 7 (PR #3057). At each wave's cherry-pick, resolve conflicts in the generated CSS bundle by re-running tldw_chatbook/css/build_css.py, never by hand.

## Task 1: Repair the red modal-inventory, endpoint-creation and anchor-geometry tests (TASK-33003.19, TASK-33006.17, TASK-33006.24)

Task files:
- backlog/tasks/task-33003.19 - The-five-drift-reds-in-test_console_modal_dismissal.py-pass-again.md
- backlog/tasks/task-33006.17 - Repair-the-endpoint-creation-and-redirected-click-tests-that-are-red-on-dev.md
- backlog/tasks/task-33006.24 - Steady-the-suspended-anchor-geometry-test.md
Wave: 1
Depends on: —
Must not edit: console_model_popover.py (Task 2 owns it): fix the [ConsoleModelPopover] contract case on the test side.

### Why

- TASK-33003.19 (The five drift reds in test_console_modal_dismissal.py pass again): Five nodes in Tests/UI/test_console_modal_dismissal.py are still red at HEAD, with the same assertions as on dev:
  - the launch declarations: new launches such as ChangeRevertConfirmModal, ConsoleAgentHistoryModal, ConsoleAgentProgressModal, ConsoleEndpointTemplateModal and nested ConfirmationDialog launches;
  - the inventory walk: a set mismatch;
  - the contract table: 16 == 15;
  - [ConsoleModelPopover]: `__init__` is missing 8 keyword-only arguments;
  - test_settings_redirected_select_click_uses_real_mro_dispatch, whose click now lands outside the modal rather than on a zero-size region.
  The sixth red that the rider defers ([escape]) now passes.
  Measured: `pytest Tests/UI/test_console_modal_dismissal.py Tests/UI/test_component_pattern_governance.py -n 6 --dist loadfile` gave 8 failed, 130 passed. Five of the failures are here; the three governance failures belong to Task 10.
  The launch inventory drifts with every new Console modal on dev, so re-measure it after each rebase.
  Restack onto Phase 8's final head (2026-10-16): the launch-declarations and inventory nodes were red again, on four modals dev added (ResponseRulesModal and three plain Goals modals), and were brought current before Wave 3. Each one is declared by the owner that opens it, as 76918d315f did; the rider's notes have the detail. The file is 120 passed at HEAD.
- TASK-33006.17 (Repair the endpoint-creation and redirected-click tests that are red on dev): Both nodes are still red: '3 failed in 34.88s'.
  - (a) test_create_endpoint_with_live_controller_rebase_settles[*] fails on `modal._current_draft_discovery_identity().provider_key == provider_id`: 'llama_cpp' != 'custom-ep:new-live-endpoint'. Every earlier assert passes. Since TASK-32566 (1b02010541) an entry draft keeps the family key in provider_key and the custom-ep id in entry_id, so the test probably means entry_id. Confirm that before rewriting.
    Restack onto Phase 8's final head (2026-10-16): (a) is fixed by the base. Dev's 8c033d006b (PR #3100, filed under TASK-32566) made the same rewrite, so the restack dropped this branch's edit of that file. Both params pass serially at HEAD and at origin/dev 111f8e48fc. (b) is still this branch's fix (2bdc8c65a6).
  - (b) In the redirected-click node, the Streaming Select is at Region(x=120, y=25, w=1, h=2), outside the modal content Region(x=20, y=3, w=100, h=54) at 140x60. Phase 7's single control width ($ds-size-13, 0e7aa920f9) may squeeze that row, so decide whether this is a real layout defect before touching the test.
  Node (b) is also one of 33003.19's five reds.
- TASK-33006.24 (Steady the suspended-anchor geometry test): test_new_focus_supersedes_pending_suspended_anchor_and_is_revealed cannot be measured today. Tests/UI/test_console_settings_geometry.py has no private_profile_test or bootstrap_profile marker, so under supported isolation every node errors at setup with RecoveryRequired raw_source_selection_changed (8 of 8 serial, and the whole file under -n 6). First make the file run under supported isolation, then measure the 20 serial runs.

### Acceptance criteria

- [ ] TASK-33003.19 #1 The five drift nodes pass, each by updating the inventory or contract to the current modals or by fixing the modal it names.
- [ ] TASK-33003.19 #2 No other node in the file changes state.
- [ ] TASK-33006.17 #1 Each node passes serially on dev, or is replaced by a test of the same guarantee named in the task notes
- [ ] TASK-33006.17 #2 The cause of each failure is recorded
- [ ] TASK-33006.24 #1 The test passes 20 of 20 serial runs at the branch head
- [ ] TASK-33006.24 #2 The cause is recorded, and the fix does not weaken what the test asserts

### References

- Tests/UI/test_console_modal_dismissal.py
- Tests/UI/test_console_endpoint_discovery.py
- Tests/UI/test_console_settings_geometry.py
- tldw_chatbook/Widgets/Console/console_settings_modal.py (this wave's modal slot; only for a real defect, net lines <= 0)
- tldw_chatbook/css/features/_console_panels.tcss (only if (b) is a real squeeze)
- Tests/private_profile.py (read only)

## Task 2: Settings and the Console read one readiness truth (TASK-33001.8, TASK-33002.9, TASK-33005.7, TASK-33005.11, TASK-33005.15)

Task files:
- backlog/tasks/task-33001.8 - New-chat-fallback-borrows-another-providers-model.md
- backlog/tasks/task-33002.9 - Provider-Test-rows-lead-with-an-unlisted-model-and-name-key-sources-in-plain-words.md
- backlog/tasks/task-33005.7 - Bring-the-remaining-readiness-copy-onto-the-one-vocabulary.md
- backlog/tasks/task-33005.11 - Never-read-verified-for-a-model-the-key-check-did-not-list.md
- backlog/tasks/task-33005.15 - Keep-the-Settings-Overview-readiness-line-in-step-with-the-Console.md
Wave: 3
Depends on: —

### Why

- TASK-33001.8 (New-chat provider/model fallback borrows chat_defaults.model across providers): provider_model_resolution.py::resolve_effective_provider_model still falls back to chat_defaults.model whatever provider is selected.
  - Console: it feeds chat_screen.py::_effective_console_provider_model, which feeds session.py::_default_console_session_settings (Ctrl+T).
  - Settings: settings_screen.py::_resolve_provider_model_for_settings uses it, and _apply_navigation_provider_context and _provider_display_setting_values pair a navigated provider with the default model.
  Reuse the identity rule in Chat/console_session_settings.py::resolve_effective_chat_configuration (`_canonical_chat_provider_id`). The palette producer is gone (TASK-33001.6), so AC#2's test must set a control provider with no model through a path that still exists.
- TASK-33002.9 (Provider Test rows lead with an unlisted model and name key sources in plain words): settings_screen.py::_provider_test_rows picks its lead row in this order: not ready, unreachable, no model, failed generation, otherwise 'Config'. A model the server did not list therefore never leads. `_apply_provider_endpoint_probe_outcome` still toasts at information severity when the probe passed and the server is reachable. AC#2 is already met, so it needs only a pin test.
- TASK-33005.7 (Bring the remaining readiness copy onto the one vocabulary): Four surfaces still describe readiness in their own words:
  - settings_screen.py::_provider_readiness_label returns 'Provider readiness: <name> / <model>' with no readiness word. providers_models_card.py calls it too.
  - chat_screen.py's inspector row is ConsoleDisplayRow('Setup', 'Provider configuration required').
  - console_display_state.py maps any 'api key' setup reason to 'Send blocked — add an API key to continue'. A rejected key is the credential_rejected blocker in console_session_settings.py.
  - console_onboarding_state.py's CONSOLE_READY_EMPTY_COPY and _ready_line_copy are unqualified.
- TASK-33005.11 (Never read verified for a model the key check did not list): console_session_settings.py::_ready_words returns 'Ready · verified HH:MM' for any accepted listing and never reads the evidence's model_ids. Settings' Readiness row (verdict_readiness_words in settings_screen.py) therefore sits above a Model row that says 'not listed by the server'. Fix it in the shared word builder, so that the Console status row, the rail and the switcher agree (AC#3).
- TASK-33005.15 (Keep the Settings Overview readiness line in step with the Console): Three gaps:
  - settings_screen.py::_provider_overview_readiness_status forces the displayed model ('never a provider-table fallback'), so the Overview can disagree with a new chat's fallback. Settle it with 33001.8's identity rule.
  - The Overview refreshes only through the 0.25 s _poll_subscription_readiness poll, keyed on (category, provider, subscription_status), so an evidence change never repaints it. Add the evidence revision to that poll's key instead of adding a timer.
  - No test types a provider name and checks the switcher's NOT RUNNING group.

### Acceptance criteria

- [ ] TASK-33001.8 #1 `resolve_effective_provider_model` returns `chat_defaults.model` only when the selected provider canonically equals `chat_defaults.provider` (the same identity rule TASK-33001.1 applies), otherwise no model
- [ ] TASK-33001.8 #2 A new Console tab (Ctrl+T) opened while the control provider has no model gets that provider's own configured model, or none, never another provider's model; a regression test drives the real session-default path
- [ ] TASK-33001.8 #3 Settings' provider navigation without an explicit model shows the navigated provider's own model, or none, never `chat_defaults.model` of another provider
- [ ] TASK-33001.8 #4 The new-chat pair for the ordinary cases (no control selection; a control provider with its model) is unchanged
- [ ] TASK-33002.9 #1 A model the server does not list leads the rows with a next step, and the toast uses warning severity
- [ ] TASK-33002.9 #2 No Credentials or Key row prints an `env:` or `config:` spelling; each names its source in words (saved in config, from env var NAME) [Already true since 99f1ce4884 (the Key row says 'from env var NAME' or 'saved in config'); add the pin test only.]
- [ ] TASK-33005.7 #1 Settings' "Readiness: <provider> / <model>" identity rows read the same word as the Readiness row for that connection
- [ ] TASK-33005.7 #2 The Inspector's setup line for an unreachable server names the failure (for example 'refused :9099') instead of 'Provider configuration required'
- [ ] TASK-33005.7 #3 After a rejected key the composer hint says the key was rejected, not 'add an API key'
- [ ] TASK-33005.7 #4 The empty-transcript 'Ready — type a message to begin.' line uses the qualified word ('Ready · not tested', 'Ready · reachable HH:MM', ...)
- [ ] TASK-33005.7 #5 Each surface is pinned against readiness_words for the same readiness value
- [ ] TASK-33005.11 #1 When an accepted key check did not list the chosen model, Settings' Readiness row and Model row do not contradict each other: the word for that model is not 'Ready · verified', or the Model row explains why it still is
- [ ] TASK-33005.11 #2 Switch model rows read 'verified' only for models the accepted listing returned; the provider's other rows keep a weaker word
- [ ] TASK-33005.11 #3 The Console status row, rail Model line and switcher agree with Settings for the same provider and model at the same moment
- [ ] TASK-33005.11 #4 Tests cover a listed and an unlisted model under one accepted key check, with a negative control
- [ ] TASK-33005.15 #1 With a blank chat-default model and a model in the provider table, the Settings Overview and a new Console chat give the same readiness words; a test pins it.
- [ ] TASK-33005.15 #2 A connection-evidence change made elsewhere while the Overview is visible updates its status line without leaving the screen, and adds no new timer (reuse an existing poll or event); a test pins it.
- [ ] TASK-33005.15 #3 Typing a provider name in Switch model's Find filters the NOT RUNNING group the same way as the other groups; a test pins it.

### References

- tldw_chatbook/UI/Screens/settings_screen.py (this wave's Settings slot; zero headroom)
- tldw_chatbook/UI/Screens/chat_screen.py (this wave's Console slot)
- tldw_chatbook/UI/Screens/provider_model_resolution.py
- tldw_chatbook/Chat/console_session_settings.py
- tldw_chatbook/Chat/console_display_state.py
- tldw_chatbook/Chat/console_onboarding_state.py
- tldw_chatbook/UI/Settings_Modules/providers_models_card.py (zero headroom)
- tldw_chatbook/Widgets/Console/console_model_popover.py
- tldw_chatbook/UI/Console_Modules/session.py (read only)
- Tests/Provider/test_provider_model_resolution.py
- Tests/Chat/test_readiness_words.py
- Tests/UI/test_settings_configuration_hub.py
- Tests/UI/test_settings_provider_test_draft.py
- Tests/UI/test_console_switcher_local_probe.py
- qa/model-config-33005-captures/README.md
- backlog/decisions/006-provider-aware-generation-settings.md

## Task 3: Boundaries and frames reach 3:1, and theme values stay paintable (TASK-33003.15, TASK-33003.16, TASK-33003.17, TASK-33641.1)

Task files:
- backlog/tasks/task-33003.15 - Library-pane-frames-reach-3-1-against-their-surfaces.md
- backlog/tasks/task-33003.16 - The-theme-sanitiser-drops-auto-NN-on-background-used-theme-variables.md
- backlog/tasks/task-33003.17 - An-unchecked-SelectionList-row-under-the-cursor-never-shows-the-checked-glyph.md
- backlog/tasks/task-33641.1 - App-chrome-lines-and-Collapsible-frames-reach-3-1-against-their-surfaces.md
Wave: 1
Depends on: —

### Why

- TASK-33003.15 (Library pane frames reach 3:1 against their surfaces): css/core/_variables.tcss still defines `$ds-column-line: $surface-lighten-2`. TASK-33003.6 moved the grid-line and control-edge tokens to $tldw-boundary, but not this one. css/features/_library.tcss uses it three times, and Tests/UI/test_theme_contrast.py has no column-line gate. $ds-column-line is also used by _home, _console (the rail rules 33641.1 names), _settings, _lab, _console_panels and _workflows, so a token change reaches all of them: measure each one.
- TASK-33003.16 (The theme sanitiser drops 'auto NN%' on background-used theme variables): css/Themes/themes.py::_is_safe_variable_value accepts any `auto NN%` value whatever the variable, and sanitize_theme_variables keeps it, but Textual's process_color accepts auto only for non-background properties. The fix needs the set of variable names that Textual's CSS uses as backgrounds; drop auto only for those, with a warning.
- TASK-33003.17 (An unchecked SelectionList row under the cursor never shows the checked glyph): css/components/_lists.tcss scopes `#settings-discovered-models-list .selection-list--button` and `-button-selected`, but `-button-highlighted` still falls to the shared `SelectionList > .selection-list--button-highlighted` rule. So the row under the cursor shows the checked glyph for an unchecked item.
- TASK-33641.1 (App chrome lines and Collapsible frames reach 3:1 against their surfaces): Three of the four lines are located:
  - css/components/_widgets.tcss still has `Collapsible { border: round $surface-lighten-1; }`. It is a bare-type rule, so AC#3's bare-type ratchet applies.
  - The Console rail section rules are `border-top: solid $ds-column-line` in css/features/_console.tcss.
  - The top tab-bar separator and the active-tab outline are in neither layout/_tabs.tcss nor components/_navigation.tcss. Locate them first; the candidates are UI/Navigation/main_navigation.py's MainNavigationBar DEFAULT_CSS and css/layout/_sidebars.tcss (`border-bottom: solid $surface-lighten-1`).

### Acceptance criteria

- [ ] TASK-33003.15 #1 Library pane frames measure at least 3:1 against surface and panel on every shipped theme whose colours resolve, gated by a test in the style of Tests/UI/test_theme_contrast.py.
- [ ] TASK-33003.15 #2 A live 211x44 capture of the Library screen records the measured ratios in the task notes.
- [ ] TASK-33003.16 #1 A user theme that sets 'auto NN%' on a background-used variable loads and applies without a stylesheet error; the value is dropped with a warning.
- [ ] TASK-33003.16 #2 'auto NN%' on a text-colour variable is still kept.
- [ ] TASK-33003.16 #3 A load_user_themes test and a Settings theme-import test cover both cases.
- [ ] TASK-33003.17 #1 Under the cursor, a checked row and an unchecked row differ by a glyph or by at least 3:1, and an unchecked row never shows the checked glyph.
- [ ] TASK-33003.17 #2 A rendered test under the production stylesheet measures both states.
- [ ] TASK-33641.1 #1 The top tab-bar separator, the active top tab's outline, the shared Collapsible frame and the Console rail's section rules each measure at least 3:1 against the surfaces they sit on, on every shipped theme whose colours resolve. A test in the style of Tests/UI/test_theme_contrast.py gates this.
- [ ] TASK-33641.1 #2 Live 211x44 and 235x52 captures of Console and Settings (with `capture-pane -e`) record the measured ratios in the task notes.
- [ ] TASK-33641.1 #3 The boot CSS byte budget and the bare-type CSS rule ratchet do not rise.

### References

- tldw_chatbook/css/core/_variables.tcss
- tldw_chatbook/css/features/_library.tcss
- tldw_chatbook/css/components/_lists.tcss
- tldw_chatbook/css/components/_widgets.tcss
- tldw_chatbook/css/features/_console.tcss
- tldw_chatbook/UI/Navigation/main_navigation.py or tldw_chatbook/css/layout/_sidebars.tcss (wherever the tab-bar lines live)
- tldw_chatbook/css/Themes/themes.py
- tldw_chatbook/css/build_css.py (regenerate the bundle)
- Tests/UI/test_theme_contrast.py
- Tests/Utils/test_user_theme_loader.py
- Tests/UI/test_settings_theme_import.py
- Tests/Performance/test_boot_css_byte_budget.py
- qa/model-config-33003-captures/settings-rail-inactive-focus-211x44.ansi.txt

## Task 4: Guarded config IO on the UI loop sees only its own task's raw scope (TASK-33641.6)

Task files:
- backlog/tasks/task-33641.6 - Guarded-config-IO-on-the-UI-loop-is-not-refused-while-another-task-holds-a-raw-scope.md
Wave: 1
Depends on: —
Must not edit: chat_screen.py (Task 2 owns it in this wave). Leave the discovery thread hop in place, and say in the notes whether it can now go.

### Why

- TASK-33641.6 (Guarded config IO on the UI loop is not refused while another task holds a raw scope): Backup_Recovery/raw_participants.py still keeps `_local = threading.local()`. `_scope`, `_runtime_operation` and `_check` read `_local.operation` and compare it with storage_admission.py::_task_identity(). So a raw config scope held across an await on the Textual loop makes every other task's guarded IO raise RecoveryRequired('raw_operation_provenance_invalid'). chat_screen.py::_discover_local_servers_for_setup_card still hops to its own thread and loop to avoid this (`asyncio.to_thread(asyncio.run, …)`, bdec9a3f8e).
  AC#1 is an investigation: name the scope held across an await at boot. A task-aware lookup, for example a ContextVar, must keep the owner task's ADR-126 provenance check exactly as strict.
  This is shared recovery infrastructure, so it gets its own task. Task 14 re-runs Tests/Chat/test_console_settings_defaults.py after it lands, because that file's red node logs RecoveryRequired on a guarded read.

### Acceptance criteria

- [ ] TASK-33641.6 #1 The raw scope held across an await on the UI loop during boot is named (file and function) and reproduced, without the discovery thread hop masking it.
- [ ] TASK-33641.6 #2 Guarded config IO from a second UI-loop task in that window is no longer refused with raw_operation_provenance_invalid, because the scope is no longer held across an await or the raw-operation lookup is task-aware; the ADR-126 provenance check for the task that owns the operation is not weakened.
- [ ] TASK-33641.6 #3 A test runs two asyncio tasks on one thread, one holding a registered raw config scope across an await, and asserts the other task's guarded config read is neither refused nor attributed to the first task's operation.

### References

- tldw_chatbook/Backup_Recovery/raw_participants.py
- tldw_chatbook/Backup_Recovery/storage_admission.py
- Tests/Backup_Recovery/test_raw_participant_lifetimes.py (or a new file beside it)
- tldw_chatbook/UI/Screens/chat_screen.py (read only)
- backlog/docs/lessons-live-verification.md
- backlog/decisions/ (ADR-126)

## Task 5: An enabled recurring reminder keeps showing its next run (TASK-33641.4)

Task files:
- backlog/tasks/task-33641.4 - An-enabled-recurring-reminder-whose-last-run-completed-shows-its-next-run.md
Wave: 1
Depends on: —

### Why

- TASK-33641.4 (An enabled recurring reminder whose last run completed shows its next run): scheduling/task_detail.py::_format_next_run returns '—' whenever `_task_status(task) == TaskStatus.COMPLETED`, and it checks this before it reads next_run_at. `_task_status` falls through to the underlying status, so an enabled recurring reminder whose last dispatch completed hides its next run. Only a fired one-time reminder (scheduling/unified_rows.py::reminder_has_fired) should take that branch. The queue column reuses _format_next_run through schedules_workbench.py.

### Acceptance criteria

- [ ] TASK-33641.4 #1 An enabled recurring reminder with a stored next run shows that next run (absolute and relative) in the task detail and the queue column, whatever its last dispatch status.
- [ ] TASK-33641.4 #2 A fired one-time reminder still shows "—", and disabled and paused reminders keep "— (disabled)" and "— (paused)".
- [ ] TASK-33641.4 #3 Unit tests cover all four cases, and the recurring case fails before the fix.

### References

- tldw_chatbook/UI/Screens/scheduling/task_detail.py
- tldw_chatbook/UI/Screens/scheduling/unified_rows.py (read only)
- Tests/UI/test_schedules_next_run_relative.py
- Tests/UI/test_schedules_disabled_state.py
- Tests/UI/test_schedules_workbench.py

## Task 6: Chat settings open focus, Endpoint row and served listings follow the connection (TASK-33006.10, TASK-33006.13, TASK-33006.14, TASK-33006.19, TASK-33006.25)

Task files:
- backlog/tasks/task-33006.10 - Open-focus-falls-back-when-Temperature-is-hidden-and-its-helper-is-named-for-what-it-does.md
- backlog/tasks/task-33006.13 - The-Endpoint-row-sync-finds-its-parts-by-id-not-by-child-position.md
- backlog/tasks/task-33006.14 - Open-focus-on-a-blocked-chat-is-tested-when-Connection-starts-collapsed.md
- backlog/tasks/task-33006.19 - Served-model-listings-are-kept-per-endpoint-not-per-provider.md
- backlog/tasks/task-33006.25 - Open-pick-mode-after-a-new-endpoints-listing-even-if-the-close-guard-appeared.md
Wave: 4
Depends on: —

### Why

- TASK-33006.10 (Open focus falls back when Temperature is hidden, and its helper is named for what it does): console_settings_field_row.py::_focus_highest_priority_connection focuses Temperature only when it can take focus, and otherwise focuses nothing. The helper now picks the open focus for both views, so its name no longer fits. The rename touches the definition and its 2 self-calls, console_settings_modal.py (3 call sites), console_settings_saved_defaults.py (1) and Tests/UI/test_console_session_settings.py (1 direct call).
- TASK-33006.13 (The Endpoint row sync finds its parts by id, not by child position): _sync_endpoint_row unpacks the Endpoint row's children into exactly three names.
- TASK-33006.14 (Open focus on a blocked chat is tested when Connection starts collapsed): No test reaches the `if disclosure.collapsed` branch. A blocked chat composes Connection expanded, and test_not_ready_chat_opens_connection_on_its_recovery_action asserts that. This rider is test-only.
- TASK-33006.19 (Served-model listings are kept per endpoint, not per provider): console_settings_modal.py keys `_served_models` by provider only. It is written in the modal's listing-result handler and in field_row::_list_then_pick, read in _open_model_picker, and never cleared when the Endpoint changes. model_switcher.py takes `served` per provider too. The modal already computes _discovery_connection_identity, which can key it instead.
- TASK-33006.25 (Open pick mode after a new endpoint's listing even if the close guard appeared): field_row::_list_then_pick calls _open_model_picker after the listing, and _open_model_picker returns silently while #console-settings-close-guard is displayed. The AC lets you either defer pick mode until the guard is dismissed or show a notice.

### Acceptance criteria

- [ ] TASK-33006.10 #1 When Temperature is hidden or unfocusable, the Model view opens with focus on the next shown field (or Change), never on nothing
- [ ] TASK-33006.10 #2 The helper is renamed to say it picks the view's open focus, and every caller and test uses the new name
- [ ] TASK-33006.13 #1 The sync looks up the label, the Base URL input and New endpoint… by id or class
- [ ] TASK-33006.13 #2 A row with an extra child still syncs (pinned by a test)
- [ ] TASK-33006.14 #1 A test reaches the recovery focus with Connection collapsed and shows it opens and focuses the fix
- [ ] TASK-33006.14 #2 Removing the branch fails that test
- [ ] TASK-33006.19 #1 A listing's models are offered only for the endpoint that listed them
- [ ] TASK-33006.19 #2 Changing the Endpoint drops the old listing from pick mode (pinned by a test)
- [ ] TASK-33006.25 #1 When the close guard is showing as a new endpoint's listing finishes, the user gets a clear outcome: pick mode opens once the guard is dismissed, or a notice says the new entry was saved and how to pick it (Change, Alt+M). A test pins whichever is chosen.

### References

- tldw_chatbook/Widgets/Console/console_settings_field_row.py
- tldw_chatbook/Widgets/Console/console_settings_modal.py (this wave's modal slot; net lines <= 0)
- tldw_chatbook/Widgets/Console/console_settings_saved_defaults.py
- tldw_chatbook/UI/Console_Modules/model_switcher.py
- Tests/UI/test_console_settings_disclosures.py
- Tests/UI/test_console_endpoint_discovery.py
- Tests/UI/test_console_session_settings.py (the one renamed call only)

## Task 7: Settings state line, field search, saved Test rows and focus after Revert (TASK-33002.10, TASK-33002.11, TASK-33003.14, TASK-33005.8, TASK-33007.11)

Task files:
- backlog/tasks/task-33002.10 - Settings-says-in-the-app-that-switching-categories-keeps-a-draft.md
- backlog/tasks/task-33002.11 - Image-Gen-and-Video-Gen-unsaved-count-counts-fields-not-draft-keys.md
- backlog/tasks/task-33003.14 - Field-search-can-reach-Console-Behaviors-generation-fallbacks.md
- backlog/tasks/task-33005.8 - Re-render-Settings-provider-test-rows-on-Save.md
- backlog/tasks/task-33007.11 - Settings-keeps-keyboard-focus-off-the-navigation-bar-after-the-Revert-dialog-closes.md
Wave: 4
Depends on: —
Must not edit: Tests/UI/test_settings_image_gen_panel.py and Tests/UI/test_settings_video_gen_defaults.py (Task 10 owns them in this wave): put the 33002.11 pins in Tests/UI/test_settings_state_line_unsaved_count.py.

### Why

- TASK-33002.10 (Settings says in the app that switching categories keeps a draft): The State line says 'Draft kept when you leave' only for ADVANCED_CONFIG. No other draft category says it in the app, and UI/Workbench/help.py has no Settings draft text. AC#2 keeps the State line to one row at 211x44, so the footer, F1 help or the guided-action row is the likelier home.
- TASK-33002.11 (Image Gen and Video Gen unsaved count counts fields, not draft keys): The fallback branch of settings_screen.py::_category_unsaved_count returns len(draft.dirty_keys). Image Gen stages both `field::<b>::<k>` and `cleared::<b>::<k>` for one field, so a single edit can count as 2.
- TASK-33003.14 (Field search can reach Console Behavior's generation fallbacks): In settings_search_index.py, FIELD_SEARCH_INDEX[CONSOLE_BEHAVIOR] gives its `settings-console-default-*` rows exactly the labels of the Providers & Models rows. SettingsScreen._top_field_match is a label-substring match and Enter opens the top category, so '/' never lands on a fallback. A qualified alias label in settings_search_index.py may avoid touching the screen.
- TASK-33005.8 (Re-render Settings provider test rows on Save): When a save preserves the test evidence, the save path skips _mark_provider_test_result_stale, so _provider_test_result keeps rows built from the pre-save dirty set, still marked '(draft)'. The Phase 7 card reads the first row through providers_models_card.py::key_check_verdict. Rebuilding the rows with an empty dirty set after such a save covers both.
- TASK-33007.11 (Settings keeps keyboard focus off the navigation bar after the Revert dialog closes): settings_screen.py::on_screen_resume restores no focus, and Settings names no widget to receive it. After Esc, r and the Discard dialog, the framework therefore focuses the navigation bar's Home button. Phase 8 Task 1 adds return-action focus handling in the same file; build on it.

### Acceptance criteria

- [ ] TASK-33002.10 #1 An in-app surface on the draft categories (footer, F1 help or guided-action row) says that switching categories keeps the draft
- [ ] TASK-33002.10 #2 The State line still fits one row at 211x44
- [ ] TASK-33002.11 #1 On Image Gen and Video Gen the count equals the number of fields that differ from their saved values, and drops when a field is set back by hand
- [ ] TASK-33002.11 #2 A test pins the count for one edit that touches two draft keys
- [ ] TASK-33003.14 #1 A '/' query can land on each Console Behavior generation fallback, for example through a qualified query or a choice between the two matches.
- [ ] TASK-33003.14 #2 '/' results for Providers & Models fields are unchanged, and Tests/UI/test_settings_search_index.py stays green.
- [ ] TASK-33005.8 #1 After Save, the provider test rows drop '(draft)' and show the saved values without a return visit
- [ ] TASK-33005.8 #2 A test drives 't', Save and asserts the rows on the same visit
- [ ] TASK-33007.11 #1 After Esc, r, then Discard changes in a Settings category, keyboard focus is not on the app navigation bar, and a following Enter does not leave Settings.
- [ ] TASK-33007.11 #2 After Esc, r, then Keep editing, the same holds and the unsaved edits are still there.
- [ ] TASK-33007.11 #3 After either button, Settings' single-key shortcuts (s, r) work without another Esc.
- [ ] TASK-33007.11 #4 Mounted tests with the real stylesheet pin #1 to #3 for a text field in at least two Settings categories.

### References

- tldw_chatbook/UI/Screens/settings_screen.py (this wave's Settings slot; zero headroom)
- tldw_chatbook/UI/Screens/settings_search_index.py
- tldw_chatbook/UI/Screens/settings_image_gen_defaults.py
- tldw_chatbook/UI/Screens/settings_video_gen_defaults.py
- tldw_chatbook/UI/Settings_Modules/providers_models_card.py (zero headroom)
- tldw_chatbook/UI/Workbench/help.py (only if F1 help is the home)
- tldw_chatbook/Widgets/confirmation_dialog.py (read only)
- Tests/UI/test_settings_state_line_unsaved_count.py
- Tests/UI/test_settings_search_index.py
- Tests/UI/test_settings_provider_test_draft.py
- Tests/UI/test_settings_connect_rows.py
- qa/model-config-p5-2026-10-01/task-4/README.md

## Task 8: Console chrome: literal mode-strip text and a fresh footer after Chat settings closes (TASK-33002.19, TASK-33006.20)

Task files:
- backlog/tasks/task-33002.19 - Console-mode-strip-raises-MarkupError-on-a-markup-like-default-provider.md
- backlog/tasks/task-33006.20 - The-app-footer-refreshes-after-a-modal-closes-so-Alt-A-is-not-pushed-off-at-211-columns.md
Wave: 4
Depends on: —
Must not edit: console_settings_modal.py (Task 6 owns it in this wave). Fix the footer from Chat settings' close callback in chat_screen.py or in a UI/Console_Modules module.

### Why

- TASK-33002.19 (Console mode strip raises MarkupError on a markup-like default provider): UI/Workbench/workbench_widgets.py renders the mode label as markup in two places: `_build_mode` (`Static(mode.label, …)`) and `_sync_mode_label` (`label.update(mode.label)`). The text comes from console_display_state.py's provider label through provider_catalog.py::provider_display_name, which returns a dangling custom-ep key unchanged. Fix it once at the sink (markup off, or escape in both places), after checking that no mode label relies on markup.
- TASK-33006.20 (The app footer refreshes after a modal closes so Alt+A is not pushed off at 211 columns): Tests/UI/test_console_switch_model_entry_points.py::_close_chat_settings still closes through the Context view's Cancel, because Esc leaves the rail-focus footer that pushes Alt+A off at 211 columns. The likely fix is a bindings refresh after focus is restored; today the only refresh_bindings() call is in UI/Console_Modules/composer_run_controls.py.

### Acceptance criteria

- [ ] TASK-33002.19 #1 Console opens on a `custom-ep:gone[/]` default with no MarkupError, and the mode strip shows the provider id as literal text
- [ ] TASK-33002.19 #2 Any other markup-like provider id or custom endpoint display name reaching the mode strip renders literally
- [ ] TASK-33002.19 #3 A mounted private-profile Console test on that config fails before the fix and passes after
- [ ] TASK-33006.20 #1 After Chat settings closes by Esc, the footer shows the hints of the focused widget, with Alt+A visible at 211x44
- [ ] TASK-33006.20 #2 The entry-points test closes with Esc again

### References

- tldw_chatbook/UI/Workbench/workbench_widgets.py
- tldw_chatbook/UI/Screens/chat_screen.py (this wave's Console slot)
- tldw_chatbook/Chat/console_display_state.py (read only)
- tldw_chatbook/Chat/provider_catalog.py (read only)
- Tests/UI/test_console_switch_model_entry_points.py
- a new mounted private-profile mode-strip test

## Task 9: Every credential reader honours credential_source (TASK-33001.14, TASK-33001.15)

Task files:
- backlog/tasks/task-33001.14 - Legacy-summarizers-send-an-env-key-for-a-provider-whose-credential-source-is-none.md
- backlog/tasks/task-33001.15 - Legacy-API-only-key-is-not-materialised-when-a-model-save-records-stored.md
Wave: 2
Depends on: —
Must not edit: settings_screen.py (Task 7 owns it in this wave). The fix belongs in provider_setup_persistence.py. If the call site really has to change, stop and report rather than editing the screen.

### Why

- TASK-33001.14 (Legacy summarizers send an env key for a provider whose credential source is none): LLM_Calls/Local_Summarization_Lib.py::_resolve_provider_credential, used by the koboldcpp and tabbyapi summarizers, falls back to os.environ[api_key_env_var] and never reads credential_source. Chat/provider_readiness.py::resolve_provider_credential honours 'none'. Keep the function's distinction between a configured blank and an absent key (its comment says tests pin it), and keep the TASK-34100.4 ciphertext handling.
  REOPENED at the restack onto Phase 8's final head (2026-10-16). The claim that the base fixed it does not hold at HEAD. The restack dropped this rider's code, tests, ADR-012 amendment and guide text. Dev's task-34895 (a79b07713e, PR #3105) now sends KoboldCpp and TabbyAPI summaries through `_summarize_via_chat_api` -> `chat_api_call`. That route reads no api_key_env_var, so an exported variable is not sent under "none". But it reads no credential_source either:
  - it still sends a legacy [API] key under "none";
  - it differs from readiness for "stored" (TabbyAPI), "environment" and no-source sections;
  - no test or ADR-012 record covers any of this.
  Local_Summarization_Lib.py is still in the tree with the unfixed resolver and no production importer. The fix now belongs on the chat-dispatch route. It needs a wave: Task 17 (Wave 5) holds the same files and the loader half of the same ruling (TASK-33641.11). The rider's notes have the measured table.
- TASK-33001.15 (Legacy [API]-only key is not materialised when a model save records "stored"): In Chat/provider_setup_persistence.py::build_provider_setup_mutation, the credential_source == 'stored' branch, with keep_stored_credential and no typed value, validates the existing value but writes nothing. A key that config.py::_normalize_legacy_provider_api_key bridged in memory from legacy [API] therefore never reaches [api_settings.<p>] on disk, although the table now says 'stored'.

### Acceptance criteria

- [ ] TASK-33001.14 #1 A provider section with credential_source = "none" makes the legacy summarizers send no API key, even when the environment variable named by api_key_env_var is set
- [ ] TASK-33001.14 #2 The summarizers resolve a credential the same way as resolve_provider_credential for every credential_source value, or the difference is recorded in ADR-012
- [ ] TASK-33001.14 #3 A test with the variable exported covers the "none" case for koboldcpp and tabbyapi and fails if the env fallback ignores credential_source
- [ ] TASK-33001.15 #1 A model-only Settings save of a provider whose key exists only under legacy [API] leaves readiness and the send path resolving the same credential, both before and after exporting the provider's env var
- [ ] TASK-33001.15 #2 A real-writer test on a scratch TLDW_CONFIG_PATH covers the [API]-only key case, with the env var unset and then set
- [ ] TASK-33001.15 #3 Keys typed by the user and untouched keys stored in [api_settings] behave exactly as after TASK-33001.13

### References

- tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py
- tldw_chatbook/Chat/provider_setup_persistence.py
- tldw_chatbook/config.py (only to tell where a stored value came from)
- tldw_chatbook/Chat/provider_readiness.py (read only)
- Tests/LLM_Calls/test_kobold_tabby_config.py
- Tests/UI/test_settings_configuration_hub.py (beside test_settings_model_save_keeps_the_stored_key_that_resolves)
- backlog/decisions/012-provider-credential-settings-boundary.md

## Task 10: Dimension tokens back at zero; the Lab GGUF row and backend names fit; a test host resolves the theme guards (TASK-33003.22, TASK-33641.2, TASK-33641.3, TASK-33641.7)

Task files:
- backlog/tasks/task-33003.22 - The-dimension-literal-and-Python-style-governance-ratchets-are-back-at-zero.md
- backlog/tasks/task-33641.2 - The-Lab-GGUF-Model-source-label-and-its-Select-share-one-row.md
- backlog/tasks/task-33641.3 - Image-and-Video-generation-backend-rows-show-each-backend-name-whole.md
- backlog/tasks/task-33641.7 - The-skills-recovery-controls-tests-resolve-the-theme-guard-variables-in-their-subprocess-host.md
Wave: 2
Depends on: —

### Why

- TASK-33003.22 (The dimension-literal and Python-style governance ratchets are back at zero): The gap is wider than filed.
  - test_dimension_literal_ratchet finds 18 raw literals: components/_agentic_terminal.tcss 1, components/_settings_splash_theme.tcss 2, features/_wizards.tcss 4 (new since filing) and features/_workflows.tcss 11.
  - test_python_style_ratchet flags Widgets/Library/library_skill_work_pane.py (a static width) and Widgets/Console/console_composer_bar.py (`set_styles(**{…: None})`, from 9a48cf3b06, TASK-33628.5.2).
  All of these are reds on dev too.
  test_sheet_size_ceiling also fails (features/_library_panels.tcss has 2004 active lines against 2000, from TASK-34000.25). No AC covers it: record it in the notes and do not fix it here.
  The offenders keep growing on dev, so re-measure after every rebase.
- TASK-33641.2 (The Lab GGUF "Model source:" label and its Select share one row): css/features/_llm-management.tcss gives `.gguf-source-mode-row`, `.gguf-source-label` (content-align right middle) and `.gguf-source-mode` each `height: $ds-size-3`. The borderless Select paints its value on its top row while the label sits on the middle row. The widgets are built in UI/LLM_Management_Window.py.
- TASK-33641.3 (Image and Video generation backend rows show each backend name whole): css/features/_settings.tcss still has `.settings-imagegen-backend-row .settings-input-label { width: $ds-size-12; min-width: $ds-size-12; }`. Its comment says a geometry guard pins 12 so that the row fits at 120 columns; that guard is the (235,52)/(120,45) parametrized pin in Tests/UI/test_settings_image_gen_panel.py (confirm by grep). The rows are built in Widgets/settings_image_gen_panel.py and Widgets/settings_video_gen_panel.py, not in the UI/Screens files the rider names.

- TASK-33641.7 (The skills recovery controls tests resolve the theme guard variables in their subprocess host): filed from the wave 1 integration. Twelve nodes of Tests/Backup_Recovery/test_skills_recovery_controls.py fail with "undefined variable '$tldw-boundary'" parsing screen_agentic_library.tcss, at wave 1's base and head alike: the test's subprocess host is a plain App, so the theme guard variables TASK-33003.6 introduced have no source. 0cfe341382 fixed the same class for Tests/UI/consolidated_css.py::ConsolidatedCSSApp by inheriting ThemeVariableDefaultsMixin; do the same here and grep Tests for other plain-App hosts that parse application CSS.

### Acceptance criteria

- [ ] TASK-33003.22 #1 test_dimension_literal_ratchet passes: each of the 14 literals becomes a token from css/core/_variables.tcss, with rendered geometry unchanged. [At HEAD the count is 18 literals in four sheets (see Why); the AC covers every offender at HEAD.]
- [ ] TASK-33003.22 #2 test_python_style_ratchet passes: library_skill_work_pane.py:170 uses a token-backed class or an allowed runtime write.
- [ ] TASK-33003.22 #3 The CSS bundle and split sheets are regenerated with build_css.py, and preflight passes.
- [ ] TASK-33641.2 #1 At 211x44 and 235x52, "Model source:" and the Select's current value are painted on the same screen row.
- [ ] TASK-33641.2 #2 The row's height comes from a design token (no new raw literal), and the governance ratchets do not rise.
- [ ] TASK-33641.2 #3 A pilot test under the production stylesheet fails before the fix and passes after it.
- [ ] TASK-33641.3 #1 At 211x44 and 235x52, every Image Generation and Video Generation backend row shows the backend's full name, with at least one blank cell between the name and its status.
- [ ] TASK-33641.3 #2 The status column still starts at the same column in every row of a panel.
- [ ] TASK-33641.3 #3 A pilot test over every backend label fails before the fix and passes after it.

- [ ] TASK-33641.7 #1 The twelve nodes pass, with the host resolving the theme guard variables the way the application does
- [ ] TASK-33641.7 #2 The cause and the fix are recorded in this task's notes
- [ ] TASK-33641.7 #3 A grep of Tests for other subprocess or scratch App hosts that parse application CSS without the theme defaults is recorded, and any that fail the same way are fixed here

### References

- tldw_chatbook/css/core/_variables.tcss
- tldw_chatbook/css/components/_agentic_terminal.tcss
- tldw_chatbook/css/components/_settings_splash_theme.tcss
- tldw_chatbook/css/features/_wizards.tcss
- tldw_chatbook/css/features/_workflows.tcss
- tldw_chatbook/css/features/_llm-management.tcss
- tldw_chatbook/css/features/_settings.tcss
- tldw_chatbook/Widgets/Library/library_skill_work_pane.py
- tldw_chatbook/Widgets/Console/console_composer_bar.py
- tldw_chatbook/Widgets/settings_image_gen_panel.py
- tldw_chatbook/Widgets/settings_video_gen_panel.py
- tldw_chatbook/UI/LLM_Management_Window.py (read only)
- tldw_chatbook/css/build_css.py (regenerate the bundle)
- Tests/UI/test_component_pattern_governance.py
- Tests/UI/test_llm_gguf_source_modes.py
- Tests/UI/test_settings_image_gen_panel.py
- Tests/UI/test_settings_video_gen_defaults.py
- qa/model-config-33003-captures/lab-211x44.txt
- qa/model-config-33003-captures/settings-video-generation-211x44.txt
- Tests/Backup_Recovery/test_skills_recovery_controls.py
- Tests/UI/consolidated_css.py (read only: the precedent)

## Task 11: Chat settings view tabs, a single readiness sync on mount, and the Context view's Apply reason (TASK-33003.11, TASK-33003.18, TASK-33006.21, TASK-33006.22, TASK-33006.23)

Task files:
- backlog/tasks/task-33003.11 - The-selected-Chat-settings-view-tab-keeps-its-whole-Selected-suffix-after-a-view-switch.md
- backlog/tasks/task-33003.18 - Keys-typed-right-after-clearing-a-Chat-settings-number-field-reach-the-field-in-a-live-terminal.md
- backlog/tasks/task-33006.21 - Chat-settings-syncs-default-readiness-once-on-mount.md
- backlog/tasks/task-33006.22 - The-Context-view-says-why-Apply-is-unavailable-while-an-endpoint-is-unsaved.md
- backlog/tasks/task-33006.23 - A-focused-view-tab-paints-its-Selected-mark-after-a-keyboard-round-trip.md
Wave: 2
Depends on: —

### Why

- TASK-33003.11 (The selected Chat settings view tab keeps its whole Selected suffix after a view switch): The view switch sets model_tab.label and context_tab.label on Buttons. Button.label is a plain reactive with no layout flag, and Button is width auto, so a tab keeps its old, shorter width and crops ' · Selected'. .console-settings-view-tab sets no width. The labels are now 'Model and generation' and 'Context and memory' (TASK-33006.5). The only pin compares str(label).
- TASK-33003.18 (Keys typed right after clearing a Chat settings number field reach the field in a live terminal): This is an investigation. In live tmux runs, typing 0.9 into Temperature right after eight backspaces dropped keys; Pilot typing never did. No lesson covers it. If the driver drops the keys, close without code: record the reproduction, and add a lesson mid-file in lessons-live-verification.md.
- TASK-33006.21 (Chat settings syncs default readiness once on mount): on_mount reaches _sync_default_readiness three times: through _show_settings_view, directly, and through _sync_completion_actions.
- TASK-33006.22 (The Context view says why Apply is unavailable while an endpoint is unsaved): _sync_completion_actions hides Apply while needs_endpoint_persistence is set, and fills #console-settings-primary-disabled-reason only when it is not. So the Context view gives no reason.
- TASK-33006.23 (A focused view tab paints its Selected mark after a keyboard round trip): Only label-string pins exist. Nothing checks the paint of a selected tab that holds focus reached with Shift+Tab. One paint probe can cover both this and 33003.11.

### Acceptance criteria

- [ ] TASK-33003.11 #1 At 211x44 and 235x52, after switching from Model to Context and back, each tab paints its whole label and the selected tab paints the whole ' · Selected' suffix.
- [ ] TASK-33003.11 #2 A painted-text pilot test drives the switches with real key presses or clicks under the production stylesheet.
- [ ] TASK-33003.18 #1 The task notes name where the keys are lost (the app's input handling or the test driver), with a reproduction.
- [ ] TASK-33003.18 #2 If the app drops them: at 211x44 in a live terminal, typing right after clearing a number field keeps every key, confirmed by repeated captures.
- [ ] TASK-33006.21 #1 Opening Chat settings builds default readiness once
- [ ] TASK-33006.21 #2 The footer, the defaults line and the readiness block read as before (existing tests pass)
- [ ] TASK-33006.22 #1 With an unsaved endpoint, the Context view names the reason and how to resolve it (save the endpoint in Model and generation)
- [ ] TASK-33006.22 #2 A mounted test pins the line
- [ ] TASK-33006.23 #1 A selected view tab that holds focus paints '· Selected', or the task records that no path paints it without the mark
- [ ] TASK-33006.23 #2 A paint probe under the production stylesheet pins it

### References

- tldw_chatbook/Widgets/Console/console_settings_modal.py (this wave's modal slot; net lines <= 0)
- tldw_chatbook/css/features/_console_panels.tcss
- tldw_chatbook/css/build_css.py (regenerate the bundle)
- Tests/UI/test_console_session_settings.py
- Tests/UI/test_console_settings_geometry.py
- backlog/docs/lessons-live-verification.md
- qa/model-config-p3-2026-09-28/task-2/tabs-after.txt

## Task 12: More readers of shared connection evidence, and the Phase 5 evidence paths hardened (TASK-33005.9, TASK-33005.10)

Task files:
- backlog/tasks/task-33005.9 - Home-Library-and-the-first-run-wizard-read-shared-connection-evidence.md
- backlog/tasks/task-33005.10 - Measure-and-harden-the-Phase-5-evidence-paths.md
Wave: 5
Depends on: —
Must not edit: Tests/UI/test_console_session_settings.py (Task 11 owns it in this wave). Put the keyed Settings → Chat settings case in a new test file.

### Why

- TASK-33005.9 (Home, Library and the first-run wizard read shared connection evidence): Three readers still ignore the shared evidence:
  - home_screen.py calls build_console_settings_readiness(...) with no connection_evidence.
  - library_screen.py's lock pre-check does the same. library_screen.py is very large and lazily parsed, so touch only that pre-check's readiness call.
  - The wizard's provider step constructs a private ProviderTestEvidenceStore(). It is in UI/Wizards/first_run_provider_step.py, not FirstRunSetupWizard.py.
- TASK-33005.10 (Measure and harden the Phase 5 evidence paths): This rider bundles four items:
  - provider_test_evidence.py advances `_BEGIN_ORDER = count(1)` with next() and no lock.
  - The warm switch path's readiness cost with evidence present is unmeasured.
  - The two named tests still exist and still cannot fail on what they claim: test_closing_mid_probe_raises_nothing_and_a_late_result_loses (Tests/UI/test_console_switcher_local_probe.py) and test_model_summary_sync_invalidates_mounted_context_allocation (Tests/UI/test_console_resize_reflow.py).
  - AC#3's premise is partly stale. Save already passes setup_mutation.semantic_identity to rebase_after_save (95857a454f), and rebuilds with replace(tested_identity, …) only when that is None and no endpoint is saved. Confirm which rebuild the review meant before changing it.

### Acceptance criteria

- [ ] TASK-33005.9 #1 Home and the Library lock pre-check read the shared connection evidence for the active connection, with no network request on read
- [ ] TASK-33005.9 #2 The first-run wizard publishes its settled test results to the shared owner instead of keeping a private copy
- [ ] TASK-33005.9 #3 A test shows a refused result reaching each surface
- [ ] TASK-33005.10 #1 The warm model-switch path's readiness cost with evidence present is measured against without, and memoized if it rises
- [ ] TASK-33005.10 #2 Begin-order allocation is safe without the GIL (a lock or an atomic), with a concurrent test
- [ ] TASK-33005.10 #3 Save persists the tested semantic identity instead of rebuilding it, and Settings reuses the snapshot model-facet logic [Premise partly stale; see Why before changing the save path.]
- [ ] TASK-33005.10 #4 The three test gaps each have a test that fails when the behaviour breaks (negative control recorded)

### References

- tldw_chatbook/UI/Screens/home_screen.py
- tldw_chatbook/UI/Screens/library_screen.py (the lock pre-check only)
- tldw_chatbook/UI/Wizards/first_run_provider_step.py
- tldw_chatbook/Chat/provider_test_evidence.py
- tldw_chatbook/Chat/console_session_settings.py
- tldw_chatbook/UI/Screens/settings_screen.py (this wave's Settings slot; zero headroom)
- Tests/UI/test_console_switcher_local_probe.py
- Tests/UI/test_console_resize_reflow.py

## Task 13: Quitting during a hung reply exits, and a second open of a chat settles its recovery (TASK-33666, TASK-33667)

Task files:
- backlog/tasks/task-33666 - Quitting-during-a-hung-reply-never-exits-the-process.md
- backlog/tasks/task-33667 - Settle-a-dispatch-recovery-from-a-second-open-of-the-same-chat.md
Wave: 2
Depends on: —

### Why

- TASK-33666 (Quitting during a hung reply never exits the process): Nothing has fixed it since it was filed. October's quit work does not touch a stalled stream: the TASK-33622.10 prompt fixes, TASK-33628.5's exit wait for Delete/Undo saves, and the quit-time summary (#3046, #3056). First name the thread or task that keeps the process alive; a likely candidate is a non-daemon worker blocked in a provider read with no timeout. Check that TASK-33628.5's exit wait is not what hangs. The reproduction is in qa/task-33662-recovery-relaunch/.
- TASK-33667 (Settle a dispatch recovery from a second open of the same chat): The function the rider names is now UI/Console_Modules/workspace.py::_resume_console_workspace_conversation. It still defaults to `reuse_existing: bool = False`, and only session.py passes True. Chat/console_chat_store.py keeps only a hydration-local persisted_to_native map. You may either reuse the open session (the smaller change, since that path exists) or map owners per session; record the reason in the notes.

### Acceptance criteria

- [ ] TASK-33666 #1 With a reply hung (a provider that accepts the request and never streams), Ctrl+Q → Quit exits the process within a bounded time (state the bound) and the reply is settled as stopped.
- [ ] TASK-33666 #2 The cause, meaning which thread or task keeps the process alive, is named in the notes with evidence.
- [ ] TASK-33666 #3 A test reproduces the hang and passes after the fix.
- [ ] TASK-33667 #1 With the same chat open twice after a relaunch, Retry and Discard settle the recovery from either open, without changing the other session's rows, OR opening an already-open conversation reuses its session so a second open never exists. The task notes say which approach was taken and why.
- [ ] TASK-33667 #2 A regression test drives the second-open case through the production hydration path.

### References

- tldw_chatbook/app_lifecycle.py
- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/LLM_Calls/ (the provider streaming call, only if the cause is there)
- tldw_chatbook/UI/Console_Modules/workspace.py
- tldw_chatbook/Chat/console_chat_store.py
- tldw_chatbook/UI/Screens/chat_screen.py (this wave's Console slot; only if the cause lives there)
- qa/task-33662-recovery-relaunch/README.md

## Task 14: Every model-config test file runs its assertions in a PR lane (TASK-33001.11, TASK-33002.17, TASK-33641.5, TASK-33641.8)

Task files:
- backlog/tasks/task-33001.11 - ProductionApp-provider-selection-tests-never-run-their-assertions-in-CI.md
- backlog/tasks/task-33002.17 - Phase-2-Settings-tests-join-the-UI-PR-gate-census.md
- backlog/tasks/task-33641.5 - A-PR-lane-runs-the-model-default-writer-tests-in-test_console_settings_defaults.md
- backlog/tasks/task-33641.8 - A-pytest-run-from-the-repo-root-does-not-import-the-test-named-files-under-Docs.md
Wave: 5
Depends on: Task 4 (raw-scope change) and Task 7 (it edits Tests/UI/test_settings_state_line_unsaved_count.py), both in earlier waves.

### Why

- TASK-33001.11 (ProductionApp provider-selection tests never run their assertions in CI): Tests/ProductionApp/test_provider_selection_ownership.py has no pytestmark and imports tldw_chatbook.app at module scope. Run alone, it still fails at collection with RecoveryRequired raw_source_selection_changed ('1 error in 3.17s'). Its three tests are now test_real_app_restart_routes_saved_global_and_model_profile_to_new_chat, test_real_console_change_model_command_opens_real_picker and test_settings_save_preserves_user_session. AC#2's isolation question is yours to settle: the bootstrap profile is shared, and these tests persist chat_defaults and api_settings to it.
- TASK-33002.17 (Phase 2 Settings tests join the UI PR gate census): None of the three Phase 2 files is in scripts/ui_pr_gate_census.txt. The lane now runs as six round-robin shards (`check_ui_pr_gate_census.py --shard`, timeout-minutes 20), and its comment says to add a shard, not minutes, when one nears the cap. Verify each file green in that job's minimal-deps environment and order before adding it.
- TASK-33641.5 (A PR lane runs the model-default writer tests in test_console_settings_defaults): `pytest Tests/Chat/test_console_settings_defaults.py -n 6 --dist loadfile` gave '1 failed, 86 passed'.
  - test_externally_reserved_newer_generation_invalidates_inflight_precondition is red; the writer logs RecoveryRequired (phase=read).
  - test_newer_generation_cannot_reserve_between_precondition_and_replacement passed in that run; check whether it is flaky.
  No workflow names the file. The PR file lists live in .github/workflows/derived-artifacts.yml, which has a sandboxed invocation and an 'admission-sensitive suites' invocation. Adding the file to an existing lane is implementer scope: it is not a new lane, a merge train or a PAT. Re-run the file after Task 4's raw-scope change.

- TASK-33641.8 (A pytest run from the repo root does not import the test-named files under Docs): filed by the controller on 2026-10-11 from Task 10's incident (ledger R13). pyproject.toml's pytest addopts has `--ignore=STests` and nothing that excludes Docs, so an empty argument or `pytest .` imports three test-named files under Docs/ outside every conftest: one calls a paid provider with the configured key, one rewrites a tracked test file. Add `--ignore=Docs` beside `--ignore=STests`; an explicit path under Docs/ must keep working.
- R11 also gives this task Task 10's three unlisted gate files (Tests/UI/test_settings_image_gen_panel.py, test_llm_gguf_source_modes.py, test_library_skills_builtin_rows.py) and the question of which job runs Tests/Backup_Recovery/test_skills_recovery_controls.py.

- From Task 9's review (ledger R18): Tests/UI/test_settings_configuration_hub.py has no row in the UI census, so TASK-33001.15's only real-writer test runs in no PR lane. Add the node test_settings_model_save_writes_a_legacy_api_section_key_into_the_provider_table (it carries @private_profile_test) under R11's terms.

### Acceptance criteria

- [ ] TASK-33001.11 #1 Every test in Tests/ProductionApp/test_provider_selection_ownership.py runs its assertions in the Core Tests job's collection order and passes there, and one negative control makes a test fail on its own assertion rather than in setup
- [ ] TASK-33001.11 #2 The isolation chosen keeps each test's profile private, or proves that no config value these tests persist is visible to a later test in the same worker
- [ ] TASK-33002.17 #1 Tests/UI/test_model_config_field_labels.py, Tests/UI/test_settings_provider_save_scope.py and Tests/UI/test_settings_state_line_unsaved_count.py are in the census, verified green in the UI fast lane's own environment and order
- [ ] TASK-33002.17 #2 The UI fast lane stays inside its time budget
- [ ] TASK-33641.5 #1 The two generation tests pass locally without bypassing the ADR-126 admission (no `bootstrap_profile` opt-out for writer tests).
- [ ] TASK-33641.5 #2 A required pull-request lane collects `Tests/Chat/test_console_settings_defaults.py` and stays within its job time cap (measure the added time).
- [ ] TASK-33641.5 #3 A deliberately broken quick-mask write (for example, writing the fallback Max tokens instead of deleting it) fails that lane.

- [ ] TASK-33641.8 #1 A pytest run started at the repository root with an empty argument or '.' imports no file under Docs/, shown by a test or a recorded measurement on the real tree
- [ ] TASK-33641.8 #2 An explicit path to a file under Docs/ still runs
- [ ] TASK-33641.8 #3 The lesson in backlog/docs/lessons-testing-evidence.md names the guard

### References

- Tests/ProductionApp/test_provider_selection_ownership.py
- Tests/ProductionApp/conftest.py
- Tests/conftest.py
- Tests/private_profile.py (read only)
- scripts/ui_pr_gate_census.txt
- scripts/check_ui_pr_gate_census.py (read only)
- .github/workflows/derived-artifacts.yml
- Tests/Chat/test_console_settings_defaults.py
- tldw_chatbook/Chat/console_settings_defaults.py (only if the red is a product defect)
- pyproject.toml (pytest addopts only)

## Task 15: Chat settings offers only what the request sends and counts only the user's edits (TASK-33002.7, TASK-33003.13, TASK-33006.11, TASK-33006.12, TASK-33006.16)

Task files:
- backlog/tasks/task-33002.7 - Conversation-settings-stops-offering-levels-the-local-request-drops.md
- backlog/tasks/task-33003.13 - Custom-endpoints-show-the-provider-choice-rows-their-registry-family-supports.md
- backlog/tasks/task-33006.11 - Providers-without-a-PROVIDER-PARAM-MAP-entry-get-a-support-answer-for-their-samplers.md
- backlog/tasks/task-33006.12 - A-field-whose-support-is-unknown-keeps-its-own-help-visible-beside-the-neutral-note.md
- backlog/tasks/task-33006.16 - A-pick-does-not-list-fields-the-user-never-touched-in-the-unsaved-prompt.md
Wave: 6
Depends on: —

### Why

- TASK-33002.7 (Conversation settings stops offering levels the local request drops): console_settings_modal.py::_generation_choice_select builds its options from the static _PROVIDER_CHOICE_VALUES; only settings_screen.py calls console_provider_support.py::reasoning_effort_values_sent. The modal has zero headroom, so put the logic in console_provider_support.py or console_settings_field_row.py. _generation_choice_validation_copy ('Saved value is unavailable') is the existing hook for AC#2.
- TASK-33003.13 (Custom endpoints show the provider-choice rows their registry family supports): OWNER RULING 2026-10-09: the family rule that 5ba142ac79 shipped stands (console_provider_support.py::_registry_family_provider decides a custom-ep id by its family, which matches what the request sends). Nothing changes in the rule. Record the ruling in the rider's task file, and pin a second family: Tests/UI/test_console_settings_hidden_fields.py::test_a_registry_endpoint_hides_what_its_family_hides pins llama_cpp only today.
- TASK-33006.11 (Providers without a PROVIDER_PARAM_MAP entry get a support answer for their samplers): local_onnx and local_transformers are in neither PROVIDER_PARAM_MAP nor API_CALL_HANDLERS (62 keys each, the same set, measured at b2d1e8e20d). They have no chat handler at all, as UI/Evals/evals_state.py notes, so there is no request map to derive their fields from. The neutral-copy branch of AC#1 is the honest choice.
- TASK-33006.12 (A field whose support is unknown keeps its own help visible beside the neutral note): console_settings_field_row.py::_field_help returns f"{GENERATION_CONTROL_UNKNOWN_COPY} {text}" on a help line that does not wrap and ends in an ellipsis, so the field's own help is cut off. Prefer a shorter marker to moving it into the Source column, which would change CSS. Tests/UI/test_console_settings_hidden_fields.py and Tests/UI/test_console_session_settings.py import the constant.
- TASK-33006.16 (A pick does not list fields the user never touched in the unsaved prompt): console_settings_unsaved.py::unsaved_labels compares every value with the opening baseline. A pin conflicts with this rider: test_credential_round_trip_keeps_the_restored_edit_unsaved[model] (Tests/UI/test_console_settings_unsaved_guard.py) asserts {'Model', 'Streaming'} <= labels, under a comment from Qodo #2937 that a pick's re-base of Streaming counts as an edit. This rider was filed later and rules that it does not.
  - Rewrite that pin on purpose and name it in the PR.
  - Keep TASK-33003.10's guarantee: a change the user makes to Streaming's Inherit/On/Off is still named.
  - The same node is the test of closed rider TASK-33003.21, so it must still pass 10 of 10 serial runs after the rewrite.

### Acceptance criteria

- [ ] TASK-33002.7 #1 The modal's Reasoning effort choices for a strict local template offer only levels the request sends, decided by the same real request builder Settings uses
- [ ] TASK-33002.7 #2 A value saved earlier stays visible and labelled as not supported rather than silently cleared
- [ ] TASK-33002.7 #3 console_settings_modal.py does not grow past its module-size ratchet
- [ ] TASK-33003.13 #1 The owner's decision on which choice rows a custom endpoint shows is recorded in this task.
- [ ] TASK-33003.13 #2 Chat settings shows a custom endpoint's provider-choice rows according to that decision, pinned by a test over at least two registry families.
- [ ] TASK-33006.11 #1 local_onnx and local_transformers either have PROVIDER_PARAM_MAP entries or their sampler rows carry the neutral support copy
- [ ] TASK-33006.11 #2 A test pins the chosen behaviour for both providers
- [ ] TASK-33006.12 #1 At 211x44 an unknown-support row shows both that support is not verified and the start of the field's own help
- [ ] TASK-33006.12 #2 The neutral wording stays the one constant the tests import
- [ ] TASK-33006.16 #1 After a pick with no other edit, the unsaved prompt and the edit count name only Model
- [ ] TASK-33006.16 #2 Fields the rebase blanked because the new provider rejects them are not counted as user edits

### References

- tldw_chatbook/Widgets/Console/console_settings_modal.py (this wave's modal slot; net lines <= 0)
- tldw_chatbook/Chat/console_provider_support.py
- tldw_chatbook/Widgets/Console/console_settings_field_row.py
- tldw_chatbook/Widgets/Console/console_settings_unsaved.py
- tldw_chatbook/Chat/Chat_Functions.py (read only)
- Tests/UI/test_console_settings_unsaved_guard.py
- Tests/UI/test_console_settings_hidden_fields.py
- Tests/UI/test_console_session_settings.py
- qa/model-config-p6-t4-2026-10-02 (capture 07)

## Task 16: Switch model shows what the server lists, full provider names and typed sources, and a picker click lands where pointed (TASK-33003.12, TASK-33005.12, TASK-33005.14, TASK-33006.18, TASK-33641.9)

Task files:
- backlog/tasks/task-33003.12 - A-click-while-a-pickers-results-list-collapses-lands-on-the-control-under-the-pointer.md
- backlog/tasks/task-33005.12 - Show-the-models-a-local-probe-listed-in-Switch-model.md
- backlog/tasks/task-33005.14 - Polish-three-small-model-surface-findings-from-the-Phase-5-captures.md
- backlog/tasks/task-33006.18 - Type-the-switcher-sources-the-Console-hands-pick-mode.md
- backlog/tasks/task-33641.9 - Switch-model-lists-a-provider-s-own-rows-first-when-Find-is-exactly-that-provider-s-name.md
Wave: 6
Depends on: —
Must not edit: providers_models_card.py (read only). Fix the picker in model_search_picker.py.

### Why

- TASK-33003.12 (A click while a picker's results list collapses lands on the control under the pointer): Chat settings has had no picker since TASK-33006.4 (5b187e2c5c). The collapse-on-blur shape remains in Widgets/model_search_picker.py::ModelSearchPicker: on_descendant_blur → _restore_committed_display_after_blur → _hide_results, after a 0.05 s timer. Settings ▸ Providers & Models (DefaultModelPicker, ProviderFilterInput) and the Switch model search both use it. Reproduce first with a pilot click, without Esc, on a Settings picker; if it does not reproduce, close the rider with that test as evidence. AC#2 is already ticked (moot since TASK-33006.2).
- TASK-33005.12 (Show the models a local probe listed in Switch model): console_model_popover.py::_saved_models builds rows from _providers_models (or a registry entry) plus self._served. _offer_probes hands the prober refresh_readiness, which re-reads readiness only, so the evidence's model_ids never reach the rows. Phase 8 Task 3 added served_models and highlight_pair for a detected server in model_switcher.py: reuse that path.
- TASK-33005.14 (Polish three small model-surface findings from the Phase 5 captures): Only AC#2 remains: `_PROVIDER_COLUMNS = 20` is a fixed width in the pair-row format. AC#1 and AC#3 close without code, as their notes say.
- TASK-33006.18 (Type the switcher sources the Console hands pick mode): UI/Console_Modules/model_switcher.py::_switcher_sources returns dict[str, object], and both callers use `**_switcher_sources(  # type: ignore[arg-type]`. Phase 8 Task 3 added parameters to this function.

- TASK-33641.9 (Switch model lists a provider's own rows first when Find is exactly that provider's name): filed by the controller on 2026-10-11 from Phase 8's Task 2 review. Find ranks by text match alone, so an aggregator whose ids start with a provider's name can outrank that provider's own rows. Phase 8 fixed only the reopen-after-return highlight (`_highlight_target`); this is the typed-query ranking in console_model_popover.py.

### Acceptance criteria

- [ ] TASK-33003.12 #1 Clicking a control while an open picker's results list collapses activates the control that was under the pointer when the button went down, or none; it never toggles or focuses a different control. A pilot test clicks without pressing Escape first.
- [x] TASK-33003.12 #2 test_css_class_coverage_contract passes for console-settings-control-support: the class is styled, or exempted with a stated reason. [Moot since TASK-33006.2: that task deleted the per-row support statics and the class with them, and no source names the class any more (only a test asserting it is absent), so nothing is left to style or exempt.]
- [ ] TASK-33005.12 #1 After a probe lists a local server's models, that provider's switcher rows show the listed models, and models the server did not list are no longer offered as ready rows
- [ ] TASK-33005.12 #2 With no listing yet (not probed, refused, or timed out), the rows keep today's saved and cached models
- [ ] TASK-33005.12 #3 No new network request is made for this: the rows read the evidence the probe already recorded
- [ ] TASK-33005.12 #4 A test drives the real popover with a stubbed server listing one model that is not in the saved list
- [ ] TASK-33005.14 #1 The Test Provider control reads as a button (focusable, styled like the other Settings action buttons), confirmed at 211x44 [Closes without code: superseded by TASK-33007.2, which made the control a compact 'Test (t)' button that is deliberately not a Tab stop. Record this in the notes.]
- [ ] TASK-33005.14 #2 Switch model shows full provider display names when the box has room, and shortens them only when it does not, at 211x44 and 235x52
- [ ] TASK-33005.14 #3 Switch model and the Console status strip state the same context size (or the same 'unknown') for one model [Closes without code: fixed by 35a3cb339b (a fallback context window reads '?', matching 'Context unknown'). Record this in the notes.]
- [ ] TASK-33005.14 #4 Each change has a test that fails without it
- [ ] TASK-33006.18 #1 The sources have a precise type (a TypedDict or dataclass) and no caller needs a type-ignore
- [ ] TASK-33006.18 #2 Behaviour is unchanged (the pick-mode tests pass)

- [ ] TASK-33641.9 #1 When Find's text is exactly a provider's display name or id, that provider's rows are listed before other providers' rows that merely match the text
- [ ] TASK-33641.9 #2 A query that is not exactly a provider's name ranks as today, pinned by the existing Find tests or by tests rewritten on purpose and named
- [ ] TASK-33641.9 #3 A test drives the two measured cases: a provider with an empty list beside a ready aggregator whose ids start with its name, saved and unsaved

### References

- tldw_chatbook/Widgets/Console/console_model_popover.py
- tldw_chatbook/UI/Console_Modules/connection_probe.py
- tldw_chatbook/UI/Console_Modules/model_switcher.py
- tldw_chatbook/Widgets/model_search_picker.py
- tldw_chatbook/UI/Settings_Modules/providers_models_card.py (read only)
- Tests/UI/test_console_switcher_local_probe.py
- Tests/UI/test_console_model_switcher.py
- Tests/Widgets/test_model_search_picker.py
- Tests/UI/test_settings_default_model_picker.py
- qa/model-config-33005-captures/README.md

## Task 17: Legacy summarizers get their arguments and their credentials right (TASK-33641.10, TASK-33641.11, TASK-33001.14)

Task files:
- backlog/tasks/task-33641.10 - The-summarization-dispatcher-passes-KoboldCpp-and-Ollama-their-arguments-by-name.md
- backlog/tasks/task-33641.11 - A-legacy-API-key-is-not-used-once-the-provider-s-credential-source-says-none-or-environment.md
- backlog/tasks/task-33001.14 - Legacy-summarizers-send-an-env-key-for-a-provider-whose-credential-source-is-none.md
Wave: 5
Depends on: Task 9 (integrated; its TASK-33001.14 half was lost in the 2026-10-16 restack and moves here)
Must not edit: tldw_chatbook/UI/Screens/settings_screen.py (Task 12 owns it in this wave).

### Why

- TASK-33641.10 (The summarization dispatcher passes KoboldCpp and Ollama their arguments by name): filed by the controller on 2026-10-11 from Task 9's review. Summarization_General_Lib.py::_dispatch_to_api calls each summarizer positionally; listed with ast, KoboldCpp gets system_message<=temp, temp<=system_message and kobold_api_ip<=streaming, and Ollama gets model<=streaming; the other 17 calls pair name to name. summarize_chunk (FIXME, no caller) passes its text as api_name. Read from the code by Task 9, not run: prove each binding through analyze() before fixing.
- TASK-33641.11 (A legacy [API] key is not used once the provider's credential source says none or environment): filed the same way. config.py::_normalize_legacy_provider_api_key never reads credential_source (0 matches for the word in config.py), so after a Clear in Settings (credential_source none) or with an unset variable (environment), a key still under [API] is copied into the provider's table and its legacy dict: readiness resolves no key while summarize_with_openai and the like send it. Task 9 made the summarizers resolve a key as readiness does (TASK-33001.14); this is the loader's half. ADR-012: a credential the user removed is not used.

- TASK-33001.14 (Legacy summarizers send an env key for a provider whose credential source is none): moved here by the controller on 2026-10-17. The 2026-10-16 restack dropped Task 9's fix for it (the full old fix is on backup/p9-before-restack-1016, 841361671c, for reference only), and dev's task-34895 (a79b07713e, PR #3105) has since moved KoboldCpp and TabbyAPI summaries onto `_summarize_via_chat_api` -> `chat_api_call`, so the old fix does not apply as written. Redo it on the chat-dispatch route; the rider's REOPENED notes hold the measured table (a legacy [API] key is still sent under "none"; TabbyAPI "stored", "environment" and no-source sections differ from readiness). TASK-33641.11's "Task 9 made the summarizers resolve a key as readiness does" is true again only once this rider is done, so do it first. Decide what happens to the orphaned LLM_Calls/Local_Summarization_Lib.py (no production importer; task-34895's notes say it was deleted, but it is in the tree) and record the choice in the rider's notes.

### Acceptance criteria

- [ ] TASK-33641.10 #1 KoboldCpp and Ollama summarization receive each argument under the parameter it is meant for, proved through analyze() with a test per provider that fails first
- [ ] TASK-33641.10 #2 Every call in _dispatch_to_api passes its arguments by keyword, and a test fails if a call binds an argument to a parameter of another name
- [ ] TASK-33641.10 #3 summarize_chunk either passes its text as text or is removed with its FIXME, with the choice recorded in this task's notes
- [ ] TASK-33641.11 #1 With credential_source none or environment recorded for a provider, the loader does not copy a legacy [API] key into the provider's table or its legacy dict, proved by a real-loader test on a scratch config for at least two of the nine providers
- [ ] TASK-33641.11 #2 A provider with no credential_source recorded keeps today's legacy fallback, pinned by the existing normalisation tests
- [ ] TASK-33641.11 #3 Readiness and the legacy readers agree for every combination of credential_source and key location the tests cover, and the combinations are listed in this task's notes
- [ ] TASK-33001.14 #1 A provider section with credential_source = "none" makes the legacy summarizers send no API key, even when the environment variable named by api_key_env_var is set
- [ ] TASK-33001.14 #2 The summarizers resolve a credential the same way as resolve_provider_credential for every credential_source value, or the difference is recorded in ADR-012
- [ ] TASK-33001.14 #3 A test with the variable exported covers the "none" case for koboldcpp and tabbyapi and fails if the env fallback ignores credential_source

### References

- tldw_chatbook/LLM_Calls/Summarization_General_Lib.py
- tldw_chatbook/config.py (_normalize_legacy_provider_api_key only)
- tldw_chatbook/Chat/provider_readiness.py (read only)
- Tests/LLM_Calls/ (the summarizer tests Task 9 touched)
- backlog/decisions/012-provider-credential-settings-boundary.md

## Task 18: A same-provider pick keeps the chat's session endpoint (TASK-35190)

Task files:
- backlog/tasks/task-35190 - A-same-provider-pick-of-an-unseen-model-keeps-the-chat-s-session-endpoint.md
Wave: 7
Depends on: Task 6 (TASK-33006.19, integrated in Wave 4)

### Why

- TASK-35190: filed from the TASK-33006.19 review (Wave 4, 2026-10-17). A chat whose session Endpoint differs from the configured URL (a second local server applied to one chat) is moved back to the configured URL when the user picks a same-provider model the draft has not seen, through the controller's settings rebaser, which Switch model shares. Since TASK-33006.19, Test connection at the session Endpoint can report models that Change then does not offer. The rider's description has the trace.

### Acceptance criteria

- [ ] TASK-35190 #1 In Chat settings, picking a saved model on the chat's own provider keeps the chat's session Endpoint (:9200) instead of resetting it to the configured URL (:9099), pinned by a test
- [ ] TASK-35190 #2 A model that Test connection listed at the chat's session Endpoint (:9200) is offered in pick mode without retyping the Endpoint, and picking it keeps the chat on :9200, pinned by a test
- [ ] TASK-35190 #3 A pick onto a different provider still takes that provider's own endpoint, so no session URL is carried across providers
- [ ] TASK-35190 #4 Switch model (Alt+M) picking an unseen model on the chat's own provider also keeps the session Endpoint (:9200), so it and Chat settings land the same pick on the same Endpoint, and the quick surface still saves no Endpoint as a default, pinned by a test

### References

- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/Chat/console_session_settings.py
- tldw_chatbook/Widgets/Console/console_model_popover.py
- tldw_chatbook/Widgets/Console/console_settings_field_row.py
- tldw_chatbook/Widgets/Console/console_settings_modal.py (net lines <= 0)
- tldw_chatbook/UI/Console_Modules/model_switcher.py (added by the controller 2026-10-18 after review round 1: the chat's own provider's readiness and probe resolve over the session endpoint, as `_open_switcher` already does for a detected server)
- tldw_chatbook/Chat/console_settings_apply.py (added 2026-10-18: `rebased_endpoint` lives beside `remember_model_draft`, because console_chat_controller.py had 2 lines of headroom)
- Tests/UI/test_console_endpoint_discovery.py

## Execution order

[[1, 3, 4, 5], [9, 10, 11, 13], [2], [6, 7, 8], [12, 14, 17], [15, 16], [18]]

- Wave 1: Tasks 1, 3, 4, 5 (done 2026-10-10). After the restack, TASK-33006.17 (a) is fixed by the base (dev's 8c033d006b). TASK-33003.19's inventory was brought current again on four modals dev added.
- Wave 2: Tasks 9, 10, 11, 13 (done and integrated 2026-10-11). After the restack, TASK-33001.14 is reopened: the restack dropped its fix, and the base does not fix it (see Task 9). It moves to Task 17 (Wave 5).
- Rebase this branch onto Phase 8's final head.
- The rebase onto Phase 8's final head is done (Phase 8 merged as #3144 at 111f8e48fc, 2026-10-17); Wave 3 starts.
- Wave 3: Task 2 alone (settings_screen.py, chat_screen.py and console_model_popover.py together) (done 2026-10-17, reviewed clean at fc9f3ec375).
- Wave 4: Tasks 6, 7, 8 (done 2026-10-17, integrated at 0499159640). The three tasks add 10 UI PR census entries (332 -> 342). Filed during its reviews: TASK-35190 (from TASK-33006.19; planned as Task 18, Wave 7) and TASK-35020.1 (a red speech-form source pin under TASK-35020, outside this phase).
- Wave 5: Tasks 12, 14, 17 (done 2026-10-17, integrated at 414b0880c1). The three tasks add 9 UI PR census entries (342 -> 351: Task 12's four, Task 14's five; Task 14 also widens the provider save-scope node to its whole file in place). Filed during its reviews: TASK-35339 (the shared scratch_config fixture, from TASK-33002.17), TASK-35227 (the Realtime, TTS, MCP and RAG key readers, from TASK-33641.11) and TASK-35369 (Moonshot and Z.ai readiness under an environment source, from TASK-33001.14). After integration, 8b9cb5daec (TASK-33001.8) fixed the census node test_settings_connect_tab_budget.py::test_restored_openai_connection_row_shows_only_while_a_review_awaits, red since Wave 3's a8a3b3f68f (found by Task 14).
- Wave 6: Tasks 15, 16 (done 2026-10-18, integrated at b478549b2d). The two tasks add 2 UI PR census entries (351 -> 353: Task 15's credential round-trip node, Task 16's local-probe listing node); the integration review's round 1 adds one more (353 -> 354: pick mode offers a Chat settings listing beside the configured server's own listing, TASK-33005.12). Pins rewritten on purpose, which the PR must call out: Task 15's Tests/UI/test_console_settings_unsaved_guard.py::test_credential_round_trip_keeps_the_restored_edit_unsaved[model], Tests/UI/test_console_session_settings.py::test_console_settings_modal_enumerated_inputs_list_accepted_values and Tests/UI/test_console_settings_hidden_fields.py::test_unknown_support_stays_visible_with_neutral_copy_in_its_help_line (its last assertion, TASK-33006.11's notes); Task 16's Find-ranking tests in Tests/UI/test_console_model_switcher.py, named in TASK-33641.9's notes (test_a_highlight_pair_past_the_match_cap_stays_listed, test_a_highlight_pair_keeps_its_needs_setup_row and test_a_list_reopened_from_settings_never_highlights_another_providers_row, eight cases in all), and Tests/UI/test_console_model_switcher.py::test_custom_endpoints_show_display_names_and_builtin_slots_stay (TASK-33005.14's notes). Filed during its reviews: TASK-35460 (a Chat settings reopen after compaction depends on the estimate cache's clock; found in integration review round 3, pre-existing). The integration review's rounds 1-2 fixed a cross-wave regression in pick mode (Task 16's server listing hid TASK-33006.19's listed models; f5419adef5, a39809393b; census 353 -> 355).
- Wave 7: Task 18 alone (added by the controller 2026-10-17; it shares files with Task 12 in Wave 5 and with Tasks 15 and 16 in Wave 6). Done 2026-10-18 at db32d911d7 (clean in review round 4). It adds 8 UI PR census entries (355 -> 363). Pins rewritten on purpose, which the PR must call out: Tests/UI/test_console_endpoint_discovery.py::test_a_listing_is_offered_only_while_its_pick_keeps_the_listed_endpoint, and two renamed with their census lines (test_a_listing_is_offered_beside_the_configured_servers_own_listing -> test_a_listing_at_this_chats_endpoint_replaces_the_configured_servers, test_a_listing_is_offered_beside_a_loaded_catalog -> test_a_listing_at_this_chats_endpoint_replaces_a_loaded_catalog), named in TASK-35190's notes.

## Closed without code

These riders get no plan task. Close each one in its own task file, with the evidence below as the closing note, so the owner can review it (parent AC#1 and AC#2). None is OBSOLETE.

- TASK-33001.12 (Six mounted roleplay-resume navigation tests are red before TASK-33001), ALREADY FIXED: 3146bbd8da (#3011, on dev) wraps the file's mounted tests in private_profile_test (12 uses) and settles the explicit-resume contract as a product fix: chat_screen.py::_reconcile_console_after_attach skips the attach-time sync while _resume_navigation_startup_in_progress ('An explicit Resume owns its final presentation'). A serial run of Tests/UI/test_console_roleplay_resume_navigation.py at HEAD: '19 passed in 59.56s'. Caveat: under -n 6 on a loaded machine, test_mounted_resume_never_focuses_setup_modal_before_final_opener failed once and then passed 5 of 5 serial reruns. It is not one of the six reds this rider names.
- TASK-33002.8 (Provider Test drops a probe result that belongs to an earlier draft), ALREADY FIXED: b7b28aec45 (Qodo #2878, on dev) gives every probe a Test draft generation. _mark_provider_test_result_stale bumps it on any provider or endpoint edit, and _apply_provider_endpoint_probe_outcome returns early through _provider_probe_generation_current. Tests/UI/test_settings_provider_test_draft.py::test_identity_less_probe_finishing_after_an_edit_leaves_the_new_draft_stale changes the endpoint mid-probe and asserts stale rows and no toast. The mechanism is a generation counter rather than a captured provider and endpoint, but the outcome AC#1 asks for holds, and AC#2's test exists.
- TASK-33002.15 (Provider-switch generation-test sync test stops depending on event order), ALREADY FIXED: 5b187e2c5c (TASK-33006.4, on dev) rewrote the test to drive _model_picked → console_settings_modal.py::_apply_rebased_state, which syncs generation-control support and then the test controls synchronously, in a fixed order. The node passed 20 of 20 isolated runs at HEAD.
- TASK-33003.21 (The credential round-trip Model test sees only the user's edit after the reopened modal settles), ALREADY FIXED: The path the rider suspected is gone. _round_trip_edit applies the model through modal._model_picked, the pick-mode result a user produces (5b187e2c5c), not ModelSearchPicker.set_model_value. f1952ff1d0 made the first modal's label check a subset check. At HEAD the [model] node passed 10 of 10 single-process runs, run concurrently, with the repo's bootstrap_profile marker applied through a scratch plugin. Without a marker the file fails at setup with RecoveryRequired, the isolation gap that Task 14 handles for its own files. Task 15 rewrites this node's Streaming pin for TASK-33006.16 and must re-run it 10 of 10 serially.
- TASK-33005.13 (Remember a provider save across Settings visits in one session), ALREADY FIXED: abb4a605e3 (TASK-33007, on the Phase 7 branch, in HEAD, not yet on dev) makes every card's result line start empty, shown through settings_screen.py::SettingsResultLine. 'have not been saved this session' now survives only in that class's docstring. Pinned by Tests/UI/test_settings_connect_rows.py::test_the_card_result_line_takes_no_row_until_there_is_a_result. This covers all five categories (AC#3); AC#2 is moot. The closure holds once PR #3057 merges.
- TASK-33006.9 (Model view rows give their controls one fixed slot so the Source column lines up), ALREADY FIXED: 0e7aa920f9 (TASK-33007, on the Phase 7 branch, in HEAD, not yet on dev) gives every Chat settings control in a view one width token ($ds-size-13; 0e2cdec06e renamed the selectors), so the Source words line up. Pinned under the production stylesheet by Tests/UI/test_console_session_settings.py::test_console_settings_select_arrows_share_one_column, which asserts that at least 6 Model view Source cells share one x. Gap for the owner to weigh: it pins 211x44 only, and by region x rather than painted text. The widths are fixed tokens, so 235x52 lays out the same, and pinning that size too is one parametrize. The closure holds once PR #3057 merges.
- TASK-33006.15 (endpoint_host recognises a scheme by '://' rather than any '//'), ALREADY FIXED: 9ed3197c85 (TASK-33006, on dev) makes console_settings_field_row.py::endpoint_host split with `urlsplit(url if "://" in url else f"//{url}")`. Tests/UI/test_console_settings_disclosures.py::test_endpoint_host_shows_host_and_port_never_userinfo has the case ('localhost:8080//v1', 'localhost:8080') beside the earlier cases.

## Needs owner

None open.

- TASK-33003.13 (Custom endpoints show the provider-choice rows their registry family supports): OWNER RULING 2026-10-09: a custom endpoint shows exactly the rows its registry family supports, the rule 5ba142ac79 already shipped (Chat/console_provider_support.py::_registry_family_provider). Task 15 records the ruling in the rider's task file (AC#1) and pins a second family with one parametrize row in Tests/UI/test_console_settings_hidden_fields.py::test_a_registry_endpoint_hides_what_its_family_hides (AC#2).
