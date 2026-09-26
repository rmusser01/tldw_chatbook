# Plan: Model-configuration redesign, Phase 1 (TASK-33001): root-cause fixes, no layout change

Spec: backlog/docs/spec-2026-09-26-model-config-redesign.md (binding authority; ADR-095 and ADR-012 amendments of 2026-09-26).
Evidence: qa/model-config-ux-review-2026-09-26/ (verified-claims.md has root-cause file:line).
Parent task: backlog/tasks/task-33001 - Model-config-P1-root-cause-fixes-no-layout-change.md

## Global Constraints

- Work ONLY in this worktree: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/model-config-p1. Start EVERY shell command with `cd <that path> &&`. Background shells reset cwd to the main checkout, which holds another session's uncommitted work: never touch it.
- Python: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python (the worktree has no venv). Run pytest FROM the worktree cwd so the worktree package wins over the editable install. Ad-hoc scripts need PYTHONPATH=<worktree>.
- NEVER write the real user profile (~/.config/tldw_cli, ~/.local/share/tldw_cli). Never bypass or disable test isolation (Tests/private_profile.py, TLDW_TEST_PRIVATE_PROFILE_NODE, HOME/XDG/TLDW_CONFIG_PATH overrides), and never run a test body outside its fixtures. Any live app run uses TLDW_CONFIG_PATH=<scratch>/config.toml with a unique [general] users_name.
- ADR-126 storage admission gate: runtime storage and config calls raise RecoveryRequired in a clean worktree. A test that fails with RecoveryRequired in setup is environmental, not a regression. Prefer gate-free unit tests plus real-implementation tests at the seam.
- Phase 1 changes NO CSS, design tokens or widget geometry.
- Size ratchets: tldw_chatbook/UI/Screens/chat_screen.py has about 32 lines of headroom (Tests/Architecture/test_screen_size_ratchet.py). tldw_chatbook/Widgets/Console/console_settings_modal.py has ZERO headroom (Tests/Architecture/test_module_size_ratchet.py:68): net lines must be <= 0. ADR-097 ratchets never rise.
- ADR-031: never bind Ctrl+C/V/X/S/D/Z/A/R/W. Footer hints must match working bindings.
- TDD: write the failing test first, see it fail, then make it pass. For config and provider surfaces, add at least one real-implementation integration test (no kwargs fakes).
- Where a fix changes behaviour an existing test pins, rewrite that test on purpose and say so in the report.
- Commit per task: `fix(model-config): <summary> (TASK-33001.N)` or `feat(...)`, ending with the line `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never push, never merge.
- Backlog hygiene per CLAUDE.md: in the subtask file, tick every AC you satisfy ([x]), set status Done, and add an `## Implementation Notes` section. Update the Docs/User_Guide pages the task names, including their Verified-against stamps.
- Before you report DONE: run the tests covering your change, plus `PYTHON=<venv> ./scripts/preflight.sh` (~35s; it must exit 0; do not pipe it through tail, which masks the exit code).

## Task 1: A provider switch resolves the target provider's own model (TASK-33001.1)

Task file: backlog/tasks/task-33001.1 - A-provider-switch-resolves-the-target-provider-s-own-model.md

### Why

Both Console editors rebase a provider change with no model: the Alt+M popover (Widgets/Console/console_model_popover.py:1090-1101) and the Conversation settings modal (Widgets/Console/console_settings_modal.py:5372-5373). Both go through ConsoleChatController.rebase_console_settings_draft (Chat/console_chat_controller.py:12946, :12982) to build_target_default_console_session_settings and then resolve_effective_chat_configuration (Chat/console_session_settings.py:1254). That resolver tries models in the order session, chat_defaults.model, provider model/api_model/default_model (:1269-1275). It applies chat_defaults.model without checking that the explicit provider is chat_defaults.provider. So every switched-to provider inherits the global default model. With the shipped default OpenAI / gpt-5.6-terra (config.py:4612-4613), switching llama.cpp to Anthropic yields gpt-5.6-terra.

This is not by design. ADR-006's precedence treats chat_defaults.model as belonging to chat_defaults.provider. The TASK-364 comment at console_model_popover.py:1097-1099 says a stale model from another provider must not linger. TASK-14812 AC#6 says a provider change cannot retain a model from the previous provider.

Tests pin only the same-provider precedence: test_chat_defaults_model_outranks_provider_fallback (Tests/Chat/test_console_session_settings.py:161). No test covers a cross-provider rebase with no model; Tests/Chat/test_console_settings_apply.py:763-775 asserts only the provider. The resolver already computes a canonical provider id (_canonical_chat_provider_id, console_session_settings.py:1970), which handles case, legacy aliases and custom-endpoint registry ids (ADR-146). Absorbs TASK-14812.

### Acceptance criteria

- [ ] #1 Switching provider in the Alt+M popover or the Conversation settings modal fills the model from the target provider's own configuration (model, api_model or default_model), never from chat_defaults.model when chat_defaults names a different provider
- [ ] #2 When the target is chat_defaults.provider under canonical identity (case, legacy alias spelling or custom-endpoint registry id), chat_defaults.model still outranks the provider fallback, and test_chat_defaults_model_outranks_provider_fallback passes unchanged
- [ ] #3 A target provider with no configured model yields a draft with no model that reads as needing one; no other provider's model is borrowed
- [ ] #4 New regression tests run a cross-provider rebase with no model through the real rebase_console_settings_draft path and the real resolver (not mocked), for a local-to-cloud and a cloud-to-cloud switch, and assert the resulting model
- [ ] #5 New chats (Ctrl+T and the startup chat) resolve the same provider and model as before this change
- [ ] #6 TASK-14812 AC#6 holds again and TASK-14812 is closed as Done
- [ ] #7 Docs/User_Guide/console.md's Alt+M section says a provider switch picks that provider's own model

### References

- tldw_chatbook/Chat/console_session_settings.py
- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/Widgets/Console/console_model_popover.py
- tldw_chatbook/Widgets/Console/console_settings_modal.py
- tldw_chatbook/config.py
- Tests/Chat/test_console_session_settings.py
- Tests/Chat/test_console_settings_apply.py
- backlog/tasks/task-14812 - Unify-Console-model-selection-into-a-searchable-picker.md
- backlog/decisions/006-provider-aware-generation-settings.md
- Docs/User_Guide/console.md

## Task 2: One field-support decision that matches what the request forwards (TASK-33001.2)

Task file: backlog/tasks/task-33001.2 - One-field-support-decision-that-matches-what-the-request-forwards.md

### Why

For Anthropic, the Console modal shows Min P, Seed, Presence and Frequency like supported fields (Widgets/Console/console_settings_modal.py:1976-2017). But PROVIDER_PARAM_MAP['anthropic'] carries none of them (Chat/Chat_Functions.py:281-300), and project_chat_handler_kwargs forwards only mapped keys, so the values are dropped with no warning (:1431-1435).

Root cause: _supported_console_settings_fields (Chat/console_chat_controller.py:774-802) starts from every field in FULL_MODEL_DEFAULT_FIELDS and gates only the reasoning and thinking fields. The projection exists three times, and the copies drift independently:
- The controller copy above.
- _supported_profile_fields (Chat/console_settings_defaults.py:344), which the model-default writer uses at :1237. Its docstring says it mirrors the controller.
- Settings' _model_profile_field_supported (UI/Screens/settings_screen.py:12596), which returns True for every sampler.

The spec names the single replacement supported_generation_fields(): the existing capability projection intersected with PROVIDER_PARAM_MAP. Watch the mapping trap: the Console request sends top_p under both 'topp' and 'maxp' (Chat/console_provider_gateway.py:5138-5139), and OpenAI's map carries only 'maxp'. A provider with no map entry must keep today's field set (TASK-30012 AC#3).

This task fixes the data. The one-line 'hidden for <provider>' summary in the Console modal is layout work outside this phase. Settings already shows 'Unavailable for <provider>' on unsupported model-profile rows (settings_screen.py:12602-12608), so the fix becomes visible there with no layout change.

### Acceptance criteria

- [ ] #1 One function (supported_generation_fields, per the spec) answers which generation fields a provider and model accept, and the Console draft rebase, the model-default writer and Settings Providers & Models model-default rows all use it
- [ ] #2 The mirrored projection in console_settings_defaults.py and the sampler-always-supported branch in settings_screen.py are gone
- [ ] #3 A field counts as supported only when the existing capability rules allow it and the provider's request mapping forwards a key that carries it; the mapping from each field to its request key or keys is defined once
- [ ] #4 For Anthropic, Min P, Seed, Presence penalty and Frequency penalty are unsupported, while Temperature, Top P, Top K, Max tokens, Streaming and the thinking fields keep today's support
- [ ] #5 For OpenAI, Top P stays supported (its map forwards 'maxp')
- [ ] #6 A provider with no PROVIDER_PARAM_MAP entry keeps today's field set (TASK-30012 AC#3)
- [ ] #7 A table-driven test over every key in CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS fails if a field the request drops is reported supported, or a field it forwards is reported unsupported
- [ ] #8 Settings Providers & Models shows its existing 'Unavailable for Anthropic' state on those four rows, captured rendered at 211x44
- [ ] #9 A value already saved for a field that becomes unsupported stays in config untouched and never reaches the request
- [ ] #10 A surface that still renders a now-unsupported field (the Conversation settings modal) keeps working: editing the field raises no error, and its value is neither sent nor saved as a model default

### References

- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/Chat/console_settings_defaults.py
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/Chat/Chat_Functions.py
- tldw_chatbook/Chat/console_provider_gateway.py
- tldw_chatbook/Chat/console_settings_apply.py
- tldw_chatbook/Chat/console_session_settings.py
- backlog/tasks/task-30012 - Recompose-Conversation-Settings-around-connection-first-disclosure.md
- backlog/decisions/006-provider-aware-generation-settings.md

## Task 3: Provider Test shows each endpoint fact once (TASK-33001.3)

Task file: backlog/tasks/task-33001.3 - Provider-Test-shows-each-endpoint-fact-once.md

### Why

This happens on URL-based providers only (Chat/provider_endpoint_contract.py:24). How the duplicate gets in:
1. Settings Test builds its detail line in _provider_readiness_test_report, which appends any stored evidence for the current draft identity (UI/Screens/settings_screen.py:15143-15150).
2. action_settings_test_category passes that detail to the probe worker (:30801-30824).
3. _apply_provider_endpoint_probe_outcome appends the fresh evidence to the same detail (:15458-15469).

ProviderTestEvidenceStore.begin (Chat/provider_test_evidence.py:593) resets the store, but the stale copy is already inside the string. So a second Test on an unchanged draft after a failed probe shows 'model listing failed' next to 'model listing reached', and 'generation not tested' twice. While the probe runs, the in-flight line shows the old failure beside 'checking' (:30806-30808).

Both appends arrived in #2365 (939dee8dc2). Existing tests cover single runs only (Tests/UI/test_settings_configuration_hub.py:4533, :4576, :4622). Cloud providers are unaffected, because their Test never probes. That is by design (TASK-191, TASK-30011 AC#2).

### Acceptance criteria

- [ ] #1 Re-running Test on an unchanged draft after a failed probe shows only the new probe's outcome; one run never reports both a failure and a success
- [ ] #2 'generation not tested' appears at most once in the result and at most once in the toast
- [ ] #3 While a probe is in flight, the result says 'checking' without the previous run's outcome
- [ ] #4 A mounted test runs Test twice on one draft (a failed probe, then a reachable one) with the real ProviderTestEvidenceStore, and asserts each fact appears once
- [ ] #5 The single-run tests at Tests/UI/test_settings_configuration_hub.py:4533, :4576 and :4622 pass unchanged

### References

- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/Chat/provider_test_evidence.py
- tldw_chatbook/Chat/provider_endpoint_contract.py
- Tests/UI/test_settings_configuration_hub.py

## Task 4: F6 and Shift+F6 cycle the Settings panes (TASK-33001.4)

Task file: backlog/tasks/task-33001.4 - F6-and-Shift-F6-cycle-the-Settings-panes.md

### Why

F6 is the app-global 'next pane' key and is advertised in the footer (app.py:7750; ADR-031 rule 1). The app hands it to the active screen's action_focus_next_workbench_pane and otherwise shows 'No workbench pane focus target is available.' (app.py:20175-20186). SettingsScreen (UI/Screens/settings_screen.py:2731) defines no handler, so on Settings F6 only shows that notice. Docs/User_Guide/settings.md:877 records this as a known limitation.

This breaks ADR-031 rule 4, which says advertised keys must work. It also leaves Settings as the one workbench whose panes can be reached only with Tab: Model is Tab stop 23.

Personas, Library and Workflows already cycle panes with the shared focus_relative_workbench_pane helper (Widgets/workbench_focus.py:20). See personas_screen.py:16333-16347, and its shift+f6 binding at :1086. The Settings panes are #settings-category-pane, #settings-detail-pane and #settings-impact-pane (settings_screen.py:22436-22476). Related: TASK-2831 (category focus intent lost in recompose races) touches the same rail focus path.

### Acceptance criteria

- [ ] #1 On Settings, F6 moves focus from the category rail to the detail pane to the inspector (when it has a focus target) and wraps around; Shift+F6 goes the other way
- [ ] #2 F6 into the rail lands on the active category row, including right after a category switch
- [ ] #3 The 'No workbench pane focus target is available.' notice never appears on Settings
- [ ] #4 F6 pressed in a focused text field moves focus and leaves the field's value unchanged
- [ ] #5 A mounted test drives F6 and Shift+F6 with real key presses (not direct action calls) and asserts which pane holds focus after each press
- [ ] #6 Live evidence at 211x44 shows focus moving across the three panes
- [ ] #7 The known-limitation paragraph at Docs/User_Guide/settings.md:877 is replaced by F6 and Shift+F6 rows in the Settings keys table

### References

- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/Widgets/workbench_focus.py
- tldw_chatbook/app.py
- tldw_chatbook/UI/Screens/personas_screen.py
- Docs/User_Guide/settings.md
- backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md

## Task 5: Untouched open chats follow newly saved defaults (D1) (TASK-33001.5)

Task file: backlog/tasks/task-33001.5 - Untouched-open-chats-follow-newly-saved-defaults-D1.md

### Why

Owner decision D1, recorded in the ADR-095 amendment of 2026-09-26 (drafted in this worktree with the spec). The task-177 refresh (_maybe_refresh_stale_default_console_settings, UI/Console_Modules/session.py:3743) already converges a pristine chat: no messages, no user work, and settings still equal to their canonical baseline. Three readiness gates stop it:
- it returns early when the chat can already send (:3786);
- it skips labels outside _CONSOLE_REFRESHABLE_BLOCKED_LABELS (:3788);
- it swaps only when the new defaults can send (:3814).

Result: after a Settings Providers & Models save, an untouched chat on a keyless provider that reads Ready (llama.cpp reads Ready even when its server is down) keeps its old provider, model and sampling, and nothing on screen says so (C1(b)).

The same early return causes the first-run gap. Only 'Start chatting' stages the CONSOLE_FIRST_CHAT handoff (UI/Wizards/FirstRunSetupWizard.py:9936). The Home, Library and Notes exits (:7512-7531) and Skip (:9604-9610) do not. The Settings exit (:9585-9595) leaves setup incomplete and opens Providers & Models, so its outcome is an ordinary Settings save. Even the handoff is released without effect if config changes before Console consumes it (session.py:1353-1360). The wizard never bumps console_new_chat_default_generation, so the creation-time check does not block convergence here. Both gaps share one root cause, so this task owns both outcomes.

Cost: this refresh runs on every provider/model display rebuild through _ensure_active_console_session_settings (session.py:3695-3741; see the TASK-26839 note at :3715-3720), which task-24454 places on the composer keystroke path. Today the send-capable early return skips re-deriving defaults. Without it, re-derivation needs another bound.

The amendment says what stays:
- the eligibility test, unchanged, and the creation-time default-generation check;
- Persona identity;
- the task-16475 provider-swap notice;
- source-owned chats (Duplicate, Branch, Continue, handoffs), and chats with work never rebase silently (ADR-095:26-28).

The explicit 'Use saved defaults' action for chats that hold work is not part of this task. Guard tests that must keep passing: Tests/UI/test_console_session_settings.py:12697 and :12727, Tests/UI/test_console_provider_persistence_regressions.py:281, and Tests/Chat/test_workspace_default_session.py:276.

### Acceptance criteria

- [ ] #1 After a Settings Providers & Models save, an open Console chat with no messages, no edited settings and no user work shows the saved provider, model and sampling defaults the next time Console renders, whether its previous provider read Ready or blocked
- [ ] #2 A converged pristine chat holds exactly what a blank chat created at that moment (Ctrl+T) would hold, even when those settings cannot send yet
- [ ] #3 After first-run setup completes through any completing exit (Start chatting, Home, Library, Notes or Skip), the next time Console is shown an untouched chat uses the provider and model that setup saved; the Start chatting handoff behaviour pinned by Tests/Wizards/test_first_run_setup_wizard.py:10536-10836 is unchanged
- [ ] #4 A chat with any message, edited setting, user-work marker or applied /system prompt keeps its settings, and the tests at Tests/UI/test_console_session_settings.py:12697 and :12727 pass unchanged
- [ ] #5 Chats created with explicit source settings (Duplicate, Branch, Continue, handoffs) never converge (ADR-095:26-28)
- [ ] #6 A pristine chat created before a Console 'Make default for new chats' is still skipped, as the amendment requires
- [ ] #7 A Persona chat keeps its Persona, system prompt and label when it converges, and test_provider_setup_recovery_keeps_created_persona_prompt (Tests/Chat/test_workspace_default_session.py:276) passes
- [ ] #8 A convergence that changes provider still posts the task-16475 swap notice, and Tests/UI/test_console_provider_persistence_regressions.py:281 passes
- [ ] #9 Between two settings saves, ordinary provider/model display rebuilds and composer keystrokes re-derive a pristine chat's defaults at most once, shown by a call-counting test shaped like Tests/UI/test_console_settings_title_laziness.py
- [ ] #10 The convergence check adds no disk, keyring or network access to the Console display path
- [ ] #11 test_real_journey_settings_save_unblocks_console_without_restart (Tests/UI/test_console_session_settings.py:12620) is extended to save through the real Settings save path into a scratch TLDW_CONFIG_PATH (no mocked config writer). It asserts that an open untouched llama.cpp chat that reads Ready follows the new default, and that a chat with one message does not
- [ ] #12 Any existing test that pins 'a send-capable untouched chat never converges' is rewritten on purpose and named in the PR notes
- [ ] #13 Docs/User_Guide/settings.md:244-246 and :546-547 describe the new scope: new chats and untouched open chats take saved defaults, and chats with work keep theirs

### References

- tldw_chatbook/UI/Console_Modules/session.py
- tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/Chat/console_chat_store.py
- backlog/decisions/095-conversation-owned-console-generation-settings.md
- backlog/tasks/task-177 - Propagate-provider-readiness-to-Console-in-session-after-Settings-save.md
- Tests/UI/test_console_session_settings.py
- Tests/UI/test_console_provider_persistence_regressions.py
- Tests/Chat/test_workspace_default_session.py
- Tests/UI/test_console_settings_title_laziness.py
- Tests/Wizards/test_first_run_setup_wizard.py
- Docs/User_Guide/settings.md

## Task 6: Remove the provider-only command-palette commands (TASK-33001.6)

Task file: backlog/tasks/task-33001.6 - Remove-the-provider-only-command-palette-commands.md

### Why

LLMProviderProvider (app.py:1493-1603, registered in COMMANDS at app.py:7779) offers 'LLM Provider Management: Switch to <X>' for every provider. It labels them by title-casing raw keys (app.py:1518, for example 'Local Llamacpp'), and its discover list hard-codes five names (:1535).

A switch stages a CONSOLE_PROVIDER handoff. Console applies it with that provider's first configured model and never shows the user which one (UI/Screens/chat_screen.py:8017-8100). The command bypasses the model picker, and its names match no other surface (C7(d)).

It is the only producer of the CONSOLE_PROVIDER channel (UI/Navigation/pending_handoff_store.py:111, with a channel-specific branch at :697). That channel's Console consumer (chat_screen.py:8061, scheduled at :16832, :16909 and :23729) would be left dead.

No capability is lost. 'Console: Change model…' (UI/console_command_provider.py:43) already opens the model surface from the palette, and the chips show the current provider (ADR-011 keeps primary actions on screen). Removing the consumer also frees chat_screen.py lines under its size ratchet.

### Acceptance criteria

- [ ] #1 The command palette no longer offers any 'LLM Provider Management' entry, neither 'Switch to' nor 'Show Current Provider'
- [ ] #2 'Console: Change model…' still opens the model surface from the palette
- [ ] #3 The CONSOLE_PROVIDER handoff channel, its store branch and its Console consumer, which have no other producer, are removed
- [ ] #4 The tests that pin the removed channel, consumer and provider are updated on purpose: Tests/UI/test_command_palette_providers.py, Tests/UI/test_chat_screen_resume_handoff_registration.py:36, Tests/test_application_state_ownership.py:2229, the CONSOLE_PROVIDER cases in Tests/State/test_pending_handoff_store.py, the intent and LLMProviderProvider half of test_real_console_consumes_typed_provider_intents_and_opens_real_picker (Tests/ProductionApp/test_provider_selection_ownership.py:200), and the consumer stubs in Tests/UI/test_console_roleplay_resume_navigation.py
- [ ] #5 Console resume and first-chat handoffs behave as before, and their existing tests pass
- [ ] #6 chat_screen.py's line count drops, and the new count is recorded against its screen-size budget
- [ ] #7 Tests/UI/COMMAND_PALETTE_TESTING.md no longer lists the provider-management provider

### References

- tldw_chatbook/app.py
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Navigation/pending_handoff_store.py
- tldw_chatbook/UI/console_command_provider.py
- Tests/UI/test_command_palette_providers.py
- Tests/UI/test_chat_screen_resume_handoff_registration.py
- Tests/test_application_state_ownership.py
- Tests/State/test_pending_handoff_store.py
- Tests/ProductionApp/test_provider_selection_ownership.py
- Tests/UI/test_console_roleplay_resume_navigation.py
- Tests/UI/COMMAND_PALETTE_TESTING.md
- backlog/decisions/011-chatbook-workbench-ui-system.md

## Task 7: Polish: fix the review's minor defects on model-config surfaces (TASK-33001.7)

Task file: backlog/tasks/task-33001.7 - Polish-fix-the-review-s-minor-defects-on-model-config-surfaces.md

### Why

These are the review's minor observations (report.md 'Minor observations') that belong to model-configuration surfaces and need no layout change. Each is small, and none has an open task.
- ModelSearchPicker says '1 models available' (Widgets/model_search_picker.py:594). It also truncates results to MAX_RESULTS = 20 (:68, :640) without saying that more matched.
- SettingsURLInput renders endpoint URLs with a zero-width break after the scheme (UI/Screens/settings_screen.py:1318, :2231-2236, :2353). The stored value stays clean. The break exists only to stop textual-web autolinking, but it is drawn in native terminals too, so a URL copied from the screen may carry it.
- Saving a keyless provider writes api_key_env_var and credential_source keys it never uses: the save builds its mutation with a credential source for every provider (settings_screen.py:30088). ADR-012 already omits credential fields from writes in the Anthropic subscription case.
- Inputs clear to their placeholder on focus, hiding the committed value while it is edited. This came from a live run and is not yet traced to code.

Out of scope: toast and tooltip placement, which are app-wide.

### Acceptance criteria

- [ ] #1 The picker's status line reads '1 model available' for one model and '<N> models available' otherwise
- [ ] #2 When the picker shows fewer results than matched, its status line says how many matched and that typing narrows the list
- [ ] #3 In a native terminal, Settings endpoint fields render the URL with no zero-width character. The autolink break applies only under textual-web, and a test covers both
- [ ] #4 Saving a keyless provider (for example llama.cpp or Ollama) with no key configured writes no api_key_env_var or credential_source key to its config section. A real-implementation test on a scratch TLDW_CONFIG_PATH asserts the written section
- [ ] #5 Focusing a field in the Conversation settings modal or Settings ▸ Providers & Models never hides its committed value, shown by a painted-text probe driven by real key presses at 211x44
- [ ] #6 console_settings_modal.py does not grow

### References

- tldw_chatbook/Widgets/model_search_picker.py
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/Widgets/Console/console_settings_modal.py
- backlog/decisions/012-provider-credential-settings-boundary.md
- qa/model-config-ux-review-2026-09-26/report.md
