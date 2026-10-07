# Plan: Model config P2: one field table and honest copy (TASK-33002)

Spec: backlog/docs/spec-2026-09-26-model-config-redesign.md (the binding authority; ADR-095 and ADR-012 amendments of 2026-09-26).
Evidence: qa/model-config-ux-review-2026-09-26/ (mockups-211x44.md, verified-claims.md).
Parent task: backlog/tasks/task-33002 - Model-config-P2-one-field-table-and-honest-copy.md

## Phase goal (parent)

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

### Parent acceptance criteria

- [ ] #1 Every model-configuration field has one label and one help line wherever it is edited
- [ ] #2 The Provider Test result reads as labelled rows that lead with the outcome, with no config-key spellings and no leaked secrets
- [ ] #3 The Providers & Models save and the State line say what a save applies to
- [ ] #4 The Settings State line keeps naming its save model and counts unsaved edits
- [ ] #5 Console chips and notices name providers by display name, and one display-name map serves every surface
- [ ] #6 No Settings copy refers to 'Console Defaults' or 'Override current Console model'
- [ ] #7 Rendered captures at 211x44, plus one at 235x52, of every changed surface are attached to the PR
- [ ] #8 The phase changes no layout, CSS, design token or geometry
- [ ] #9 console_settings_modal.py does not grow, chat_screen.py stays within its budget, and no ADR-097 ratchet value rises
- [ ] #10 TASK-486 is closed as Done
- [ ] #11 Docs/User_Guide pages updated (settings.md, console.md), including their Verified-against stamps
- [ ] #12 ./scripts/preflight.sh passes

## Global Constraints

- Work ONLY in this worktree: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/model-config-p2. Start EVERY shell command with `cd <that path> &&`. Background shells reset cwd to the main checkout, which holds another session's uncommitted work: never touch it.
- Python: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python (the worktree has no venv). Run pytest FROM the worktree cwd. Ad-hoc scripts need PYTHONPATH=<worktree>.
- NEVER write the real user profile (~/.config/tldw_cli, ~/.local/share/tldw_cli). Never bypass or disable test isolation (Tests/private_profile.py / private_profile_test, TLDW_TEST_PRIVATE_PROFILE_NODE, HOME/XDG/TLDW_CONFIG_PATH overrides), and never run a test body outside its fixtures. Any live app run uses TLDW_CONFIG_PATH=<scratch>/config.toml with a unique [general] users_name, and captures at FULL SCREEN: 211x44 (primary) and 235x52.
- ADR-126 storage gate: RecoveryRequired in a clean worktree is environmental. Prefer gate-free unit tests plus real-implementation tests at the seam, and compare failure-name sets against origin/dev.
- Size ratchets never rise (ADR-097). console_settings_modal.py has ZERO headroom: net lines must be <= 0. Check Tests/Architecture/test_module_size_ratchet.py and test_screen_size_ratchet.py before and after.
- Geometry (heights, widths, spacing) only through tokens in tldw_chatbook/css/core/_variables.tcss (ADR-150/161; raw dimension literals are banned in .tcss sheets). Rebuild the CSS bundle with the repo script; never hand-edit the bundle.
- ADR-031: never bind Ctrl+C/V/X/S/D/Z/A/R/W. Footer hints must match working bindings.
- TDD: write the failing test first. For config and provider surfaces, add at least one real-implementation integration test (no kwargs fakes).
- Where a fix changes behaviour an existing test pins, rewrite that test on purpose and name it in the report.
- Commit per task: `fix(model-config): <summary> (TASK-33002.N)` or `feat(...)`, ending with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never push. NEVER merge origin/dev into the branch (the owner requires rebases).
- Backlog hygiene per CLAUDE.md: in the subtask file, tick the ACs you satisfy, set status Done, and add `## Implementation Notes`. Update the Docs/User_Guide pages the task names, including their Verified-against stamps.
- Before you report DONE: run the covering tests plus `PYTHON=<venv> ./scripts/preflight.sh` (the exit code must be 0; do not pipe it through tail).

Phase 2 builds on Phase 1 (PR #2846): supported_generation_fields() in Chat/console_provider_support.py (TASK-33001.2), the Provider Test single-append fix (TASK-33001.3) and pristine-chat convergence (TASK-33001.5). This worktree is cut from the Phase 1 branch and will be rebased onto dev after #2846 merges.

## Task 1: One field table labels every model-configuration field (TASK-33002.1)

Task file: backlog/tasks/task-33002.1 - One-field-table-labels-every-model-configuration-field.md
Depends on: TASK-33001.2

### Why

Four editors give the same fields different labels:
- 'Think budget' (UI/Screens/settings_screen.py:17378, :18695, :15861), 'Budget' (Widgets/Console/console_settings_modal.py:2125) and 'Thinking budget' (settings_screen.py:12177).
- 'Budget strategy' (:18403) and 'Budget mode' (modal :2250).
- 'When limit nears' (:18427) and 'Behavior' (modal :2295).
- 'Endpoint' (:16810), and 'Base URL' in the modal (:1755) and the custom-endpoint editor (:17567).
- 'Presence' and 'Frequency' (:17274, :17284, :18658, :18668; modal :2005, :2012) against 'Presence penalty' and 'Frequency penalty' (:12171-12172, :15825-15831).
- 'Response max tokens' (:17254, :18636, :15813) and 'Max tokens' (:12169).

Settings also keeps two private copies of field copy that already disagree with each other: the dirty-field label map (:12156-12177) and the inspector guidance table (:15800-15860). The Console surfaces show raw sampler names with no help (report, heuristic 10).

STORAGE_FIELD_LABELS (UI/Screens/settings_storage_defaults.py:55) is the in-repo precedent for one label table. This table extends phase 1's field-to-request-key definition, so labels, help, ranges and request keys cannot drift apart. Rendering help lines inside the Console modal is layout work outside this phase.

### Acceptance criteria

- [ ] #1 One table gives each model-configuration field its label, a plain-language one-line help and its valid range. It covers every field in FULL_MODEL_DEFAULT_FIELDS, plus Endpoint and the two context-budget fields. Generation fields also carry their request key or keys from phase 1's definition, with no second copy
- [ ] #2 The Alt+M popover, the Conversation settings modal, Settings Providers & Models model defaults (form rows, inspector and dirty-field names) and Settings Console Behavior fallbacks all take their labels from the table
- [ ] #3 A test collects each of those surfaces' rendered labels and fails on any label that differs from the table
- [ ] #4 Each drift pair resolves to one label: 'Thinking budget', 'Endpoint', 'Max tokens', 'Presence penalty' and 'Frequency penalty', and one shared label for each of the two context-budget pairs
- [ ] #5 Settings' focused-field inspector for model-default fields shows the table's help and range; the inline guidance table and the dirty-field label map in settings_screen.py are gone
- [ ] #6 Help lines say what the field does in plain words and contain no config keys (no 'min_p', no 'chat_defaults.*')
- [ ] #7 console_settings_modal.py does not grow (its module-size ratchet at 7,807 has zero headroom)
- [ ] #8 Tests that pin the old labels are rewritten on purpose and named in the PR notes
- [ ] #9 Rendered captures at 211x44 of the modal's Model view and of the Settings model-default section show the unified labels

### References

- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/Widgets/Console/console_settings_modal.py
- tldw_chatbook/Widgets/Console/console_model_popover.py
- tldw_chatbook/UI/Screens/settings_storage_defaults.py
- tldw_chatbook/Chat/console_settings_apply.py
- Tests/Architecture/test_module_size_ratchet.py

## Task 2: Provider Test result reads as labelled rows (TASK-33002.2)

Task file: backlog/tasks/task-33002.2 - Provider-Test-result-reads-as-labelled-rows.md
Depends on: TASK-33001.3

### Why

_build_provider_readiness_findings (UI/Screens/settings_screen.py:15213-15335) puts all of these in one list and joins it with ' | ' (:15325, :15332): the verdict prose, 'model=', 'api_key_source=config:api_settings.<provider>.api_key', '<ENV>=<redacted>', the endpoint summary and 'configuration=complete'. The line is shown in #settings-provider-test-result (:15493-15502). The in-flight line adds '| model listing checking' (:30806-30808).

An unreachable server leads with 'configuration is complete'. The failure sits in the eighth segment, with no next step (report, heuristic 9).

The spec fixes the result as labelled rows: Config, Key, Endpoint, Model, Generation. Each fact appears once, and no ' | ' dump remains.

Redaction is name-based. It catches api_key, token, secret and password, but a custom-named credential query parameter such as '?mycred=SEKRET' still prints verbatim (TASK-486, absorbed).

Cloud copy stays a local readiness check: key present, provider acceptance not tested (TASK-191, TASK-30011 AC#2; pinned by test_settings_provider_test_skips_probe_for_cloud_providers at Tests/UI/test_settings_configuration_hub.py:4622).

### Acceptance criteria

- [ ] #1 The Test result shows labelled rows, one fact per row: Config, Key, Endpoint, Model and Generation, plus the model listing outcome for URL providers. It has no ' | '-joined line and no key=value config spellings ('model=', 'api_key_source=', 'configuration=')
- [ ] #2 The first row states the outcome that matters: when the live probe fails, it names the failure and a next step (such as 'start the server or check the URL') instead of 'configuration is complete'
- [ ] #3 The Key row states the key's source (saved in config, from env var NAME, or missing) and never its value
- [ ] #4 A custom-named credential query parameter in the endpoint (for example '?mycred=SEKRET') never appears verbatim in any row or in the toast; a test proves it (TASK-486)
- [ ] #5 For cloud providers the rows still say the key is present but not verified and that generation is not tested; test_settings_provider_test_skips_probe_for_cloud_providers (:4622) passes
- [ ] #6 The toast stays a one-line summary consistent with the first row
- [ ] #7 Tests that pin the pipe form are rewritten on purpose: Tests/UI/test_settings_provider_test_draft.py (:824, :903, :920, :934, :1522), Tests/UI/test_settings_subscription_readiness.py (:184, :202, :207, :271) and Tests/UI/test_settings_configuration_hub.py (:10121, :10430)
- [ ] #8 Rendered captures at 211x44 of a failed and a passing Test are attached

### References

- tldw_chatbook/UI/Screens/settings_screen.py
- Tests/UI/test_settings_provider_test_draft.py
- Tests/UI/test_settings_subscription_readiness.py
- Tests/UI/test_settings_configuration_hub.py
- backlog/tasks/task-486 - Redact-custom-named-credential-query-params-in-provider-test-endpoint-display.md
- backlog/tasks/task-30011 - Separate-Conversation-Settings-operability-from-verification-evidence.md

## Task 3: Providers & Models save and State line name their scope (TASK-33002.3)

Task file: backlog/tasks/task-33002.3 - Providers-Models-save-and-State-line-name-their-scope.md
Depends on: TASK-33001.5

### Why

C1(a): no Providers & Models copy says which chats a save affects:
- the save result 'Provider settings saved.' (UI/Screens/settings_screen.py:30215);
- the toast 'Provider and model settings saved.' (:30242-30244);
- the State scope 'Shared with Console' (:9455);
- the Provider field's Purpose, 'Selects the provider used for Console generation defaults.' (:15673-15684).
Only the User Guide says it (settings.md:244-246).

The scope is fixed by ADR-095 together with its 2026-09-26 amendment (D1). New chats take the saved defaults. Untouched open chats follow them, since phase 1. Open chats with work keep their own settings. The amendment requires the Settings save to say that open chats with work keep their own settings. The spec's scope line is 'Applies to new chats · open chats keep their own settings (in Console: Alt+M)'. Since phase 1 that line is only true for chats with work, so the copy must also name untouched chats. TASK-30012 AC#6 requires scope copy. Credentials stay in Settings (D4, ADR-012), so the copy points to Console only for per-chat changes.

### Acceptance criteria

- [ ] #1 After a successful Providers & Models save, the result line and the toast say the save applies to new chats and to open chats nobody has used yet, and that open chats with work keep their own settings (change them in Console with Alt+M)
- [ ] #2 The State line's scope for Providers & Models says the same thing in one row at 211x44, instead of 'Shared with Console'
- [ ] #3 The Provider field's inspector Purpose names the new-chat scope instead of 'Console generation defaults'
- [ ] #4 The partial-failure save copy (file written but reload failed) is unchanged
- [ ] #5 Tests that pin the old copy are rewritten on purpose and named in the PR notes
- [ ] #6 A rendered capture at 211x44 shows the save result and the State line

### References

- tldw_chatbook/UI/Screens/settings_screen.py
- backlog/decisions/095-conversation-owned-console-generation-settings.md
- backlog/decisions/012-provider-credential-settings-boundary.md
- backlog/tasks/task-30012 - Recompose-Conversation-Settings-around-connection-first-disclosure.md
- Docs/User_Guide/settings.md

## Task 4: State line keeps its save-model badge and counts unsaved edits (TASK-33002.4)

Task file: backlog/tasks/task-33002.4 - State-line-keeps-its-save-model-badge-and-counts-unsaved-edits.md
Depends on: —

### Why

When a category has unsaved edits, the State line drops its save-model badge and reads 'State: Unsaved changes | Save (s) or Revert (r) — switching categories keeps this draft.' (UI/Screens/settings_screen.py:9391-9400). The badge ('Draft — save with s', from _persistence_badge at :9408-9446) appears only when the category is clean.

The badge is ADR-033's honest label for the category's commit model (task-1717, DESIGN.md:272-278). It must stay visible and truthful. Users also cannot see how much is unsaved.

The judge's and the spec's rule: keep the badge and add a count, for example 'Draft — save with s · 2 unsaved', rather than replacing it with a dirty flag.

Tests that pin today's copy: test_state_banner_dirty_branch_keeps_priority (Tests/UI/test_settings_configuration_hub.py:12047), and, for Speech & TTS's leave-resolution wording (task-2708), test_speech_tts_dirty_banner_names_leave_resolution (:12062).

### Acceptance criteria

- [ ] #1 With unsaved edits, the State line still leads with the category's save-model badge and adds the number of unsaved fields (for example 'State: Draft — save with s · 2 unsaved'), keeping the Save and Revert guidance
- [ ] #2 The count equals the number of fields that differ from their saved values, and it updates as fields are edited or reverted back
- [ ] #3 Speech & TTS keeps its leave-resolution wording alongside the badge and count
- [ ] #4 Categories that do not stage edits show the same badge as today
- [ ] #5 test_state_banner_dirty_branch_keeps_priority (:12047) is rewritten on purpose to the new copy; test_speech_tts_dirty_banner_names_leave_resolution (:12062) is updated if its copy changes
- [ ] #6 DESIGN.md's persistence-badge note (DESIGN.md:272-278) mentions the unsaved count
- [ ] #7 A rendered capture at 211x44 shows the dirty State line

### References

- tldw_chatbook/UI/Screens/settings_screen.py
- Tests/UI/test_settings_configuration_hub.py
- DESIGN.md
- backlog/decisions/033-settings-commit-models-three-honestly-labeled.md
- backlog/tasks/task-1717 - Settings-persistence-badge-names-the-save-model-on-every-category.md

## Task 5: Console names providers by display name from one catalog (TASK-33002.5)

Task file: backlog/tasks/task-33002.5 - Console-names-providers-by-display-name-from-one-catalog.md
Depends on: TASK-33001.5

### Why

C7(d): Console prints raw provider spellings in several places.
- The status-row chip uses selection.provider, a config key, or the saved chat_defaults spelling (UI/Screens/chat_screen.py:9826-9831), rendered as 'Provider: {raw}' (Chat/console_display_state.py:776). The inspector and pickers already show catalog names.
- The task-16475 swap notice prints raw keys (UI/Console_Modules/session.py:3846). Phase 1's D1 convergence makes that notice fire more often.

The shipped [providers] table (config.py:4120-4151) spells keys such as 'Llama_cpp', 'local-llm', 'local_onnx' and 'local_transformers'. provider_display_name looks up the key exactly (Chat/provider_catalog.py:71-82), so these render raw.

Chat/console_provider_support.py:79-108 keeps a second display-name map with a title-case fallback. It uses different names ('Google' against 'Google Gemini', 'Custom OpenAI' against 'Custom OpenAI-compatible') and feeds the Console-compatible provider options at :391. So Settings lists and Console pickers disagree.

The provider value returned by _active_console_provider_model_display is also used as an identity, for example for Settings recovery navigation (chat_screen.py:20543-20553). So only rendered labels may change.

The 25-character model-chip cap is user-requested (TASK-1671, css/features/_console.tcss:253) and stays. Legacy aliases stay selectable (ADR-066, task-180). TASK-194 (the popover's own rows) is not closed here.

### Acceptance criteria

- [ ] #1 The status-row Provider chip shows the catalog display name (for example 'llama.cpp' or 'OpenAI'), the same name the inspector and pickers show, and never a raw key or the saved chat_defaults spelling
- [ ] #2 Code that uses the provider as an identity (Settings recovery navigation, readiness, prompts) still receives the config key
- [ ] #3 The stale-default swap notice names both providers by display name, and the raw-key assertion in Tests/UI/test_console_provider_persistence_regressions.py:281 is rewritten on purpose
- [ ] #4 Every provider key in the shipped [providers] table, spelled as it is there (including Llama_cpp, local-llm, local_onnx and local_transformers), renders a human name, and a test walks the shipped table and fails on any raw fallback
- [ ] #5 console_provider_support.py's private display-name map and its title-case fallback are gone, every provider label comes from Chat/provider_catalog, and tests that pin the removed names are rewritten on purpose
- [ ] #6 Legacy aliases stay selectable and keep their '(legacy alias)' label
- [ ] #7 Tests/UI/test_console_session_settings.py:11225, which pins 'Provider: llama_cpp', is rewritten on purpose
- [ ] #8 The model chip's 25-character cap is unchanged
- [ ] #9 A rendered capture at 211x44 of the status row, with one local and one cloud provider, is attached

### References

- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/Chat/console_display_state.py
- tldw_chatbook/Chat/provider_catalog.py
- tldw_chatbook/Chat/console_provider_support.py
- tldw_chatbook/UI/Console_Modules/session.py
- tldw_chatbook/config.py
- Tests/UI/test_console_session_settings.py
- Tests/UI/test_console_provider_persistence_regressions.py
- backlog/tasks/task-194 - console_model_popover-uses-the-shared-provider-display-name-catalog.md
- backlog/decisions/066-local-provider-thinking-controls.md

## Task 6: Settings copy names the real category and the real control (TASK-33002.6)

Task file: backlog/tasks/task-33002.6 - Settings-copy-names-the-real-category-and-the-real-control.md
Depends on: —

### Why

Settings copy points to a category that does not exist. 'Console Defaults' appears at UI/Screens/settings_screen.py:1975, :5486, :17214 and :17427, but the rail calls the category 'Console Behavior' (:1376; SettingsCategoryId.CONSOLE_BEHAVIOR in settings_config_models.py:26).

The collapsible titled 'Override current Console model' (:18036) holds the reasoning-replay and native-tool preferences for the current local model (task-32273, #2575). It is not a model override. The spec renames it 'Reasoning replay override'.

Tests pin the old strings at Tests/UI/test_settings_configuration_hub.py:3630 and :8691.

### Acceptance criteria

- [ ] #1 Every 'Console Defaults' string in Settings reads 'Console Behavior', matching the rail label, and the tests at Tests/UI/test_settings_configuration_hub.py:3630 and :8691 are updated on purpose
- [ ] #2 The collapsible formerly titled 'Override current Console model' is titled 'Reasoning replay override' and keeps its contents and widget ids
- [ ] #3 Rendered captures at 211x44 show both changes: Console Behavior for the retitled collapsible, and Providers & Models for the renamed 'Console Behavior' copy, which renders only there (synced with the subtask after the final review; the renamed strings render only in Providers & Models)

### References

- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/UI/Screens/settings_config_models.py
- Tests/UI/test_settings_configuration_hub.py
