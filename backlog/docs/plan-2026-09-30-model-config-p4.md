# Plan: Phase 4: Switch model, a provider·model pair switcher on Alt+M (TASK-33004)

Spec: backlog/docs/spec-2026-09-26-model-config-redesign.md (binding authority; ADR-095 and ADR-012 amendments of 2026-09-26).
Evidence: qa/model-config-ux-review-2026-09-26/ (mockups-211x44.md, verified-claims.md).
Parent task: backlog/tasks/task-33004 - Phase-4-Switch-model-a-provider-model-pair-switcher-on-Alt-M.md

## Phase goal (parent)

Phase 4 of the model-configuration redesign (judge-synthesis.md section 2a, section 3 'Keep and reshape', and section 4 P4; spec §8). It ships as one PR.

The Alt+M quick popover (Widgets/Console/console_model_popover.py, 1,432 lines) is a 170x32 form, not a switcher (C6):
- Its 2x2 action grids put 1-row buttons in 3-row cells (:253-262).
- Temperature is clipped at the fold of a body about 20 rows tall.
- Focus lands on the scroll body, because on_mount sets none (:682-687).
- Max tokens is a read-only Static, because ADR-095:74 limits the quick mask.

Other problems in the same area:
- Choosing a provider rebases with no model (:1090-1101). That is how a provider switch inherited a model from another provider (C7(a)); phase 1 guards the resolver by provider.
- The shared ModelSearchPicker never marks or pre-highlights the committed model (Widgets/model_search_picker.py:643-644, :697-705, :866-875; C7(b)).
- There are no recents, so an A/B toggle takes 16-20 keys and Alex's switch-and-tune task takes 45 actions.
- /model drops its argument (Chat/console_command_grammar.py:110; UI/Screens/chat_screen.py:19211 and :19362-19383).
- Chat settings has no Console key of its own: Ctrl+O is unbound anywhere in the app, yet the spec's footer teaches it.

This phase replaces the popover's compose() with 'Switch model':
- provider·model pairs grouped as PREVIOUS / RECENT / READY PROVIDERS / NEEDS SETUP
- a ● CURRENT mark on the current pair
- the previous pair highlighted when the switcher opens
- a one-row value strip of exactly the quick mask: Temperature, Max tokens, Streaming

Owner decisions applied:
- D3: the ADR-095 amendment of 2026-09-26 (already drafted) adds max_tokens to the quick mask, and the quick surface's editable fields equal that mask, so Thinking stays in Chat settings (spec §8 deliberate change 3).
- D4: credentials stay in Settings. NEEDS SETUP rows route there, and Console gets no key entry.

Absorbed tasks:
- task-338: Streaming in the rail.
- task-32859: one provider-selection builder, which the switcher consumes.
- task-194: popover display names. It is popover-only; spec deliberate change 4 closes it here, and the chips are phase 2's separate fix.

Depends on:
- phase 1: the provider-guarded default model and the single supported-field function
- phase 2: shared field labels, Source words and display names
- phase 3: compact control tokens and the scoped CSS leaks

Constraints:
- The ConsoleModelPopover class stays, and so do the ids console-popover-apply, -temperature, -streaming, -save-model-default and -make-new-chat-default. 17 test files query popover ids.
- chat_screen.py has 32 lines of headroom (25,331 of 25,363, Tests/Architecture/test_screen_size_ratchet.py:85), so switcher logic lives under UI/Console_Modules/.
- The popover is imported on the boot path (task-32644), so ADR-097's _ui_ready census applies.
- Textual Input owns Ctrl+A/E/D/K/U/W/X/C/V while Find has focus, so the switcher's chords are Ctrl+N and Ctrl+O.

Out of scope:
- favourites, pins and Alt+1..9 (the synthesis waits for user testing)
- any readiness probing
- task-18922's one-turn override, which stays open

### Parent acceptance criteria

- [ ] #1 Alt+M, the Provider/Model chips, the palette entry and /model [query] all open Switch model. At 211x44 and 235x52 it is 120 columns wide with auto height up to 80% of the screen, and focus starts in Find.
- [ ] #2 Alex's task (switch to a Sonnet model, set temperature 0.9 and max tokens 8192, return to the composer) takes 14 key presses. The A/B toggle takes 2, a model-only change takes 5, and a model change saved as the model default takes 16 or fewer.
- [ ] #3 No switcher path applies a provider without a model (closes C7(a) in the UI). The current pair is marked, and the previous pair is highlighted when the switcher opens (closes C7(b)). ModelSearchPicker also marks the committed model.
- [ ] #4 C6 is closed: no 3-row button cells remain, and at 211x44 Temperature and Max tokens are editable and visible without scrolling.
- [ ] #5 The quick default mask in code is temperature, max_tokens and streaming, matching the ADR-095 amendment of 2026-09-26 (D3).
- [ ] #6 Ctrl+O opens Chat settings from the Console, and the Console footer, F1 help and palette teach Alt+M, Ctrl+O and /model [query].
- [ ] #7 This PR closes task-338, task-32859 and task-194.
- [ ] #8 chat_screen.py ends the phase within its line and method budget (Tests/Architecture/test_screen_size_ratchet.py:85). If the phase shrank the file, the budget is lowered to the measured size.
- [ ] #9 No ADR-097 ratchet rises: the boot CSS byte budget (608,090, Tests/Performance/test_boot_css_byte_budget.py:117) and the _ui_ready census (1,031, Tests/Performance/test_ui_ready_module_census.py:150). Modules used only by the switcher are not imported on the boot path. If dev is already red, the branch's number equals dev's.
- [ ] #10 No binding from ADR-031 rule 2 (Ctrl+C, V, X, S, D, Z, A, R or W) is added.
- [ ] #11 Every printed key hint matches a working binding (ADR-031 rule 4).
- [ ] #12 Live captures at 211x44 and 235x52, taken with a scratch TLDW_CONFIG_PATH profile, are attached to the PR.
- [ ] #13 Docs/User_Guide pages are updated: console.md covers Switch model, /model [query], Ctrl+O, the keys table and the rail's Change action, with a Verified-against stamp.
- [ ] #14 ./scripts/preflight.sh passes.

## Global Constraints

- Work ONLY in this worktree: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/model-config-p4. Start EVERY shell command with `cd <that path> &&`. Never touch the main checkout (another session's uncommitted work).
- Python: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python (the worktree has no venv); run pytest FROM the worktree cwd. Tests/Architecture needs -p no:xdist.
- NEVER write ~/.config/tldw_cli or ~/.local/share/tldw_cli. Never bypass test isolation (private_profile_test, TLDW_TEST_PRIVATE_PROFILE_NODE, HOME/XDG/TLDW_CONFIG_PATH). Live runs: scratch TLDW_CONFIG_PATH with a unique users_name, FULL SCREEN 211x44 (primary) and 235x52.
- ADR-126 RecoveryRequired in a clean worktree is environmental: compare failure-NAME sets against origin/dev.
- Size ratchets never rise (ADR-097); console_settings_modal.py net lines <= 0 against its current row.
- Geometry only through tokens in css/core/_variables.tcss (ADR-150/161); rebuild the CSS bundle with the repo script.
- ADR-031: never bind Ctrl+C/V/X/S/D/Z/A/R/W; footer hints must match working bindings.
- TDD; real-implementation tests for config/provider surfaces; rewrite pinned tests on purpose and name them.
- Commit per task `fix|feat(model-config): <summary> (TASK-33004.N)` + `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never push; NEVER merge origin/dev into the branch.
- Tick ACs, set Done, add Implementation Notes in each subtask file. Update Docs/User_Guide CONTENT; never add "Verified against" paragraphs (record verification in task notes).
- Before DONE: covering tests + `PYTHON=<venv> ./scripts/preflight.sh` (rc 0; never piped through tail).

Phase 4 builds on Phases 1-3 (P1/P2 merged to dev; P3 in flight). This worktree is cut from the Phase 3 branch at 70c3e90ddcfcd9f18867e3fec584184aab1bac5a and will be rebased onto dev after Phase 3 merges. The Alt+M popover is replaced by the Switch model pair list (spec rule 1 'pairs only'; mockup (a) in qa/model-config-ux-review-2026-09-26/mockups-211x44.md). Owner decision D3: the quick default mask is temperature, max_tokens and streaming (ADR-095 amendment). Use Phase 3's compact control tokens and contrast tokens; do not add geometry literals. Rider TASK-33004.8 (fold hint) belongs to this phase.

## Task 1: Make the quick default mask temperature, max_tokens and streaming (D3) (TASK-33004.1)

Task file: backlog/tasks/task-33004.1 - Make-the-quick-default-mask-temperature-max_tokens-and-streaming-D3.md
Depends on: —

### Why

This implements owner decision D3, already recorded in the ADR-095 amendment of 2026-09-26 (drafted in this worktree with the spec). ADR-095:74 fixed the quick model-profile mask at temperature and streaming (QUICK_MODEL_DEFAULT_FIELDS, Chat/console_settings_apply.py:19). The switcher makes Max tokens editable, so without the change 'Save as model default' would silently drop a visible edit.

Adding the field alone would break both default actions whenever no cap is set:
- Quick-mask defaults are written from effective values (Chat/console_settings_defaults.py:216).
- Validation rejects any quick field whose value is missing (:1127-1131).
- Max tokens is optional (Chat/console_session_settings.py:350).
The amendment already says a blank Max tokens deletes the exact profile override.

Apply to this chat sends an empty mask (Widgets/Console/console_model_popover.py:1388-1392), so it is unaffected.

ADR-095:103 says quick Apply includes compaction mode. The switcher stops showing compaction but still commits the unchanged snapshot. The drafted amendment does not say so yet, and it should.

Two tests pin the old mask: Tests/Chat/test_console_settings_apply.py:158 and Tests/UI/test_console_model_apply_chips.py:235.

### Acceptance criteria

- [ ] #1 The ADR-095 amendment of 2026-09-26 (D3) links this task, and the quick model-profile mask in code is exactly temperature, max_tokens and streaming.
- [ ] #2 The same amendment also records that the quick surface no longer edits compaction mode while its Apply still commits the unchanged context-policy snapshot, so ADR-095:103's durable behaviour is unchanged.
- [ ] #3 From the quick surface, Save as model default and Make default for new chats write Max tokens to the exact model profile together with Temperature and Streaming, and they preserve sibling profiles and unexposed fields. An integration test proves this through the real default-mutation owner against a scratch config file, with no mock of the writer.
- [ ] #4 With Max tokens blank (no effective cap), both default actions succeed and leave no max-tokens override in the profile, instead of failing the quick-field validation.
- [ ] #5 Apply to this chat still sends an empty default mask and writes no configuration.
- [ ] #6 Two tests are deliberately rewritten to the new mask: test_default_profile_masks_are_exact_and_exclude_other_owners (Tests/Chat/test_console_settings_apply.py:157) and the mask assertion in test_model_apply_popover_commits_selected_provider_and_model_once (Tests/UI/test_console_model_apply_chips.py:235). The full mask is unchanged.

### References

- backlog/decisions/095-conversation-owned-console-generation-settings.md
- tldw_chatbook/Chat/console_settings_apply.py
- tldw_chatbook/Chat/console_settings_defaults.py
- tldw_chatbook/Chat/console_session_settings.py
- Tests/Chat/test_console_settings_apply.py
- Tests/UI/test_console_model_apply_chips.py

## Task 2: Build every Console provider selection through one builder (TASK-33004.2)

Task file: backlog/tasks/task-33004.2 - Build-every-Console-provider-selection-through-one-builder.md
Depends on: —

### Why

This subtask absorbs task-32859.

Commit 097411fca0 already extracted a single selection core (ConsoleSelectionCore and resolve_console_selection_core, Chat/console_chat_controller.py:593-682). Two builders still construct the full ConsoleProviderSelection separately:
- build_console_provider_selection_from_settings (:685-720)
- the screen chain at UI/Screens/chat_screen.py:9614, :9631, :9666 and :9687, which constructs its own at :9731

ConsoleProviderSelection is constructed at eight sites. The _provider_settings helpers also disagree: Chat/console_session_settings.py:1996 is registry-aware per ADR-146, while Chat/console_provider_gateway.py:7171 raises.

The switcher resolves a selection and readiness for many provider·model pairs. A third path would recreate exactly the drift task-32859 documents: the PR-2668 fix had to be applied twice. Removing the screen's copy also frees lines in chat_screen.py, which has only 32 lines of headroom. No ADR is needed, because ADR-006 ('Console owns effective session resolution') already covers the consolidation.

### Acceptance criteria

- [ ] #1 Exactly one builder constructs a ConsoleProviderSelection from Console session settings. The screen chain (chat_screen.py:9614-9731) and the controller builder (console_chat_controller.py:685-720) collapse into it, and the screen's extra fields (endpoint policy, workspace, identity) survive.
- [ ] #2 Every other ConsoleProviderSelection construction site either routes through that builder or is listed in the task notes with the reason it is not a selection built from settings.
- [ ] #3 The model / api_model / default_model fallback chain is spelled once.
- [ ] #4 The unified provider-settings lookup is registry-aware (ADR-146). The gateway's raising error policy (console_provider_gateway.py:7171) is kept at its call sites, or a recorded decision changes it.
- [ ] #5 ADR-006 precedence is unchanged and pinned by the existing Console settings tests.
- [ ] #6 The PR-2668 CE-001 custom-endpoint slug fix exists in exactly one place.
- [ ] #7 Switch model resolves each row's selection and readiness through this builder; the switcher has no selection logic of its own.
- [ ] #8 chat_screen.py is shorter at the end of this subtask than at its start.

### References

- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/Chat/console_session_settings.py
- tldw_chatbook/Chat/console_provider_gateway.py
- backlog/decisions/006-provider-aware-generation-settings.md
- backlog/decisions/146-console-custom-endpoint-registry.md
- backlog/tasks/task-32859 - Unify-the-console-provider-selection-builder.md

## Task 3: Derive recent and previous model pairs from existing conversation data (TASK-33004.3)

Task file: backlog/tasks/task-33004.3 - Derive-recent-and-previous-model-pairs-from-existing-conversation-data.md
Depends on: —

### Why

An A/B toggle takes 16-20 keys today because nothing remembers what the user ran before (report.md issue 6). Synthesis rule 6 and spec §4 rule 6 forbid a new store, and spec §10 rules out schema migrations and new indexes, so recents must come from data that already exists:
- list_all_active_conversations (DB/ChaChaNotes_DB.py:11160) returns conversations newest first with a LIMIT, but only for global scope (scope_type = 'global', :11208).
- get_conversations_metadata_by_ids (:11306) reads metadata in batches of up to 500 ids.
- parse_console_generation_settings (Chat/console_generation_settings_metadata.py:240) parses the ADR-095 snapshot and fails closed on malformed or future-version objects.

Open chats, including temporary ones that were never persisted, contribute only through the live session store. Workspace-scoped chats that are not open therefore do not appear in RECENT.

The read must stay off the UI loop and be bounded; the Fedora lag program found blocking calls on the UI loop before. Favourites and pins wait until user testing shows that PREVIOUS plus RECENT is not enough.

### Acceptance criteria

- [ ] #1 The RECENT group lists distinct provider·model pairs from open Console sessions (temporary chats included) and from the 50 most recently modified persisted global-scope chats. Pairs are newest first, and each shows when it was last used.
- [ ] #2 PREVIOUS is the most recent pair for this chat other than the current one, falling back to the most recent pair overall. After switching from A to B, PREVIOUS is A; after switching back, it is B.
- [ ] #3 The read runs off the UI thread and touches at most 50 conversations, with one batched metadata read. The switcher is usable before the read finishes, and the group fills in when the results arrive.
- [ ] #4 Malformed, missing and future-version snapshots are skipped (fail closed). A test against a real in-memory ChaChaNotes database covers this.
- [ ] #5 The read is timed on a database of 10,000 conversations and the timing is recorded in the task notes.
- [ ] #6 No schema change, index, config key or new persisted store is added.
- [ ] #7 The User Guide says that recents come from open chats and recent global-scope chats, and that workspace chats appear only while open.

### References

- tldw_chatbook/DB/ChaChaNotes_DB.py
- tldw_chatbook/Chat/console_generation_settings_metadata.py
- tldw_chatbook/UI/Console_Modules/model_switcher.py
- Docs/User_Guide/console.md

## Task 4: Replace the quick popover form with the Switch model pair list (TASK-33004.4)

Task file: backlog/tasks/task-33004.4 - Replace-the-quick-popover-form-with-the-Switch-model-pair-list.md
Depends on: TASK-33001, TASK-33002, TASK-33003, TASK-33004.2, TASK-33004.3

### Why

This is the core of the phase. ConsoleModelPopover.compose() is replaced with the pair list from judge-synthesis.md section 2a. The class, its constructor seams and the kept ids stay, so the Apply path does not change. The seams are origin, draft_rebaser, live_committer and default_readiness_resolver, pushed at UI/Screens/chat_screen.py:5527-5550.

Removed from the popover:
- the 'Conversation settings' title (Widgets/Console/console_model_popover.py:455)
- the provider Select (:472-477), whose change rebases with no model (:1090-1101; C7(a))
- the context and compaction block (:524-591)
- the Defaults… subview (:592-621)
- both 2x2 grids of 3-row cells (:253-262, :630-670; C6)

A fixed width token replaces the 85%/170-column wide tier (:162-184).

Each row's readiness comes from the existing config-only resolver (_console_default_readiness, chat_screen.py:2846), passed in by the screen, so the widget calls no provider service (ADR-011). The readiness words must stay honest. Ready means 'no known blocker', never acceptance (TASK-30011 AC#2). Readiness work must not repeat on every keystroke (task-24454, task-32804.3).

Catalogs come from the existing snapshot resolution (resolve_provider_model_options, UI/Screens/provider_model_resolution.py:237). ModelSearchPicker's rule that typing never triggers discovery (Widgets/model_search_picker.py:60-67) carries over. So do the guarantees TASK-14812 gave the old picker: explicit loading, empty and unavailable states (AC#4), and a custom model id escape hatch validated as bounded single-line text (AC#5, AC#7).

NEEDS SETUP rows reuse the existing Settings recovery navigation (_open_console_provider_recovery, chat_screen.py:20533-20563) with that row's own provider, per ADR-012 and owner decision D4.

Other rules this list follows:
- Legacy aliases may be hidden but not deleted (ADR-066).
- Custom endpoints appear as custom-ep instances, and the built-in custom and custom_2 slots stay (ADR-146).
- The active model is always preserved (ADR-020).

This subtask closes task-194, which is about the popover's provider labels only. Behaviour when no provider is configured (the setup gate) is unchanged in this phase.

Absorbs TASK-194.

### Acceptance criteria

- [ ] #1 Switch model shows four groups: PREVIOUS, RECENT, READY PROVIDERS (at most 3 models per ready provider plus an '… N more' row) and NEEDS SETUP. Each row is one line showing the model, the provider's display name, context size, readiness words and last use.
- [ ] #2 Every selectable row is a provider·model pair, and no key or row applies a provider without a model.
- [ ] #3 Provider names come from the shared display-name catalog (provider_display_name, Chat/provider_catalog.py:71), never from raw config keys, which satisfies TASK-194 AC#1.
- [ ] #4 The current pair is marked '● CURRENT' in text and is always listed, even when its provider's catalog omits it (ADR-020).
- [ ] #5 The PREVIOUS row is highlighted when the switcher opens.
- [ ] #6 Typing in Find filters every provider's cached catalog and saved model list in memory and highlights the best match. No keystroke triggers discovery, a network call or a credential-store read.
- [ ] #7 With an OpenRouter-sized catalog of at least 2,000 models, filtering on one keystroke takes under 50 ms on the UI thread, measured by a test.
- [ ] #8 A model id that is in no catalog can still be typed and applied as a pair with the chosen provider, validated as bounded single-line text (TASK-14812 AC#5, AC#7).
- [ ] #9 A provider whose catalog is loading, empty or unavailable shows an explicit row saying so, never silently no rows (TASK-14812 AC#4).
- [ ] #10 Readiness is resolved at most once per provider each time the switcher opens, off the UI thread wherever it could block, through a resolver the screen injects. The switcher widget calls no provider service itself (ADR-011).
- [ ] #11 With no test evidence, a row reads 'Ready · not tested' or 'Not ready · <reason>'; no row claims verified or reachable.
- [ ] #12 Legacy alias providers are hidden unless configured or current, and are then labelled as legacy aliases (ADR-066).
- [ ] #13 custom-ep endpoints appear under their display names, and the built-in custom and custom_2 slots stay listable (ADR-146).
- [ ] #14 Enter on a NEEDS SETUP row never applies it. It closes the switcher and opens Settings ▸ Providers & Models with that row's provider selected and its credential or endpoint field focused (ADR-012).
- [ ] #15 The switcher has no credential input (D4).
- [ ] #16 At 211x44 and 235x52 the switcher is 120 columns wide, set by a width token in css/core/_variables.tcss, with auto height up to 80% of the screen and its key rows always visible.
- [ ] #17 Focus lands in Find when the switcher opens.
- [ ] #18 Group headers measure at least 4.5:1 against the switcher background instead of the disabled dim.
- [ ] #19 The highlighted row carries a glyph plus a background shift of at least 3:1.
- [ ] #20 The provider Select, the 2x2 button grids, the Defaults… subview, the 'Conversation settings' title and the context and compaction block are gone.
- [ ] #21 The ConsoleModelPopover class and the ids console-popover-apply, -temperature, -streaming, -save-model-default and -make-new-chat-default remain.
- [ ] #22 Tests that pin the removed form are deliberately rewritten, not deleted without replacement coverage: the wide-tier tests in Tests/UI/test_console_model_popover_geometry.py and its entry in Tests/UI/modal_wide_tier_registry.py:76; the provider-Select tests in Tests/UI/test_console_model_popover_no_provider.py, Tests/UI/test_console_rail_sections.py and Tests/UI/test_console_model_apply_chips.py:150; the popover-picker half of Tests/ProductionApp/test_provider_selection_ownership.py:200; test_model_popover_provider_options_include_registry_entries (Tests/UI/test_console_model_popover_registry_options.py:10); and the context and compaction pins in Tests/UI/test_console_context_controls.py, Tests/UI/test_console_popover_context_window.py and Tests/UI/test_console_provider_apply_defaults_flow.py.
- [ ] #23 Live captures at 211x44 and 235x52, taken with a scratch profile, are attached to the task notes.

### References

- tldw_chatbook/Widgets/Console/console_model_popover.py
- tldw_chatbook/UI/Console_Modules/model_switcher.py
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Screens/provider_model_resolution.py
- tldw_chatbook/Chat/provider_catalog.py
- tldw_chatbook/css/core/_variables.tcss
- Tests/UI/test_console_model_popover_geometry.py
- Tests/UI/modal_wide_tier_registry.py
- Tests/ProductionApp/test_provider_selection_ownership.py
- backlog/decisions/011-chatbook-workbench-ui-system.md
- backlog/decisions/012-provider-credential-settings-boundary.md
- backlog/decisions/066-local-provider-thinking-controls.md
- backlog/decisions/146-console-custom-endpoint-registry.md
- backlog/decisions/020-automatic-model-catalog-refresh.md
- backlog/tasks/task-194 - console_model_popover-uses-the-shared-provider-display-name-catalog.md
- backlog/tasks/task-14812 - Unify-Console-model-selection-into-a-searchable-picker.md

## Task 5: Add the Switch model value row, its Source words and commit keys (TASK-33004.5)

Task file: backlog/tasks/task-33004.5 - Add-the-Switch-model-value-row-its-Source-words-and-commit-keys.md
Depends on: TASK-33001, TASK-33002, TASK-33003, TASK-33004.1, TASK-33004.4

### Why

In today's popover (C6), Temperature sits at the fold, Max tokens is read-only (console_model_popover.py:530-536), and Streaming is a toggle Button (:513-517). The value row puts the three values people change on one line under the list. Those three are exactly the quick mask: the ADR-095 amendment of 2026-09-26 makes the quick surface's editable fields and its default mask the same set, and D3 added only max_tokens. So Thinking stays in Chat settings (spec §8 deliberate change 3; mockup (a) leaves it out).

The value row relies on existing machinery:
- Pressing Tab rebases the draft through rebase_console_settings_draft (Chat/console_chat_controller.py:12946), so the values shown are the highlighted pair's effective values.
- A→B→A edits come back through remember_model_draft (Chat/console_settings_apply.py:202).
- Apply stays the ADR-095 transaction (_commit_console_settings_submission_live, chat_screen.py:2861, wired at :5546).
- Ctrl+O replaces 'Full settings…' and carries a ConsoleSettingsTransfer (console_settings_apply.py:171).
- Field support comes from phase 1's single supported-field function, and labels from phase 2's field table.

This is the first surface that prints Source words, so it introduces the one resolver later editors reuse. Provenance today records only inherited/explicit/carried (ConsoleSettingsFieldProvenance, console_settings_apply.py:64-70). The Console parameter stack has more layers than the six words (ADR-147: model default, Console per-provider saved defaults, custom-endpoint params, chat_defaults, provider scalars, built-in fallbacks), and every layer must map to one word without being dropped.

Save as model default has no key because Ctrl+S is banned (ADR-031 rule 2). Streaming is On/Off at chat scope (ADR-095:79). A Select posts Changed on mount, so an unguarded Streaming Select would invent an edit.

### Acceptance criteria

- [ ] #1 The value row shows exactly Temperature, Max tokens and Streaming for the highlighted pair; Thinking and every other field stay in Chat settings.
- [ ] #2 Each value is one row tall and shows that pair's effective value with a Source word.
- [ ] #3 Source words come from one shared resolver that maps every layer of the Console parameter stack (ADR-147: edited draft, this chat, model default, Console per-provider default, custom-endpoint params, chat_defaults, provider scalars, built-in fallback) to exactly one of 'edited *', 'this chat', 'model default', 'Console Behavior', 'provider' or 'built-in'. A unit test covers each layer.
- [ ] #4 Tab from a list row rebases the draft to that pair and focuses Temperature with its value selected.
- [ ] #5 Edits made for pair A come back when the user returns to A in the same session (A→B→A).
- [ ] #6 Streaming is a single Select offering On and Off at this scope (ADR-095:79).
- [ ] #7 Opening and closing the switcher without changes leaves no edited state; the Select's mount-time Changed echo creates no edit.
- [ ] #8 Enter applies to this chat only, through the unchanged live-commit path (APPLY_TO_CHAT, QUICK_POPOVER surface), and writes no configuration.
- [ ] #9 After Enter the switcher closes, focus returns to the composer, and the status chips show the new pair.
- [ ] #10 [Save as model default] and Ctrl+N (default for new chats) save exactly Temperature, Max tokens and Streaming, and the button copy names those fields.
- [ ] #11 An integration test covers Ctrl+N through the real default-mutation owner against a scratch config file. It asserts that chat_defaults provider and model and the model profile are written, and that sibling profiles are preserved.
- [ ] #12 Ctrl+O opens Chat settings carrying the highlighted pair and its unapplied edits, without applying or discarding them.
- [ ] #13 Esc with no edits closes with no change.
- [ ] #14 Esc with edits shows 'Enter apply · d discard · Esc keep editing' and discards nothing until the user chooses.
- [ ] #15 The key rows print every accelerator the switcher binds (Enter, Tab, Ctrl+N, Ctrl+O, Esc) plus 'Applies to: this chat only'.
- [ ] #16 Every printed key works while Find has focus.
- [ ] #17 No binding from ADR-031 rule 2 (Ctrl+C, V, X, S, D, Z, A, R or W) exists on the switcher.
- [ ] #18 Pilot tests driven by real key presses from the composer pin Alex's path at 14 keys, the A/B toggle at 2, a model-only change at 5, and a model change saved as the model default at 16 or fewer.
- [ ] #19 A live capture of Alex's path at 211x44 is attached to the task notes.

### References

- tldw_chatbook/Widgets/Console/console_model_popover.py
- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/Chat/console_settings_apply.py
- tldw_chatbook/Chat/console_settings_defaults.py
- tldw_chatbook/Chat/console_session_settings.py
- tldw_chatbook/UI/Screens/chat_screen.py
- backlog/decisions/095-conversation-owned-console-generation-settings.md
- backlog/decisions/147-agent-provider-routing.md
- backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md

## Task 6: Add pick-only mode and mark the committed model in ModelSearchPicker (TASK-33004.6)

Task file: backlog/tasks/task-33004.6 - Add-pick-only-mode-and-mark-the-committed-model-in-ModelSearchPicker.md
Depends on: TASK-33004.4

### Why

C7(b) does not affect only the switcher. The shared ModelSearchPicker (Widgets/model_search_picker.py) renders options as plain ids (:643-644, :697-705), and Down highlights the first enabled option instead of the committed one (:866-875). The Chat settings modal still chooses its model through it (Widgets/Console/console_settings_modal.py:1794), and it still makes the user choose a provider first (ConsoleProviderPicker at console_settings_modal.py:1725; _rebase_to(provider, None) at :5372).

The synthesis ships a pick-only mode of the switcher in this phase. The pair list, its tests and its result contract then live in one widget, and Chat settings can pick provider·model pairs instead of a provider alone. This phase does not wire the modal to the new mode, because the modal has zero line headroom. The picker's mark is a widget-level fix: every host that keeps ModelSearchPicker benefits from it.

### Acceptance criteria

- [ ] #1 Switch model can open in a pick-only mode. It lists the same pairs, hides the value row and default actions, prints 'Enter picks · Esc cancel', and returns the chosen pair to its opener without applying anything or writing configuration.
- [ ] #2 In pick-only mode, NEEDS SETUP rows cannot be picked, and Esc returns nothing.
- [ ] #3 Wherever ModelSearchPicker is used, it marks the committed model in its results with the text '● CURRENT', not colour alone, and Down highlights that model first when it is present.
- [ ] #4 Tests in Tests/Widgets/test_model_search_picker.py pin the mark and the highlight, and a pilot test pins the pick-only result contract.

### References

- tldw_chatbook/Widgets/model_search_picker.py
- tldw_chatbook/Widgets/Console/console_model_popover.py
- tldw_chatbook/Widgets/Console/console_settings_modal.py
- tldw_chatbook/Widgets/Console/console_provider_picker.py
- Tests/Widgets/test_model_search_picker.py

## Task 7: Wire /model [query], Ctrl+O, the rail Change action and a rail Streaming row (TASK-33004.7)

Task file: backlog/tasks/task-33004.7 - Wire-model-query-Ctrl-O-the-rail-Change-action-and-a-rail-Streaming-row.md
Depends on: TASK-33004.4

### Why

This subtask absorbs task-338.

/model is registered with an empty argument hint (Chat/console_command_grammar.py:110). _console_command_run_action calls the mapped action (UI/Screens/chat_screen.py:19211) with no argument (:19362-19383), so '/model son' ignores 'son'.

The rail Model section (UI/Console_Modules/left_rail.py:2263-2410) shows Temperature and Max tokens but not Streaming. Commit 0c26a8408 dropped it (task-338). The summary state carries structured temperature and max_tokens values but no structured streaming value (Chat/console_session_settings.py:763-783). The section's updater (_apply_console_settings_summary_state, chat_screen.py:9384) already writes only ids that are composed: TASK-32811.7's fix is on dev even though that task still reads In Progress.

The rail's 'Configure' action (left_rail.py:2397-2403, routed at chat_screen.py:24608 and :25055) opens the full modal. The synthesis renames it 'Change  Alt+M' so the rail teaches the switcher key. That removes the rail's direct entry to Chat settings, and the spec gives Chat settings its own Console key, Ctrl+O. No Ctrl+O binding exists anywhere in the app today, and the spec asks for a live check that the terminal delivers it.

The palette entry 'Console: Change model…' (UI/console_command_provider.py:42-46) still describes a provider and temperature form.

Constraints: ADR-083:20-22 caps the Model section's natural height at 15 rows, and chat_screen.py has 32 lines of headroom.

### Acceptance criteria

- [ ] #1 /model with no argument opens Switch model. /model <query> opens it with the query in Find and the best match highlighted. Nothing applies until Enter, and /help lists /model [query].
- [ ] #2 The rail Model section shows a Streaming row (On or Off). It updates whenever the chat's settings change through Apply, resume or a new chat, and a test pins the rendered value (task-338 AC#1 and AC#2).
- [ ] #3 The rail Model section's action reads 'Change  Alt+M' and opens Switch model. It keeps its button id, so the rail reconciliation tests (Tests/UI/test_console_rail_reconciliation.py, Tests/UI/test_console_model_section_dedup.py) stay valid.
- [ ] #4 The Model section stays within ADR-083's cap of 15 rows of natural height.
- [ ] #5 Ctrl+O from the Console opens Chat settings, and a live check at 211x44 confirms the terminal delivers Ctrl+O to the app.
- [ ] #6 The Console footer, F1 help, the palette entry and the Provider/Model chip help text name Switch model (Alt+M) and Chat settings (Ctrl+O), and every key they teach works (ADR-031 rule 4).
- [ ] #7 Query dispatch for /model lives under UI/Console_Modules/, not in chat_screen.py.
- [ ] #8 A 211x44 live capture of the rail and of '/model son' is attached to the task notes.

### References

- tldw_chatbook/Chat/console_command_grammar.py
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Console_Modules/left_rail.py
- tldw_chatbook/Chat/console_session_settings.py
- tldw_chatbook/UI/console_command_provider.py
- tldw_chatbook/Widgets/Console/console_status_chips.py
- backlog/decisions/083-console-edge-rails-and-workspace-tree-ownership.md
- backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md
- backlog/tasks/task-338 - Restore-Streaming-on-off-visibility-in-the-Console-rail-Model-section.md
- Docs/User_Guide/console.md

## Task 8: The Alt+M surface shows its fold hint only while content remains below (TASK-33004.8)

Task file: backlog/tasks/task-33004.8 - The-Alt-M-surface-shows-its-fold-hint-only-while-content-remains-below.md
Depends on: —

### Why

The Alt+M popover's '▼ more — scroll for conversation settings' hint (Widgets/Console/console_model_popover.py:630-636) stays visible after its body is scrolled to the bottom, at 211x44 and 235x52. origin/dev 89dd84943a does the same, so the defect predates Phase 2. It was found in the Phase 2 capture triage.

task-33003.4 fixes only the Chat settings modal's hint. task-33004.4 replaces the popover form with Switch model but does not say whether a fold hint survives, so neither closes this defect.

### Acceptance criteria

- [ ] #1 While content remains below the viewport, the surface Alt+M opens shows its fold hint. At the bottom the hint is hidden, and scrolling back up shows it again. A pilot test proves this with real scroll or key presses.
- [ ] #2 If Switch model carries no fold hint at all, a test pins that no stale hint renders at 211x44 and 235x52.

### References

- tldw_chatbook/Widgets/Console/console_model_popover.py
- qa/model-config-33002-captures/console-model-popover-fields-211x44.txt
