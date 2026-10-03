# Plan: Phase 7: Reorder Settings ▸ Providers & Models into Connect / Default model / Model defaults / Advanced (TASK-33007)

Spec: backlog/docs/spec-2026-09-26-model-config-redesign.md (binding authority; ADR-095 and ADR-012 amendments of 2026-09-26).
Evidence: qa/model-config-ux-review-2026-09-26/ (mockups-211x44.md, verified-claims.md).
Parent task: backlog/tasks/task-33007 - Phase-7-Reorder-Settings-Providers-Models-into-Connect-Default-model-Model-defaults-Advanc.md

## Phase goal (parent)

Phase 7 of the model-configuration redesign, "Switchboard with field truth" (qa/model-config-ux-review-2026-09-26/judge-synthesis.md §2(c) and §4 P7; spec §8). It ships as one PR.

Why: at 211x44 the Providers & Models card runs to about 115 rows, or 3.5 viewports. Its layout has six problems:
- Prompt-cache snapshots sit above Connect (settings_screen.py:16698-16734).
- The default model is a free-text Input at roughly Tab stop 23 (:16800-16807).
- Every generation default is inside a disclosure that starts collapsed (:17203-17207, state at :3005).
- The card ends in catalog and config-key prose (:17410-17438).
- Nothing on it says who a save reaches.
- It nests frames: the card is a bordered .settings-focus-card (:16694, css/features/_settings.tcss:805) inside the bordered detail pane, and the refresh group adds a third border (:691). Its Select rows are 3 rows tall while its Input rows are 1 (_settings.tcss:465-470; the inversion in verified C5).

Verified finding C4: a discovered model can only be appended to the saved list. Save selected (:17103-17106 → _append_saved_discovered_models :14649-14668) never replaces a default that is already set, because _model_to_activate_after_save keeps a non-empty field by design (:14775-14791, TASK-369).

Verified finding C1(a): no copy on the card gives the scope of a save. Phase 2 fixes the save messages at :30215/:30242; this phase adds the Applies-to row. Owner decision D1 (shipped in phase 1) makes an untouched open chat converge to new defaults, so the scope copy has to tell an untouched chat apart from one that holds work.

What the phase delivers is the judge's order:
1. Connect.
2. Default model for new chats: a ModelSearchPicker with discovery merged in.
3. Model defaults, expanded, using the shared row grammar: label, one-row control, Source word, help.
4. Advanced, as one-row disclosures.

The card moves into UI/Settings_Modules/, the home DESIGN.md:359-364 names for Settings regions. Console Behavior's global fallbacks adopt the same rows and one streaming control form.

Constraints:
- ADR-002:10-12 and ADR-020:52 keep Discover / Save selected / Clear unchanged. The saved list therefore moves under Advanced instead of disappearing.
- ADR-033 keeps three honestly labelled commit models and the State badge (settings_screen.py:9408-9419).
- ADR-066 allows legacy aliases to be hidden but not deleted.
- ADR-012 and owner decision D4 keep all credential entry here.
- ADR-150/161 allow geometry only in css/core/_variables.tcss (Tests/UI/test_component_pattern_governance.py:266-289).
- ADR-097 ratchets never rise.

Absorbs TASK-31202: a settings_screen.py size-ratchet row at its measured post-phase size. The phase also delivers the Providers & Models slice of task-1378, which stays open for the rest of that split.

Dependencies:
- Phase 1: the single supported-field projection and D1 convergence.
- Phase 2: the field table, the State badge count and scoped save copy.
- Phase 3: one-row control and disclosure tokens, and contrast.
- Phase 4: the Source-word resolver, ModelSearchPicker's current-model mark, and the Alt+M switcher the scope copy points to.
- Phase 5: shared readiness evidence and the 't' key check (D2).

Baseline reds: task-15512 lists Settings provider-default contract tests that are already red on dev. Compare failing-test names against dev, not counts.

### Parent acceptance criteria

- [ ] #1 At 211x44 the Providers & Models card reads top to bottom Connect, Default model for new chats, Model defaults, Advanced. With every Advanced disclosure closed, the whole card is visible in the detail pane without scrolling (today it is about 115 rows).
- [ ] #2 For a cloud provider, the default Model control is reached in at most 5 Tab presses from the card's first control (today it is about the 23rd stop).
- [ ] #3 A discovered model can be made the default for new chats with one selection even when a default model is already set (closes C4), and doing so does not append it to the saved model list.
- [ ] #4 The card and the Inspector both say who a save reaches: new chats; an untouched open chat, which converges (D1); and an open chat with messages or edits, which keeps its own settings and can switch with Alt+M.
- [ ] #5 For a provider that does not accept a sampler, Settings neither shows nor saves that field and names it as hidden (Settings side of C8(1)).
- [ ] #6 Streaming uses one Select family everywhere in Settings: Inherit/On/Off per model and On/Off for the global fallback.
- [ ] #7 Every Input and Select row in Providers & Models and Console Behavior renders one row tall at 211x44 (the Settings select-row inversion ends on these cards).
- [ ] #8 Inside the detail pane the card draws no frame of its own: the pane border is the only frame, and each section starts with a one-row header.
- [ ] #9 Discover, Save selected and Clear behave as before under Advanced (ADR-002, ADR-020).
- [ ] #10 Legacy provider aliases stay selectable and are listed last (ADR-066).
- [ ] #11 The State badge and each control's commit model are unchanged and labelled (ADR-033).
- [ ] #12 No raw numeric dimension appears outside css/core/_variables.tcss, and there are no new Python style writes.
- [ ] #13 The boot CSS bytes, ui-ready module census and screen pre-import payload ratchets are not raised (ADR-097).
- [ ] #14 Keyboard-only live captures are attached to the PR at 211x44 and 235x52, using the real stylesheet and a scratch TLDW_CONFIG_PATH. They cover the card at rest, the model picker open, one Advanced disclosure open, and Console Behavior's fallback section.
- [ ] #15 Every existing test this phase rewrites on purpose is named in the PR description with the reason. Failing Settings test names match dev's baseline reds (task-15512), and no new test fails.
- [ ] #16 Docs/User_Guide pages updated: settings.md Providers & Models and Console Behavior sections (content only; verification is recorded in the task notes, never as a "Verified against" paragraph, per CLAUDE.md).
- [ ] #17 ./scripts/preflight.sh passes.

## Global Constraints

- Work ONLY in this worktree: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/model-config-p7. Start EVERY shell command with `cd <that path> &&`. Never touch the main checkout (another session's uncommitted work).
- Python: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python (the worktree has no venv); run pytest FROM the worktree cwd. Tests/Architecture needs -p no:xdist.
- NEVER write ~/.config/tldw_cli or ~/.local/share/tldw_cli. Never bypass test isolation (private_profile_test, TLDW_TEST_PRIVATE_PROFILE_NODE, HOME/XDG/TLDW_CONFIG_PATH). Live runs: scratch TLDW_CONFIG_PATH with a unique users_name, FULL SCREEN 211x44 (primary) and 235x52.
- ADR-126 RecoveryRequired in a clean worktree is environmental: compare failure-NAME sets against origin/dev.
- Size ratchets never rise (ADR-097); console_settings_modal.py net lines <= 0 against its current row.
- Geometry only through tokens in css/core/_variables.tcss (ADR-150/161); rebuild the CSS bundle with the repo script.
- ADR-031: never bind Ctrl+C/V/X/S/D/Z/A/R/W; footer hints must match working bindings.
- TDD; real-implementation tests for config/provider surfaces; rewrite pinned tests on purpose and name them.
- Commit per task `fix|feat(model-config): <summary> (TASK-33007.N)` + `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never push; NEVER merge origin/dev into the branch.
- Tick ACs, set Done, add Implementation Notes in each subtask file. Update Docs/User_Guide CONTENT; never add "Verified against" paragraphs (record verification in task notes).
- Lessons (backlog/docs/lessons-*.md): insert new entries MID-FILE near a related section, never appended at the end of the file (dev appends there constantly and every rebase conflicts).
- Evidence: commit captures (.txt/.ansi.txt) under qa/, but NEVER commit one-off driver or probe scripts (.sh/.py with machine-specific paths); describe the capture procedure in a short qa README instead. Captures are reviewed by AI reviewers, so they must not contradict the PR's claims.
- Scratch hygiene: every scratch directory carries a unique name with your task and sha (e.g. $SP/p<N>-t<M>-base-<sha>); never generic names (base, head, tree, cfgroot, devtree, neg); never rm -rf a directory you did not create; remove your own trees when done (git worktree remove for worktrees).
- Boot census: a NEW module only user actions need is imported inside the function that uses it; measure Tests/Performance/test_ui_ready_module_census.py with PYTHONPATH=<worktree> (the venv editable install otherwise measures the MAIN checkout); the limit must not rise.
- Before DONE: covering tests + `PYTHON=<venv> ./scripts/preflight.sh` (rc 0; never piped through tail).

## Lessons carried from Phases 4-6 (binding)

- The venv editable install points at the MAIN checkout: compare failure-NAME sets against a clean origin/dev worktree (unique scratch name, removed after); strip timestamps from FAILED names before diffing; a -q log can hold NO failure text, so count anything you need from --junitxml.
- Qodo flags EVERY new public function/method/fixture without Google-style Args:/Returns: (Yields: for generators) and every repeated literal that already has a constant; write both up front.
- No two surfaces may contradict each other at the same moment (readiness words, Key rows, Overview status): every readiness word comes from the shared vocabulary module and the shared connection-evidence owner (Phase 5).
- Owner rulings in force: only a 401 reads "key rejected" and blocks (a 403 from any model listing is "model listing unavailable", non-blocking, never verified); "verified" applies only to the model actually tested; untouched shipped local defaults that refuse sit quietly under NOT RUNNING; every closed disclosure title stays ONE row (names hidden fields only when they fit, otherwise a count, names inside the opened disclosure).
- Tests/real_profile_guard.py now refuses any test write to the real ~/.config/tldw_cli or ~/.local/share/tldw_cli; never work around it - point tests at the sandbox.
- Census/timing tests (storage units, keystroke work) flake on dev; run serially and compare with dev before acting.
- Never run_worker(exclusive=True) without group=; gated UI pilot tests carry bootstrap_profile in the file itself; census-gated UI files go in scripts/ui_pr_gate_census.txt with its MINIMUM_FILES floor raised.

## Task 1: Move the Providers & Models card into a Settings region module without changing behaviour (TASK-33007.1)

Task file: backlog/tasks/task-33007.1 - Move-the-Providers-Models-card-into-a-Settings-region-module-without-changing-behaviour.md
Depends on: —

### Why

settings_screen.py is 32,167 lines and has no size-ratchet row. Tests/Architecture/test_module_size_ratchet.py:19-23 leaves it out pending task-1378 and task-31202.

The card this phase rebuilds is composed inline in _render_provider_detail (settings_screen.py:16660-17443) and _render_custom_endpoints_section (:17470). DESIGN.md's One Home Rule (DESIGN.md:359-364) names UI/Settings_Modules/ as the home for Settings region widgets; that package does not exist yet.

Moving the composition first, with no behaviour change, lets the reorder subtasks land in a small module instead of growing the god file. This is the Providers & Models slice of task-1378. The card's handlers may stay on the screen and keep receiving the region's bubbled events; moving them is task-1378's remaining scope.

Tests depend on the card's ids: #settings-model-value alone is queried by 14 test files, so every id has to survive.

settings_screen.py is not resident at ui-ready (Tests/Performance/boot_budget_snapshots/ui_ready_modules.txt). It is in the screen pre-import payload (preimport_payload.json), which is budgeted at Tests/Performance/test_screen_preimport_payload_budget.py:93-100. A new package must stay inside both budgets.

### Acceptance criteria

- [ ] #1 The Providers & Models card's composition lives in a region module under UI/Settings_Modules/, and settings_screen.py no longer composes it inline.
- [ ] #2 Every widget id the card rendered before still renders after the move, under the same classes.
- [ ] #3 With only this change applied, the Settings test suites pass without test edits, and the failing test names are identical to dev's baseline reds.
- [ ] #4 settings_screen.py measures fewer lines than at c4225b5d38 (32,167).
- [ ] #5 The ui-ready module census (1,031, Tests/Performance/test_ui_ready_module_census.py:150) and the screen pre-import payload budgets (test_screen_preimport_payload_budget.py:93-100) are not raised.
- [ ] #6 A live capture at 211x44 with the real stylesheet shows the card rendering the same before and after the move.

### References

- tldw_chatbook/UI/Screens/settings_screen.py
- DESIGN.md
- Tests/Architecture/test_module_size_ratchet.py
- Tests/Performance/test_ui_ready_module_census.py
- Tests/Performance/test_screen_preimport_payload_budget.py
- backlog/tasks/task-1378 - Split-settings_screen.py-into-maintainable-modules.md

## Task 2: Rebuild Connect as one-row provider, key, endpoint and key-check rows (TASK-33007.2)

Task file: backlog/tasks/task-33007.2 - Rebuild-Connect-as-one-row-provider-key-endpoint-and-key-check-rows.md
Depends on: TASK-33007.1, TASK-33003, TASK-33005

### Why

Connect today has four problems:
- The provider list is a searchable OptionList fixed at 6 rows, and its focus border clips the first character of every option, so Anthropic reads 'nthropic' (css/features/_settings.tcss:458-463; composed at settings_screen.py:16743-16765).
- Credentials are a separate section (:16865-16904).
- Test Provider is a button with a guidance line and a multi-line result (:16925-16943).
- A Provider readiness block repeats readiness, provider and model source, endpoint and key status (:17006-17031).

Providers are grouped Cloud / Local / Custom & legacy aliases (settings_provider_view_model.py:142-146, built at :190). The few configured providers are therefore mixed in among 27 or more entries.

ADR-012 (012:29) requires the card to distinguish a key saved in config, a key from an env var and a missing key. Entry must stay masked, the saved key must stay clearable, and the env var must be described as the safer path. D4 keeps all key entry here. Phase 5 supplies the shared readiness vocabulary and the non-generating key check behind 't' (D2); this subtask gives it one row. The labelled result detail (phase 2) needs a home that does not push Default model down.

Conditional rows must survive under their current conditions:
- Manual provider (:16783-16798)
- QwenCloud API mode and its guidance (:16818-16864)
- Hosted-provider guidance (:16902-16914)
- The OpenAI restored-connection review (:16916-16921)
- The navigation-conflict and return-continuation blocks (:16955-17005)
- The vLLM recovery gate that disables the card (:16696)

Tests/UI/test_settings_provider_view_model.py:121 pins today's group order.

### Acceptance criteria

- [ ] #1 Provider is one row and one Tab stop.
- [ ] #2 The provider list shows providers that already have a credential or a saved endpoint first, then the rest, with legacy aliases last; aliases stay selectable and labelled as legacy (task-180, ADR-066).
- [ ] #3 Typing in the provider control filters by display name or id, and the focus cue hides no option's first character at 211x44.
- [ ] #4 The API key row is masked and says in words whether the key is saved in config, comes from an env var, or is missing (ADR-012:29). It still offers clearing the saved key.
- [ ] #5 Env var and Endpoint are one row each, each with a Source word and a one-line help.
- [ ] #6 Connect ends in a single Key check row. It shows this provider's latest phase-5 verdict (for example 'Ready · verified HH:MM' or 'Not ready · <reason>') and a visible test action labelled with its key t.
- [ ] #7 The labelled result detail appears in the Inspector's Key block, so a test result never adds rows above Default model.
- [ ] #8 The separate Provider readiness block is gone. Each fact it showed (readiness, provider source, model source, endpoint, key source) is still on screen as a Source word, in the Key check row, or in the Inspector.
- [ ] #9 The following still appear under the same conditions as before: Manual provider, QwenCloud API mode, hosted guidance, the OpenAI restored-connection review, the navigation-conflict and return-continuation blocks, and the vLLM recovery gate.
- [ ] #10 test_provider_picker_grouping_is_stable_and_empty_search_lists_catalog (Tests/UI/test_settings_provider_view_model.py:121) is rewritten on purpose for the configured-first order. Any test that queries a removed readiness id is rewritten on purpose and named in the PR.
- [ ] #11 Keyboard-only live captures at 211x44 show Connect for one cloud provider and one local provider.

### References

- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/UI/Screens/settings_provider_view_model.py
- tldw_chatbook/css/features/_settings.tcss
- Tests/UI/test_settings_provider_view_model.py
- backlog/decisions/012-provider-credential-settings-boundary.md
- backlog/decisions/066-local-provider-thinking-controls.md

## Task 3: Make the default model a searchable picker with discovered models merged in (TASK-33007.3)

Task file: backlog/tasks/task-33007.3 - Make-the-default-model-a-searchable-picker-with-discovered-models-merged-in.md
Depends on: TASK-33007.1, TASK-33003, TASK-33004.6

### Why

The default Model field is a free-text Input with a SuggestFromList typeahead of discovered ids (settings_screen.py:16800-16807; _model_field_suggester :14748-14763). That ghost suggestion measured 2.8:1 and is accepted only with an undiscoverable → key (report persona Sam).

Verified finding C4: selecting or saving a discovered model never replaces a default that is already set. Ticking a discovered row only records the selection (handler :28984-28997), Save selected only appends to the list (:17103-17106), and _model_to_activate_after_save keeps a non-empty field (:14775-14791, TASK-369).

Console already uses ModelSearchPicker (Widgets/model_search_picker.py:60). It has provenance groups (show_provenance, :131), a discovery overlay (set_discovered_models, :363-396) and, since phase 4, a '● CURRENT' mark and highlight for the committed model. The same widget gives Settings recognition over recall.

Constraints:
- ADR-002 (002:10-12) requires explicit persistence of discovered ids into [providers]. Choosing one as the default must not bypass that.
- task-14812 AC#6 and the TASK-364 comment at console_model_popover.py:1097-1099 say a model from the previous provider must never linger. A provider change already stages the new provider's default by writing #settings-model-value directly (settings_screen.py:28214-28227).
- 14 test files query #settings-model-value. A hidden adapter keeps that id, following the modal's legacy-adapter precedent (console_settings_modal.py:1809-1840).

Tests on today's typeahead:
- Tests/UI/test_settings_provider_test_draft.py:1193 (test_model_field_suggester_completes_discovered_ids) pins the SuggestFromList typeahead.
- :1180 pins the empty-field auto-fill, which stays.

### Acceptance criteria

- [ ] #1 The Default model row is a searchable picker scoped to the selected provider. It lists saved, catalog and discovered ids grouped by where they came from, marks the saved default in text, and highlights that default when the list opens.
- [ ] #2 The Default model row offers no ghost-text completion that only a hidden → key accepts; every suggestion is a visible, selectable row.
- [ ] #3 Choosing a discovered model stages it as the default for new chats even when a default is already set (closes C4). The category is marked dirty, and s saves it.
- [ ] #4 Choosing a discovered model does not append it to the saved model list. Save selected under Advanced stays the only path that persists ids to [providers] (ADR-002).
- [ ] #5 A model id that is in no list can still be entered, and it is validated as bounded single-line text (task-14812 AC#5, AC#7).
- [ ] #6 Changing provider re-scopes the picker and stages that provider's own default model. A model from the previous provider never remains (task-14812 AC#6).
- [ ] #7 #settings-model-value stays queryable and always holds the picker's value, and staging, dirty markers, save and revert all act on that one value.
- [ ] #8 No new Python style writes are added (test_component_pattern_governance.py:291).
- [ ] #9 Save selected still fills an empty default model with the first saved id, so test_settings_provider_test_draft.py:1180 stays green. test_model_field_suggester_completes_discovered_ids (:1193) is rewritten on purpose to prove that prefix search in the picker finds a discovered id.
- [ ] #10 A real-implementation integration test runs on a scratch TLDW_CONFIG_PATH with only the discovery transport stubbed. It discovers models, picks one that is not the current default, and saves. It then asserts that chat_defaults.model changed and the [providers] list did not.
- [ ] #11 A keyboard-only live capture at 211x44 shows the picker open for a provider that has discovered models.

### References

- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/Widgets/model_search_picker.py
- tldw_chatbook/Widgets/Console/console_settings_modal.py
- Tests/UI/test_settings_provider_test_draft.py
- backlog/decisions/002-openai-compatible-model-discovery.md
- backlog/tasks/task-14812 - Unify-Console-model-selection-into-a-searchable-picker.md

## Task 4: State what a Providers & Models save reaches, in the card and in the Inspector (TASK-33007.4)

Task file: backlog/tasks/task-33007.4 - State-what-a-Providers-Models-save-reaches-in-the-card-and-in-the-Inspector.md
Depends on: TASK-33007.3, TASK-33001, TASK-33002, TASK-33004

### Why

Nothing on the card says who a save affects.
- The Inspector for this category leads with 'Affects Console and provider-backed generation.' (settings_screen.py:22230-22240).
- The card ends in catalog, key-policy, manual-entry and sampling-route prose plus a raw endpoint-key row (:17410-17438). The review scored these as config keys in the UI.

The scope depends on the state of the open chat:
- ADR-095 (095:23-28) keeps open conversations that hold work on their own settings.
- Owner decision D1 (shipped in phase 1) makes an untouched open chat converge to newly saved defaults whatever its readiness. An untouched chat has no messages and no edited fields.
- The copy must therefore name the open chat and say which case applies to it. The per-chat way to switch is Switch model (Alt+M, phase 4).

Precedent: Settings already reads the active Console session through the app's console runtime (_current_reasoning_target, :6094-6108).

Two tests pin the prose inside the card:
- Tests/UI/test_settings_configuration_hub.py:3635 (test_settings_provider_category_lists_console_supported_catalog; it reads #settings-provider-catalog at :3658).
- :4366 (test_settings_provider_model_defaults_appear_before_reference_copy).

### Acceptance criteria

- [ ] #1 Directly under the Default model row, an Applies-to row says the choice applies to new chats. It names the open Console chat and the provider·model that chat will use: the new default if the chat is untouched (D1), or its own pair if it holds messages or edits. With no Console chat open, the row says so.
- [ ] #2 The Inspector for Providers & Models shows an Applies-to block covering new chats, untouched open chats, open chats with work (and how to switch there with Alt+M), and chats that switch to this model and so pick up its defaults.
- [ ] #3 The Inspector also shows a Next-new-chat block with the provider·model and core values a new chat gets from saved config. It notes when unsaved edits will apply only after save.
- [ ] #4 The Inspector's focused-field guide shows the phase-2 field-table help and range for the focused row. The config key appears only inside a closed disclosure.
- [ ] #5 The catalog, key-policy, manual-entry, sampling-route and endpoint-key prose rows no longer appear in the card. Their facts are available in the Inspector's config-key disclosure.
- [ ] #6 test_settings_provider_category_lists_console_supported_catalog and test_settings_provider_model_defaults_appear_before_reference_copy are rewritten on purpose to read those facts from their new home.
- [ ] #7 A real-implementation integration test uses a live Console store with the active chat on provider A, run once with the chat untouched and once with the chat holding a message. For each case, saving default B in Settings shows the correct Applies-to text, and a new Ctrl+T chat resolves to B.
- [ ] #8 A live capture at 211x44 shows the card and the Inspector while the open chat holds work.

### References

- tldw_chatbook/UI/Screens/settings_screen.py
- Tests/UI/test_settings_configuration_hub.py
- backlog/decisions/095-conversation-owned-console-generation-settings.md
- qa/model-config-ux-review-2026-09-26/verified-claims.md

## Task 5: Show model defaults expanded as one-row field-truth rows, naming unsupported fields instead of showing them (TASK-33007.5)

Task file: backlog/tasks/task-33007.5 - Show-model-defaults-expanded-as-one-row-field-truth-rows-naming-unsupported-fields-instead.md
Depends on: TASK-33007.1, TASK-33001, TASK-33002, TASK-33003, TASK-33004

### Why

Every per-model default sits in a Generation defaults disclosure that starts collapsed (settings_screen.py:17203-17207, state at :3005). Temperature is therefore about Tab stop 48, and the section costs rows even when closed. Empty rows show range placeholders instead of the effective inherited value, and the review found a placeholder indistinguishable from a value.

Settings also inverts the Console's heights: input rows are 1 row (.settings-input-row, css/features/_settings.tcss:369-374) while select rows are 3 (.settings-select-row, :465-470; verified C5(d)). The streaming Select and the enum rows on this card and on Console Behavior inherit the 3-row height.

Sampler support is not gated in Settings copy:
- Since phase 1, _model_profile_field_supported uses the single supported-field function, so unsupported rows read 'Unavailable for <provider>' but are still drawn.
- The one summary line names only Reasoning and Thinking (_provider_generation_support_copy, :12626-12647).
- Anthropic therefore still shows Min P, Seed, Presence and Frequency. PROVIDER_PARAM_MAP['anthropic'] omits them, and project_chat_handler_kwargs drops them silently (verified C8(1); Chat/Chat_Functions.py:281-300, :1431-1435).

Phase 1 supplies the single supported-field projection, phase 2 the field table (label, range, help), and phase 4 the Source-word resolver; this subtask renders them with the judge's row grammar. The default model stays the subject of this section (judge §2(c)), and no new 'subject' concept is added. ADR-095:74-79 fixes the blank-deletes-override semantics and the Inherit/On/Off streaming form.

Two tests pin today's shape in Tests/UI/test_settings_configuration_hub.py:
- :4386, test_settings_provider_connect_block_precedes_collapsed_generation_defaults, asserts the disclosure is collapsed.
- :4432, test_settings_provider_unavailable_fields_render_single_summary_line, pins the Reasoning/Thinking-only copy.

### Acceptance criteria

- [ ] #1 Model defaults sits directly after Default model and is open by default. Its heading names the provider·model it edits, and changing the default model changes both the heading and the values shown.
- [ ] #2 Core rows come first: Temperature, Max tokens, Streaming (Inherit/On/Off), and Reasoning, Thinking and Thinking budget only when the provider supports them. Each row is a label, a one-row control sized to its value, a Source word from the shared resolver, and one help line.
- [ ] #3 Every Select row on this card and on Console Behavior's fallbacks renders one row tall at 211x44; the 3-row select-row height no longer applies there.
- [ ] #4 An empty field shows the effective inherited value and where it comes from (for example 'inherits 1.0 · Console Behavior'). Placeholders only state a range or unit.
- [ ] #5 Help and Source text measure at least 4.5:1 against the detail pane in agentic_terminal and one light theme.
- [ ] #6 Top P, Top K, Min P, Seed, Presence and Frequency sit in one closed one-row Sampling disclosure whose title summarises their state. Fields the provider does not accept are hidden and named in that title; for Anthropic these are Min P, Seed, Presence and Frequency.
- [ ] #7 Blanking a field and saving still deletes only that override, so lower layers apply. A real-implementation integration test on a scratch config saves one edited field and one blanked field and checks that exactly those config keys changed.
- [ ] #8 test_settings_provider_connect_block_precedes_collapsed_generation_defaults and test_settings_provider_unavailable_fields_render_single_summary_line are rewritten on purpose: Connect still comes first, the defaults are now open, and the summary copy is new.
- [ ] #9 Live captures at 211x44 for Anthropic and for llama.cpp show Connect through Model defaults without scrolling.

### References

- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/css/features/_settings.tcss
- tldw_chatbook/Chat/Chat_Functions.py
- Tests/UI/test_settings_configuration_hub.py
- backlog/decisions/095-conversation-owned-console-generation-settings.md
- backlog/decisions/006-provider-aware-generation-settings.md

## Task 6: Fold rarely used provider controls into one-row Advanced disclosures and drop the nested frames (TASK-33007.6)

Task file: backlog/tasks/task-33007.6 - Fold-rarely-used-provider-controls-into-one-row-Advanced-disclosures-and-drop-the-nested-f.md
Depends on: TASK-33007.1, TASK-33003

### Why

Below Connect the card stacks four blocks:
- Context capacity (settings_screen.py:17033-17071).
- Model discovery with its SelectionList (:17072-17124).
- A bordered Automatic refresh group of 14 per-provider checkbox pairs (:17125-17199; .settings-instant-apply-group border at css/features/_settings.tcss:691).
- Custom endpoints (:17200).

Prompt-cache snapshots sits above Connect (:16698-16734). The card itself is a bordered .settings-focus-card (:16694; _settings.tcss:805) inside the bordered detail pane, so the card shows two or three frame levels where the spec allows one (spec §6).

Checkbox and discovered-list state is shown by colour only; task-32465 covers radios only. All per-provider refresh boxes also look active while 'Refresh on startup' is off.

Constraints:
- ADR-002 (002:10-12) and ADR-020 (020:52) keep Discover / Save selected / Clear as they are.
- ADR-033 rule 3 requires the instant-apply controls (catalog refresh, custom endpoints) to keep their 'applies immediately' label and stay out of the staged draft. Tests/UI/test_settings_save_commit_models.py:40 and :113 pin this.
- ADR-146 governs custom endpoints, and ADR-119 owns snapshots.
- .settings-focus-card is shared by other categories (15 uses), so only this card drops its frame.

Field search already expands a collapsed ancestor when it jumps to a field (settings_screen.py:9343-9348), so folding does not hide fields from '/'.

The judge's mockup also lists a 'Reasoning replay override' row here, but that control is not on this card. It is composed in Console Behavior (_render_console_behavior_card :17989; 'Override current Console model' at :18036) as part of the device-local reasoning-history policy (ADR-090:157). It stays there, and phase 2 renames it.

### Acceptance criteria

- [ ] #1 Advanced follows Model defaults as closed one-row disclosures in this order: Context window, Saved model list, Catalog refresh, Custom endpoints, Prompt-cache snapshots. Each title carries a live one-line summary, for example the context size and whether an override is set, or saved versus discovered counts.
- [ ] #2 Prompt-cache snapshots no longer appears above Connect, and its controls and next-launch copy are unchanged.
- [ ] #3 Discover models, Save selected and Clear behave exactly as before inside Saved model list (ADR-002, ADR-020).
- [ ] #4 Each discovered row says in text whether it is ticked and whether it is already saved.
- [ ] #5 Catalog refresh shows one row per provider with On/Off words. It carries the 'applies immediately' label and stays outside the staged draft.
- [ ] #6 While startup refresh is Off, Catalog refresh says the per-provider choices are not in effect.
- [ ] #7 test_model_catalog_controls_are_labeled_and_visually_separated (test_settings_save_commit_models.py:40) stays green.
- [ ] #8 The Providers & Models card draws no border of its own and no bordered group inside the detail pane; each section starts with a one-row header. Other categories keep their card frames.
- [ ] #9 A colour-stripped capture of the open Saved model list and Catalog refresh still shows every row's state.
- [ ] #10 '/' field search reaches fields inside closed disclosures and opens the one it lands in. Tests/UI/test_settings_search_index.py stays green.
- [ ] #11 Ids that tests query are kept (#settings-snapshot-controls, #settings-discovered-models-list, #settings-model-catalog-group, #settings-mc-auto-*, #settings-model-context-window). Any test changed on purpose is named in the PR.
- [ ] #12 Live captures show Saved model list open after a discovery at 211x44 and Catalog refresh open at 235x52.

### References

- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/css/features/_settings.tcss
- Tests/UI/test_settings_save_commit_models.py
- Tests/UI/test_settings_model_catalog_toggles.py
- Tests/UI/test_settings_model_discovery_journeys.py
- Tests/UI/test_settings_search_index.py
- Tests/UI/test_llamacpp_snapshot_settings.py
- backlog/decisions/002-openai-compatible-model-discovery.md
- backlog/decisions/020-automatic-model-catalog-refresh.md
- backlog/decisions/033-settings-commit-models-three-honestly-labeled.md
- backlog/decisions/119-llamacpp-prompt-cache-snapshot-ownership.md
- backlog/decisions/146-console-custom-endpoint-registry.md
- backlog/decisions/090-console-thinking-block-ownership-and-replay.md

## Task 7: Give Console Behavior's global fallbacks the shared rows and one streaming control (TASK-33007.7)

Task file: backlog/tasks/task-33007.7 - Give-Console-Behavior-s-global-fallbacks-the-shared-rows-and-one-streaming-control.md
Depends on: TASK-33002, TASK-33003, TASK-33007.5

### Why

Across the product, streaming is edited through four control types (report: Consistency 1/4). In Settings the two controls disagree: the model default is an Inherit/On/Off Select (settings_screen.py:17397-17408), while the global fallback is a Checkbox (:18586-18592). Console Behavior also prints the raw config line 'chat_defaults.streaming is canonical; enable_streaming is read as fallback only.' (:18704-18708).

ADR-095 (095:78-79) settles Settings on a Select for streaming. ADR-006 and ADR-052 keep Console Behavior as the owner of the global fallbacks (from :18562) and of the memory and compaction defaults; this subtask does not touch the latter.

Five tests in Tests/UI/test_settings_configuration_hub.py query #settings-console-default-streaming as a Checkbox:
- :5164, the Console Behavior focus-reveal test (at :5203)
- :6143, test_settings_console_behavior_renders_global_default_controls
- :6608, test_settings_console_behavior_saves_global_defaults
- :6854, test_settings_console_behavior_uses_batched_save_adapter
- :9029, test_settings_provider_streaming_and_enums_prevent_invalid_input

### Acceptance criteria

- [ ] #1 Global fallback defaults use the same row grammar and labels as Model defaults. Core rows come first (Temperature, Max tokens, Streaming), then reasoning and thinking, with Top P, Top K, Min P, Seed, Presence and Frequency in one closed Sampling disclosure.
- [ ] #2 Global streaming is a one-row On/Off Select from the same control family as the model default's Inherit/On/Off Select.
- [ ] #3 Saving global streaming writes chat_defaults.streaming exactly as the Checkbox did, and the legacy enable_streaming read fallback still works.
- [ ] #4 The raw 'chat_defaults.streaming is canonical…' line no longer appears in Settings; the fact is in the Inspector's config-key disclosure.
- [ ] #5 The five tests that query the streaming Checkbox are rewritten on purpose for the Select.
- [ ] #6 A real-implementation integration test on a scratch config sets global streaming to Off and saves. A new Ctrl+T chat then resolves streaming Off, and a model default left at Inherit shows 'inherits Off · Console Behavior'.
- [ ] #7 Memory and compaction defaults (ADR-052) render and save as before.
- [ ] #8 A live capture at 211x44 shows Console Behavior's fallback section.

### References

- tldw_chatbook/UI/Screens/settings_screen.py
- Tests/UI/test_settings_configuration_hub.py
- backlog/decisions/095-conversation-owned-console-generation-settings.md
- backlog/decisions/006-provider-aware-generation-settings.md
- backlog/decisions/052-console-conversation-memory-and-compaction-policy.md

## Task 8: Pin settings_screen.py's size-ratchet row at its post-phase size (TASK-33007.8)

Task file: backlog/tasks/task-33007.8 - Pin-settings_screen-py-s-size-ratchet-row-at-its-post-phase-size.md
Depends on: TASK-33007.1, TASK-33007.2, TASK-33007.3, TASK-33007.4, TASK-33007.5, TASK-33007.6, TASK-33007.7

### Why

task-31202: settings_screen.py has no size-ratchet row, so nothing stops it growing back once this phase moves the card out. Tests/Architecture/test_module_size_ratchet.py:19-23 deliberately left it out, pending task-1378 and task-31202, and names this file as the row's home.

The personas_screen.py row (:67) is the precedent for a screen in this file. The ratchet also fails when a budget is left more than 50 lines slack (test_budget_is_not_left_slack, :126; tolerance at :89), so the row must be set at the measured size. Pinning the row after the extraction and reorder banks the shrink. This subtask closes task-31202; task-1378 stays open for the rest of the split.

Absorbs TASK-31202.

### Acceptance criteria

- [ ] #1 Tests/Architecture/test_module_size_ratchet.py has a settings_screen.py row set to the file's measured line count at the PR head, below 32,167 (TASK-31202 AC#1).
- [ ] #2 The module docstring no longer says settings_screen.py is deliberately absent, and it names task-1378 as the remaining split.
- [ ] #3 Adding a dummy method to settings_screen.py makes test_module_does_not_grow_past_its_budget fail, and this mutation check is recorded in the PR (TASK-31202 AC#2).
- [ ] #4 test_budget_is_not_left_slack passes.

### References

- Tests/Architecture/test_module_size_ratchet.py
- tldw_chatbook/UI/Screens/settings_screen.py
- backlog/tasks/task-31202 - settings_screen.py-needs-a-size-ratchet-budget-row.md

## Task 9: The Settings provider control shows the chosen provider after a choice or Revert (TASK-33007.9)

Task file: backlog/tasks/task-33007.9 - The-Settings-provider-control-shows-the-chosen-provider-after-a-choice-or-Revert.md
Depends on: —

### Why

The Phase 2 capture triage reproduced this in Settings ▸ Providers & Models ▸ Provider. Type 'llama', Tab into the list and press Enter on a provider. The Provider control still holds the typed filter 'llama', and the filtered list stays open and framed. The same happens after r ▸ Discard changes. The list's focus frame covers the first cell of every row ('lama.cpp', 'ocal Llamafile'), and its right edge is clipped. At 211x44, origin/dev 89dd84943a behaves the same way, so the defect predates Phase 2.

task-33007.2 AC#3 fixes the lost first character. This rider covers what is left: the leftover filter text and the clipped frame.

### Acceptance criteria

- [ ] #1 After a provider is chosen, the Provider control shows the chosen provider's display name, not the typed filter.
- [ ] #2 After r ▸ Discard changes, the Provider control shows the saved provider and no filtered list is left open.
- [ ] #3 At 211x44 and 235x52, whenever the provider list is visible, its whole frame is drawn inside the card, right edge included.
- [ ] #4 A keyboard-only live capture at 211x44 shows the control after a choice and after Revert.

### References

- tldw_chatbook/UI/Screens/settings_screen.py
- qa/model-config-33002-captures/settings-provider-picker-after-select-211x44.txt
- qa/model-config-p2-2026-09-27/task-5/settings-legacy-alias-selected-211x44.txt
