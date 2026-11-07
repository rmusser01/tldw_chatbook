---
id: TASK-33007
title: 'Phase 7: Reorder Settings ▸ Providers & Models into Connect / Default model
  / Model defaults / Advanced'
status: Done
assignee: []
created_date: 2026-09-26 11:47
labels:
- model-config-redesign
- phase-7
- settings
- ux
- a11y
- css
dependencies:
- TASK-33001
- TASK-33002
- TASK-33003
- TASK-33004
- TASK-33005
references:
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/UI/Screens/settings_provider_view_model.py
- tldw_chatbook/Widgets/model_search_picker.py
- tldw_chatbook/css/features/_settings.tcss
- tldw_chatbook/css/core/_variables.tcss
- Tests/UI/test_settings_configuration_hub.py
- Tests/UI/test_settings_provider_test_draft.py
- Tests/Architecture/test_module_size_ratchet.py
- Docs/User_Guide/settings.md
- DESIGN.md
- backlog/decisions/002-openai-compatible-model-discovery.md
- backlog/decisions/020-automatic-model-catalog-refresh.md
- backlog/decisions/033-settings-commit-models-three-honestly-labeled.md
- backlog/decisions/095-conversation-owned-console-generation-settings.md
- backlog/decisions/012-provider-credential-settings-boundary.md
- backlog/decisions/066-local-provider-thinking-controls.md
- backlog/decisions/161-component-pattern-library.md
- backlog/decisions/097-boot-budget-ratchets.md
- qa/model-config-ux-review-2026-09-26/judge-synthesis.md
- qa/model-config-ux-review-2026-09-26/verified-claims.md
priority: high
updated_date: 2026-10-09 21:53
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
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
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 211x44 the Providers & Models card reads top to bottom Connect, Default model for new chats, Model defaults, Advanced. With every Advanced disclosure closed, the whole card is visible in the detail pane without scrolling (today it is about 115 rows).
- [x] #2 For a cloud provider, the default Model control is reached in at most 5 Tab presses from the card's first control (today it is about the 23rd stop).
- [x] #3 A discovered model can be made the default for new chats with one selection even when a default model is already set (closes C4), and doing so does not append it to the saved model list.
- [x] #4 The card and the Inspector both say who a save reaches: new chats; an untouched open chat, which converges (D1); and an open chat with messages or edits, which keeps its own settings and can switch with Alt+M.
- [x] #5 For a provider that does not accept a sampler, Settings neither shows nor saves that field and names it as hidden (Settings side of C8(1)).
- [x] #6 Streaming uses one Select family everywhere in Settings: Inherit/On/Off per model and On/Off for the global fallback.
- [x] #7 Every Input and Select row in Providers & Models and Console Behavior renders one row tall at 211x44 (the Settings select-row inversion ends on these cards).
- [x] #8 Inside the detail pane the card draws no frame of its own: the pane border is the only frame, and each section starts with a one-row header.
- [x] #9 Discover, Save selected and Clear behave as before under Advanced (ADR-002, ADR-020).
- [x] #10 Legacy provider aliases stay selectable and are listed last (ADR-066).
- [x] #11 The State badge and each control's commit model are unchanged and labelled (ADR-033).
- [x] #12 No raw numeric dimension appears outside css/core/_variables.tcss, and there are no new Python style writes.
- [x] #13 The boot CSS bytes, ui-ready module census and screen pre-import payload ratchets are not raised (ADR-097).
- [x] #14 Keyboard-only live captures are attached to the PR at 211x44 and 235x52, using the real stylesheet and a scratch TLDW_CONFIG_PATH. They cover the card at rest, the model picker open, one Advanced disclosure open, and Console Behavior's fallback section.
- [x] #15 Every existing test this phase rewrites on purpose is named in the PR description with the reason. Failing Settings test names match dev's baseline reds (task-15512), and no new test fails.
- [x] #16 Docs/User_Guide pages updated: settings.md Providers & Models and Console Behavior sections (content only; verification is recorded in the task notes, never as a "Verified against" paragraph, per CLAUDE.md).
- [x] #17 ./scripts/preflight.sh passes.
<!-- AC:END -->

## Implementation Notes

### Final review fix wave 1 (2026-10-06)

The whole-branch review (`.superpowers/sdd/plan-2026-10-03-model-config-p7/final-review.md`) raised 10 findings; all are fixed in code, none deferred.

- AC#2: no owner ruling is needed. The Chat-settings return actions (conflict, continuation, **Return without saving**) now compose after Default model's Applies-to row, so a pending return with an unsaved edit adds no stop between Provider and Model (pinned for Anthropic, OpenAI and QwenCloud in `test_model_stays_within_five_presses_while_a_return_is_pending`).
- AC#4: the Applies-to row describes an open chat with no settings snapshot instead of raising, and never calls a chat with messages "unused".
- AC#7: the three remaining tall Console Behavior Selects (reasoning history, reasoning replay override, rail layout scope) are compact one-row labelled rows; the one-row test now measures every Select on both cards.
- TASK-33007.2 AC#2/AC#4: Azure, Cloudflare and Databricks keys read **saved in config** / **from env var**, Clear works and the provider is Configured before the base URL is set (`provider_credential_source` in `provider_readiness.py`, used by Settings and `configured_provider_keys`).
- An emptied API key field is no edit; only Clear (or Ctrl+L) stages removal.
- Also: both missing test files added to the UI PR gate census (floor 142), the card module ruff-formatted with `ClassVar` BINDINGS, the narrow Sampling title clamps its name and drops its state first, the picker handlers are typed, and the saved-provider-return projection asserts the visible picker.

### Capture fixes: Providers & Models (2026-10-09)

The full-screen captures' review notes 1-8 (`qa/model-config-33007-captures/README.md`), each with a RED test first.

- Note 1: Azure, Cloudflare and Databricks ship no base URL, so a blank Endpoint reads **not set** with "required: your resource host / account URL / workspace host" and an "Enter your …" placeholder (`required_base_url_target` in `provider_readiness.py`), not built-in and "blank uses the provider default".
- Note 2: Provider and Model read **edited \*** only when their own value differs from the saved one (`_resolve_provider_model_for_settings` reads `draft.dirty_keys`; the draft pins both beside any edit).
- Note 3: "Review restored OpenAI connection" is a Connect row (Connection | Review | restored | help) shown only while `recovery_review.openai_reconnect_pending()` says a restored connection awaits review; it is read off the UI thread when the card mounts and on a switch to OpenAI, cached on the screen, and cleared when a review is recorded. Focused, the Inspector explains it. Rewritten on purpose: `_OPENAI_STOPS` in `Tests/UI/test_settings_connect_rows.py` is now the three Connect stops (the pinned profiles restored nothing), `_OPENAI_REVIEW_STOPS` pins the fourth stop while a review awaits, and `test_settings_openai_reconnect.py` asserts the row (and `openai_reconnect_pending`) instead of the button's display.
- Note 4: Applies to reads "new chats; open chat “Chat 1” is unused and will use OpenAI · o4-mini-2025-04-16." on one row at 211x44 and 235x52; "(Ctrl+T, temporary, workspace)" moved to the Inspector's New chats row. Textual wraps at a no-break space too, so `AppliesToLine` breaks a too-long row before the pair, never inside it. `test_settings_save_reach.py`'s copy pins are rewritten on purpose.
- Note 5: the Provider help counts instead of naming: "3 of 60 configured · listed first" (or "none of 60 configured yet"); `test_provider_picker_summary_*` and the connect-row pins are rewritten on purpose.
- Note 6: API key placeholders fit the 25-cell field: "Paste to replace", "Paste API key", "Subscription in use".
- Note 7: Sign in with has the row grammar: Source word (**built-in** / **config** / **edited \***) and a one-line help ("bills API credits through your key" / "bills your Claude plan, not API credits"); its long copy is the Inspector's guide for the field. `test_settings_anthropic_auth_source.py` asserts the row help, rewritten on purpose.
- Note 8: with a filter typed the Default model picker says "N found · Enter picks · Esc cancels", highlights the first match (or an id typed in full), and Enter picks it. The committed id echoed back is not a filter. `test_truncated_results_say_how_many_matched_and_that_typing_narrows` is rewritten on purpose for the count.
- Capture 04 follow-up: Temperature's help was cut to "…higher is more var…" at 211x44 (the Model defaults help column is 61 cells; Console Behavior's fallbacks, 55). The shared field table's Temperature, Top P, Seed, Thinking and Frequency penalty help lines are shorter, each 55 cells or fewer with the same meaning; `test_every_row_paints_its_whole_help` paints all fourteen rows (three providers, Sampling open) at 211x44 and 235x52.
- Note 1 follow-up: the focused Endpoint's Inspector guide still said "an http:// or https:// address when set" where the row reads "required: …". `endpoint_requirement()` in `providers_models_card.py` is now the one decision both use (local servers in `API_URL_PROVIDER_KEYS`, then `required_base_url_target()`), and the guide moved there as `endpoint_field_guide()`: "…address; required: your resource host" (Azure) or "…; required: the server's base URL" (llama.cpp, Ollama). `test_settings_provider_detail_shows_field_guidance_and_readable_draft_state` (Ollama) is rewritten on purpose; `settings_screen.py`'s ratchet row is lowered to 33,643 and `providers_models_card.py`'s re-pinned at 2,238.

### Capture review items 9-11 (p7cf-b, 2026-10-09)

- Item 9: Console Behavior draws one frame (the detail pane's): the wrapper, the card, the replay-override disclosure and the Permission summaries group lose their borders; every one-row Input and Select shares one 32-cell control column (the longest option, "Memory with latest exchange", fits whole); the replay rows read **Default replay** / **This model's replay**. The fallbacks' unset help became "not set · a provider's setting comes first" so it stays whole in the 45-cell help column at 211x44. Pinned in `Tests/UI/test_settings_console_behavior_grammar.py`.
- Item 10: the read-only "Composer behavior" / "Global fallback defaults" summary after the card is removed (it restated the card's rows and the Inspector's Override rules); the Inspector's closed "config key" disclosure gains a **Fallbacks** row listing `user_display_name` and all 14 generation fallbacks (`CONSOLE_FALLBACK_KEYS_FACT`); "Default chat display name" is now **Chat display name** (fits the 24-cell label).
- Item 11: Advanced ▸ Context window opens to one row -- field in Model defaults' 16-cell column, **Reset to detected** in the row and hidden while the window is unknown, a Source word (`detected` / `saved in config` / `edited *` / `not set`) and a one-line help; the wrapped status and capacity paragraphs are gone (`_context_window_row_copy`). Pinned in `Tests/UI/test_settings_advanced_disclosures.py`.
- Tests rewritten on purpose: `test_settings_configuration_hub.py::test_settings_console_behavior_renders_global_default_controls` (the display-name row now reads "Chat display name"), `test_settings_context_memory_controls.py::test_provider_context_window_reset_preserves_other_capabilities` (the Source word and help replace the "Configured override" sentence; the reset path now also asserts "edited *" / "detected 128,000"), and a comment in `test_settings_console_fallback_rows.py` (the control column is no longer 16 cells).
- `test_settings_console_storage_journeys.py`'s three instant-toggle journeys now wait out Textual's 0.2 s press effect (`-active`) before a second Enter: without the nested frames Settings settles faster, and the second press landed inside the window Textual drops (6 status-row cases failed at this head, passed at the base; confirmed by applying only the CSS change to the base).
- Self-review of item 9: without the frames, the unclassed help lines ("Global keeps one arrangement...") started one cell left of the section headers and detail rows; every prose Static on the card (and in its instant-apply group) now takes the headers' one-cell inset (`test_console_behavior_prose_starts_where_its_section_headers_do`).
- `Tests/UI/test_settings_console_behavior_grammar.py` joins the UI PR gate census (floor 142 -> 143); the size ratchet re-pins `settings_screen.py` at 33,635 and `providers_models_card.py` at 1,962.
- Captures `05b`, `06a`, `06b`, `06c` re-taken at both sizes; README review notes 9-11 marked fixed.

### Live-capture finding 12 (2026-10-09)

- **Ruling (12a), by the spec's one-vocabulary rule (§4 rule 3, §5 "each fact appears once"; §6 "a field shows its effective value with its source"):** a model's context window has one owner, `resolve_context_window` (saved override, then catalog and table, OpenRouter through its upstream, then the provider or application fallback). Settings ▸ Advanced reads it with its own config minus the override, so it now sees the provider fallbacks and the OpenRouter upstream it used to skip. Only a live server's own metadata stays Console-only: Settings runs no serving probe (D2). A fallback is a guess and is never shown as a known size: both surfaces call the window **unknown** and name the fallback as **assumed**. Settings' field stays blank, so an assumed size is never saved.
- 12a: Chat settings' MODEL row reads "context unknown" (was "~32k context"); Request estimate and the Console's context label end "(assumed; window unknown)" (was "(estimated; model unverified)"); the Context view reads "Model window  unknown, 32,000 assumed" (was "Model window (est.) … (application fallback)"); Settings' Advanced title reads "unknown, 32,000 assumed · enter the model's documented limit". The MODEL row leaves the size to Request estimate, so a long id keeps its width. Tests rewritten on purpose for the new words: `test_unknown_openai_model_uses_shared_unverified_api_fallback`, `test_unverified_model_capacity_is_labeled_unknown` (renamed from `…_as_estimated`).
- 12b: every Select in a Chat settings view shares one width: 13 in the Model view, where the field-row numbers take 13 too so the Source words line up, and 32 in the Context view. Before, they were 8, 12 and 13, and 12 to 32. `test_console_settings_fields_are_sized_by_value_type` is rewritten on purpose: a Select used to be exactly as wide as its own longest option, and now takes its view's width.
- 12c: Switch model (Alt+M) follows the same ruling: its dense context column read "~32k" for a fallback and now reads "?" (unknown; the column is 5 wide, so "unknown" would not fit), while a catalog size keeps its number (gpt-4o "128k", gpt-5.6-terra "?", pinned at 211x44 and 235x52 with the real resolver). The cost chip's tooltip, whose "safe input" was computed from the assumed window, ends its fullness line "(assumed; window unknown)" too. No other Console surface printed a "~" window: the remaining "~" marks are token-count estimates, and the Inspector's Input budget is the enforced ceiling. Tests rewritten on purpose: `test_context_copy_calls_a_fallback_unknown_and_shortens_sizes` (renamed from `testcontext_copy_marks_estimates_and_shortens_sizes`, "~32k" became "?"), and `test_switcher_lists_four_groups_of_one_line_pairs`, whose `"~200k" or "200k"` now asks the real resolver for claude-sonnet-4-5's size (the catalog's "200k", or "?" when the catalog cannot load).

### Integration of the capture-fix branches (2026-10-09)

- p7cf-b (items 9-11) and p7cf-c (item 12) cherry-picked onto the Providers & Models fixes (notes 1-8 and their follow-ups). Settings.md keeps every side: the new Connect, Default model and Model defaults copy, B's one-row Context window and C's assumed title. The UI PR gate floor stays 143, the true count. The size ratchet re-pins `settings_screen.py` at the integrated head's measured 33,645 and `providers_models_card.py` at 2,238 (B's 33,635 / 1,962 and C's 1,964 were measured on the older base).
- Follow-up (a): Context window's unknown row read **not set** with "required for Automatic conversation budgets", though under the item 12 ruling Automatic budgets use the assumed size. Its help now names that size in the title's words, "unknown, 32,000 assumed · budgets use it until set" (200,000 for an unlisted Anthropic model), from `ModelContextWindowState.assumed_tokens`; an emptied known window still reads "required …". `test_an_unknown_context_window_opens_to_one_row_and_no_reset` is rewritten on purpose (OpenAI and Anthropic).
- Follow-up (b): the dead `ConsoleContextControlState.request_row` (no caller or test in tldw_chatbook or Tests; it still said "estimated input; model unverified") is deleted.

### Review of #3057: evidence corrections (2026-10-11)

The independent review of the whole PR (R4, tests / docs / CI) found criteria whose tick, wording or notes did not agree with what shipped. Each is corrected where it was wrong; no behaviour changed for these.

- AC#1 ("without scrolling") was asserted for OpenAI · gpt-4.1 alone. `Tests/UI/test_settings_providers_models_card_fits_the_pane.py` now chooses every provider the picker lists, in turn on one mounted card at 211x44 with Advanced closed, and none needs scrolling.
- AC#2: with "Enter provider ID" opened, its row is a Tab stop the user opened, so Anthropic's chain to Model is six presses. Accepted by the review's ruling C10 (the budget of five is for the card at rest) and pinned by `test_an_opened_provider_id_row_is_a_sixth_stop_for_anthropic`.
- AC#14: the captures in `qa/model-config-33007-captures/` were taken at `0cd360a642` on 2026-10-09. The reviewed head `37f5e55b5b` is later and was rebased again, and this review's fixes change copy the captures show, so the PR description's "all taken on this head" did not hold. They are retaken after the fixes are integrated.
- Captures behind subtask criteria that the final set does not repeat: TASK-33007.3 AC#11 (the picker with discovered models) and TASK-33007.6 AC#9 and AC#12 (Saved model list and Catalog refresh open) rest on `qa/model-config-p7-2026-10-03/task-3/` and `task-6/`, taken on 2026-10-03 and 2026-10-04 and showing the card as it was then. They are retaken with the final set (the review's ruling C13); their task files say so.
- AC#15: the PR description's generated list of rewritten tests had been cut off after most headings (15 of 17 names missing). The full list, taken from the subtasks' notes, was handed to the controller for the PR description, with the TASK-33007.8 mutation check.
- Subtask wording corrected in their own files: TASK-33007.2 AC#9 (the OpenAI review row's condition changed on purpose), TASK-33007.5 AC#6 (Anthropic's hidden fields are counted in the title, not named), TASK-33007.6 AC#1 (six disclosures) and AC#10 (the search-index test was red before and after, on the same two tests), and the notes of TASK-33007.7 (AC#8) and TASK-33007.9 (AC#4, AC#5), which still said "open" under ticked criteria and now name the captures.
- There is no TASK-33007.10: the phase plan's Task 10 (Clear becomes a key on the API key field) is recorded under TASK-33007.2.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Rebased onto dev 2f37c1a478 (862 commits) on 2026-10-09. Dev's task-34384 Session summary on quit, added inside the card compose this phase moved, is ported into Advanced as a sixth one-row disclosure (0cd360a642). After the rebase: the 50 test files this branch changes show 3 head-only failures, all renamed-on-purpose tests that fail with ADR-126 RecoveryRequired exactly as the tests they replace; size ratchets fail the same 11 rows as dev; boot CSS, ui-ready census and pre-import payload ratchets pass (19 passed); preflight passes. Checkpoint review fixes: SettingsResultLine (no at-rest 'not saved' row), a blank row names its layer once, and a typed name prefix highlights the provider it starts. Captures re-taken on the rebased head (qa/model-config-33007-captures/).

**Spec amendment (2026-10-13, controller ruling on capture-fix B #15 (a)):** Chat settings is 176 columns wide (`$ds-chat-settings-width`), not spec §6 / mock (b)'s 150, at 211x44 and 235x52 (95% of the viewport below that); the owner uses the app full-screen only, so whole help lines win over the 150x22 mock. Height stays 22 rows.

**UI lane and TASK-34917 (2026-10-14):** this PR moves the UI Fast Lane from dev's six shards to ten, with 92 more census entries; its heaviest shard replays at 13.7 pytest min at most (~15.4 with setup), inside TASK-34917's 16.7-min rule. TASK-34917's AC#1-#3 were met before this merged, on 49 six-shard runs (its 2026-10-15 20-run measurement, which left out this branch's ten-shard runs); after the merge its shard check becomes "every shard ≤ 16.7 min over ten-shard runs" rather than `UI Fast Lane (3)` alone. The four extra runners (eleven to fifteen test runners per PR, sixteen with perf-guard, no `max-parallel`) bear on the open AC#4 of TASK-34917 and TASK-34916: after #3106's fan-out the required check's wall clock got worse at the tail (p90 66.8 to 95.2 min), and whether that needs `max-parallel` or fewer runners is still an owner decision there. Asked before re-arming on 2026-10-15, the owner chose to land this with fifteen runners and no `max-parallel`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
