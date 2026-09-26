---
id: TASK-33007
title: 'Phase 7: Reorder Settings ▸ Providers & Models into Connect / Default model / Model defaults / Advanced'
status: To Do
assignee: []
created_date: '2026-09-26 11:47'
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
  - 'tldw_chatbook/UI/Screens/settings_screen.py'
  - 'tldw_chatbook/UI/Screens/settings_provider_view_model.py'
  - 'tldw_chatbook/Widgets/model_search_picker.py'
  - 'tldw_chatbook/css/features/_settings.tcss'
  - 'tldw_chatbook/css/core/_variables.tcss'
  - 'Tests/UI/test_settings_configuration_hub.py'
  - 'Tests/UI/test_settings_provider_test_draft.py'
  - 'Tests/Architecture/test_module_size_ratchet.py'
  - 'Docs/User_Guide/settings.md'
  - 'DESIGN.md'
  - 'backlog/decisions/002-openai-compatible-model-discovery.md'
  - 'backlog/decisions/020-automatic-model-catalog-refresh.md'
  - 'backlog/decisions/033-settings-commit-models-three-honestly-labeled.md'
  - 'backlog/decisions/095-conversation-owned-console-generation-settings.md'
  - 'backlog/decisions/012-provider-credential-settings-boundary.md'
  - 'backlog/decisions/066-local-provider-thinking-controls.md'
  - 'backlog/decisions/161-component-pattern-library.md'
  - 'backlog/decisions/097-boot-budget-ratchets.md'
  - 'qa/model-config-ux-review-2026-09-26/judge-synthesis.md'
  - 'qa/model-config-ux-review-2026-09-26/verified-claims.md'
priority: high
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
- [ ] #16 Docs/User_Guide pages updated: settings.md Providers & Models and Console Behavior sections, with a new Verified-against stamp.
- [ ] #17 ./scripts/preflight.sh passes.
<!-- AC:END -->
