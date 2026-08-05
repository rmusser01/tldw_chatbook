---
id: TASK-369
title: Let users pick the Active Model from discovered models instead of retyping from memory
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
updated_date: '2026-07-23 08:40'
labels: [console, ux]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Saving the discovered model only appends it to a provider list; the 'Model' field stays an empty free-text input ('Model name' placeholder) and readiness stays 'llama.cpp / not selected'. The discovery list disappears from screen right after 'Save selected', and typing 'gemma' into Model offers no autocomplete or dropdown (j1-32). To reach readiness I had to type the full 56-character 'gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf' exactly, from memory of a string no longer visible anywhere.

**Repro:** Discover + Save selected -> observe list disappears, Model still placeholder -> focus Model, type 'gemma' -> no suggestions -> readiness only passes after typing the full gguf filename.

**Verifier note:** Code-confirmed: the Settings Model field is a bare Input with placeholder 'Model name' (settings_screen.py:6118-6124), no suggester, no Select, and Save selected only appends model ids to the provider list (_save_selected_discovered_provider_models) — nothing offers them for activation, and the discovery list state resets. Not covered by ledger: provider-catalog-display-names covers the provider dropdown, settings-modal-model-prefill covers the Console modal, task-188 added a discovered-models Select to the CONSOLE settings modal and one-click detected-server on the card — but the Settings screen, which is exactly where the setup card routes first-run users, still demands full recall of a 56-char gguf filename. P2: real recognition-over-recall failure on the primary onboarding path; partial mitigations exist on other surfaces.

**Source:** Console UX expert review 2026-07-20 (finding j1-model-field-pure-recall; P2, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J1 new-user cold start journey. Evidence: `j1-30-save-selected-retry.png`, `j1-31-model-field-focused.png`, `j1-32-model-typeahead.png`, `j1-33-category-saved.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After discovery/save, the Model field should offer the discovered/saved models for selection (dropdown or typeahead), or Save selected should offer to set the active model directly
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Took BOTH branches of the AC (they cover complementary moments in the flow):

1. **Save selected sets the active model.** After a successful Save selected, the
   (previously empty) Model field is populated with the first saved model id, so
   readiness passes without retyping a 56-char gguf name the now-cleared
   discovery list no longer shows. A field the user already set is left
   untouched. Pure decision in `_model_to_activate_after_save`; applied via
   `_activate_saved_model_if_field_empty` (setting `.value` fires Input.Changed,
   which stages the model draft).
2. **Typeahead on the Model field.** The Model `Input` now carries a
   `SuggestFromList` suggester (`_model_field_suggester`) of the discovered model
   ids, so while a discovery result is on screen, typing a prefix (`gemma`)
   completes to the full id. Refreshed when discovery finishes and after a save
   (`_refresh_model_field_suggester`); `None` when nothing is discovered.

Verified RED→GREEN: `test_model_to_activate_after_save_prefers_first_saved_when_field_empty`
(pure: empty→first-saved, skips blanks, keeps an existing choice) and
`test_model_field_suggester_completes_discovered_ids` (a prefix completes to the
full gguf id; empty discovery → no suggester). 271 settings tests green (the one
sweep failure, `test_theme_category_opens_without_crashing`, is a pre-existing
load-contention flake — passes isolated on this branch AND on origin/dev).
<!-- SECTION:NOTES:END -->
