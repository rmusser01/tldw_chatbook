---
id: TASK-369
title: Let users pick the Active Model from discovered models instead of retyping from memory
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
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
- [ ] #1 After discovery/save, the Model field should offer the discovered/saved models for selection (dropdown or typeahead), or Save selected should offer to set the active model directly
<!-- AC:END -->
