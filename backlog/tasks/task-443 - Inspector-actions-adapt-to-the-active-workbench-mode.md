---
id: TASK-443
title: Inspector actions adapt to the active workbench mode
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 09:38'
updated_date: '2026-07-24 07:20'
labels:
  - roleplay
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). The inspector's action block is identical in every mode: Start Chat and Export PNG render in Dictionaries/Lore modes where they cannot apply, while Duplicate exists in the Dictionaries/Lore library rails but not Characters. Buttons that can never apply to the selected kind should not render; parity gaps (Duplicate for characters) should be closed or justified.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Inspector shows only actions applicable to the selected item kind (or clearly disabled with a reason)
- [x] #2 Duplicate is available for characters or its absence is an explicit decision
<!-- AC:END -->

## Implementation Plan

1. Read the inspector pane's action block and `_apply_action_state`, and how `PersonasScreen` pushes kind + Console gate into it (task-440 machinery).
2. AC1: add kind-applicability gating in `_apply_action_state` — buttons that can NEVER apply to the selected kind stop rendering (`display = False`), while "applies but currently blocked" keeps the existing disabled+tooltip+readiness-copy flow untouched.
3. AC2: extend the library rail's existing Duplicate seam (`set_mode` visibility + `PersonaActionRequested(action="duplicate")` dispatch) to characters, adding `_duplicate_selected_character` that mirrors the dictionary/lore duplicators and reuses `ccp_character_handler.create_character`.
4. Tests: per-kind visibility matrix in the inspector pane tests (RED against the identical-block baseline), duplicate flow + conflict path in the workbench tests, update the dictionaries-mode visibility test.
5. Verify: touched test files, workbench readiness/Console/gate suite (task-440 must stay green), app import.

## Implementation Notes

**Visibility pattern chosen (AC1):** rendering-level gating via `Button.display`, not disabled-with-reason, for actions that can NEVER apply to the selected kind. Rationale: "never applies" is categorically different from "applies but currently blocked" — the existing readiness machinery (task-440's `set_console_actions_enabled` enabled/reason/provider_block_reason flow) already owns the second case, and keeping the two axes separate means the kind gate composes with (never fights) the Console gate: `display` is set from kind, `disabled`/tooltip/readiness copy stay driven exactly as before. Three module-level constants in `personas_inspector_pane.py` document the matrix (`_CONSOLE_ACTION_APPLICABLE_KINDS`, `_EXPORT_JSON_APPLICABLE_KINDS`, `_EXPORT_PNG_APPLICABLE_KINDS`). With no selection (kind `None`) every action renders, preserving the pre-selection "Console blocked: select an item" baseline.

**Per-kind action matrix (rendered / hidden):**

| Action             | character | persona_profile | dictionary | lore |
|--------------------|-----------|-----------------|------------|------|
| Attach to Console  | shown     | shown           | hidden     | hidden |
| Start Chat         | shown     | shown           | hidden     | hidden |
| Export JSON        | shown     | shown           | hidden     | hidden |
| Export PNG         | shown     | hidden          | hidden     | hidden |
| Delete             | shown     | shown           | shown      | shown |

Export PNG hides for persona_profile because personas have no PNG card — `_open_export_dialog` hard-rejects `fmt=="png"` for non-characters, so rendering it was a dead end. Dictionaries/lore keep their own export flows (`DictionaryExportRequested` / `LoreBookExportRequested` in their center panes), so hiding the inspector's card-export buttons removes dead UI without removing capability.

**Duplicate decision (AC2):** characters genuinely had NO duplicate seam anywhere (the P3a "New/Duplicate/Delete" memory was about dict/lore; `set_mode` gated `#personas-library-duplicate` to `("dictionaries","lore")` and the `action == "duplicate"` dispatch had no characters arm). Closed the parity gap where dictionaries/lore already have it — the library rail: `set_mode` now shows Duplicate for `("characters","dictionaries","lore")` and `_handle_action_requested` routes characters to a new `_duplicate_selected_character` that mirrors `_duplicate_selected_dictionary`/`_duplicate_selected_lore` (off-thread full-record read via `fetch_character_by_id`, "(copy)"/"(copy N)" disambiguation against the cached list, then the EXISTING `ccp_character_handler.create_character` seam — no new duplication engine). Personas mode still has no Duplicate: persona profiles are server-scope records with no local duplicate seam, and the review filed no parity claim for them — explicitly out of scope here.

**Modified files:**
- `tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py` — kind-applicability constants + `display` gating in `_apply_action_state`.
- `tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py` — Duplicate visible in characters mode (`set_mode` + `on_mount` first paint).
- `tldw_chatbook/UI/Screens/personas_screen.py` — `_duplicate_selected_character` + characters arm in the duplicate dispatch.
- `Tests/UI/test_personas_inspector_pane.py` — per-kind visibility matrix (6 new tests; 4 were RED against the identical-block baseline: dictionary/lore hide-all, persona PNG-hide, clear-selection restore).
- `Tests/UI/test_personas_workbench.py` — character duplicate happy path + ConflictError path.
- `Tests/UI/test_personas_dictionaries.py` — mode-visibility test updated: Duplicate now hidden ONLY in Personas mode.
