---
id: TASK-436
title: Characters mode center pane empty-state guidance
status: Done
assignee: []
created_date: '2026-07-21 09:38'
updated_date: '2026-07-23 20:27'
labels:
  - roleplay
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). Observed live on first visit: the Characters mode center pane is a large blank area with no copy at all until a selection is made (the Lore and Personas modes at least have library-rail guidance). Combined with "Console blocked: select an item" in the inspector, the first impression reads as broken rather than "select or create a character".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With no selection, the Characters center pane shows guidance for the three next actions (select, New, Import)
- [x] #2 Guidance disappears once a selection exists
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Characters mode now shows onboarding guidance in the center pane when no character is selected, instead of a blank area that read as broken.

**Approach:** a dedicated `Static` `#personas-characters-empty` in `#personas-detail-stack` (registered in `_CENTER_VIEW_IDS`), resolved once at the top of `_show_center`: when `visible_id is None` and `active_mode == "characters"` and `state.selected_entity_id` is empty, it shows the guidance widget instead of blanking. Because every no-selection path (mode-enter, first mount, delete, cancel-New, restore-failure) funnels through `_show_center(None)`, this single choke point covers them all, and explicit-id calls (character card / editor) hide the guidance automatically — so **AC#2 needs no extra wiring**.

**Copy (AC#1):** a single non-adaptive constant `_CHARACTERS_EMPTY_GUIDANCE` naming all three next actions (pick from the list / **New** / **Import**). Deliberately not count-adaptive: `on_mount` renders the center before `_character_total` loads, so an adaptive copy would be wrong on first mount.

**Not reused:** `#personas-mode-placeholder` (the retired-"prompts" fallback, pinned by tests) is left untouched — the guidance is a separate widget. CSS lives in the screen's inline `DEFAULT_CSS` (not the generated bundle).

**Testing:** 4 pilot tests (guidance shown when no selection naming New/Import; visible-before → hidden-after selection pinning the AC#2 transition; hidden in other modes + returns on switch-back; returns after deleting the selected character). Full personas regression (310) green; live tmux-verified render + swap-to-card. Task review: spec ✅, code quality Approved.

**Scope:** Characters mode only; the inspector "Console blocked" copy is TASK-443's remit; extending center guidance to other empty modes is a possible follow-up. Files: `personas_screen.py`, `Tests/UI/test_personas_workbench.py`.
<!-- SECTION:NOTES:END -->
