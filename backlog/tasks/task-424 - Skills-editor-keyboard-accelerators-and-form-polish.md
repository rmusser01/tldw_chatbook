---
id: TASK-424
title: Skills editor - keyboard accelerators and form polish
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 15:19'
updated_date: '2026-07-21 20:03'
labels:
  - skills
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P2 from the 2026-07-21 Skills UX/NNG review. Everything on the Skills surface is mouse-button-only: Save/Delete/trust actions require scrolling past the whole form; no save/back accelerators; the create editor does not focus the Name field; name format (lowercase/numbers/hyphens) is only validated at save although the rule is known upfront; list rows separate the name button from its flags/description line by two blank rows weakening association. NNG heuristics 7 (flexibility and efficiency) and 5 (error prevention).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Save is reachable via a keyboard binding from anywhere in the editor,Back/escape leaves the editor honoring the unsaved-changes guard,Create editor focuses the Name field on open,Name format guidance is visible before save (placeholder or hint) and invalid names still error as today,List rows render name and metadata as one visually associated block,Covered by tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Keyboard: LibraryScreen gains ctrl+s -> action_library_skill_save and escape -> action_library_skill_back, both hard-gated by check_action to the open skill editor (_library_skill_editor_active: skills/create row + editor view) so the keys stay untouched everywhere else on the screen; Escape reuses a new shared _exit_library_skill_editor_guarded (dirty veto + toast, else reset-to-list) that the Back button handler now also delegates to. Focus: the Create > New skill entry call_after_refresh-focuses the Name Input. Guidance: create-mode Name Input placeholder states the format rule upfront ('lowercase letters, numbers, hyphens (e.g. code-review)'); save-time validation unchanged. Row proximity: .library-skill-row was height:2 + bottom margin 1 with the secondary flags/description line below - two blank rows inside one logical row (verified live); now height:1/margin:0 with the block-separation margin on the secondary line (source _agentic_terminal.tcss edited, bundle regenerated via build_css - diff verified timestamp+block only); parity CSS pin updated to document the anatomy difference vs the prompts row (which packs both lines into one 2-high label). 7 tests watched fail first (binding presence, check_action gating, save action worker, back action dirty/clean paths, create focus, placeholder, CSS pin). Canvas 85 passed; Skills+state+prompts-canvas 207 passed.
<!-- SECTION:NOTES:END -->
