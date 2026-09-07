---
id: TASK-2078
title: >-
  Library: reason tooltips on disabled export/skill-trust buttons and editor
  shortcut hints (F-018, F-019)
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-04 06:19'
labels:
  - ux-review
  - library
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Export selected (3 canvases), skill-trust Unlock/Review/Approve, and editor Discard are disabled without reason tooltips; editor ctrl+s/escape are advertised nowhere. Evidence: library_conversations_canvas.py:100-105, library_skills_canvas.py:1154-1175, library_screen.py:877-884. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every disabled button on Library surfaces a reason tooltip,Skill editor shows its ctrl+s/escape hints inline,Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (tooltips + one hint line; no behavior changes). Steps: 1. RED tests: (a) export-selected buttons in the three multiselect canvases carry a reason tooltip while disabled (and none/an action one while enabled), flipping in the screen's in-place toggle patcher; (b) skill-trust Unlock/Review/Approve tooltips explain the disabled state per trust status, flipping in the screen's no-recompose trust patcher; (c) skill-editor Discard tooltip explains 'not dirty' while disabled and flips in _set_library_skill_discard_enabled; (d) the skill editor shows a dim inline 'ctrl+s / esc' hint line in the normal editing state, absent during conflict/delete-confirm (where ctrl+s is gated off). 2. Implement: shared tooltip constants/helpers in library_shell_state.py (export) and library_skills_canvas.py (trust + discard helpers mirroring the existing pure-function copy pattern); wire the three canvas composes + two screen patchers; hint Static reusing the existing dim library-prompt-field-hint class in _compose_editor. 3. Run multiselect x3 + skills canvas + prompts canvas (sibling pattern) + shell tests + ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
(a) Reason tooltips on every disabled Library action, following the rail-row disabled_tooltip / workspaces-handoff pattern: the three canvases' 'Export selected' (shared constants LIBRARY_EXPORT_SELECTED_TOOLTIP / _DISABLED_TOOLTIP in library_shell_state.py, wired at compose in conversations/media/notes canvases AND in the screen's no-recompose row-toggle patcher so the tooltip flips live with disabled); skill-trust Unlock/Review/Approve (new pure helpers skill_trust_unlock_tooltip/_review_tooltip/_approve_tooltip in library_skills_canvas.py, wired at compose AND in the no-recompose trust patcher); skill-editor Discard (SKILL_DISCARD_TOOLTIP_CLEAN/_DIRTY constants, compose + _set_library_skill_discard_enabled patcher). (b) Skill editor shortcut hints: SKILL_EDITOR_SHORTCUT_HINTS ('ctrl+s Save · esc Back to list') rendered as a dim inline Static (library-prompt-field-hint class) right under the Back button -- the file-notes git panel's guide-line pattern -- hidden during conflict/delete-confirm where ctrl+s is gated off. Files: library_shell_state.py, 4 canvas widgets, Widgets/Library/__init__.py (re-exports), library_screen.py (2 patchers + import), Tests/UI/test_library_skills_canvas.py (+4 tests, 1 extended screen test, 2 stale-pin fixes: test_footer_u_hint_only_registered_on_search_row updated to the F-012 landing-shortcut contract -- missed in task-2073's sweep -- and test_library_screen_binds_skill_editor_keys handles Binding objects, stale since RAG-36), test_library_multiselect_{conversations,media,notes}.py (+1 each; notes gained its first widget-mount test). Verified: 4 files 124 passed; full test_library_shell.py 314 passed; destination/parity/prompts/footer sweep 303 passed + 1 skip. Ruff clean on all changed files. ADR: not required (tooltips + one hint line; no behavior changes). Commit e7aadcf71.
<!-- SECTION:NOTES:END -->
