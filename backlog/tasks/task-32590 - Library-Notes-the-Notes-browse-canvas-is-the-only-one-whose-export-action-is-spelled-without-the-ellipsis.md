---
id: TASK-32590
title: >-
  Library Notes: the Notes browse canvas is the only one whose export action is
  spelled without the ellipsis
status: To Do
assignee: []
created_date: '2026-09-14 23:46'
labels:
  - library
  - notes
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by task-32558 fix round 2, after the label caused two false guide sentences. Four browse canvases offer the same scoped-export action and three spell it 'Export…': Media (library_media_canvas.py:1178), Conversations (library_conversations_canvas.py:150) and Prompts (library_prompts_canvas.py:890). Notes ships it bare — ('Export', 'library-notes-export') at library_notes_canvas.py:1842 — and six of task-32558's captures paint 'Add from files…     Export'.

The ellipsis is not decoration in this codebase's grammar: it marks an action that opens something rather than doing something, which is exactly what all four of these do (each opens the same 'Export bundle (.zip)' form). So the odd one out is the one that is wrong, not the three.

The cost is already paid twice. import-and-export.md told readers to press 'Export…' in Notes in two separate places, and the task-32558 stamp certified one of them as matching the shipped toolbar. Both are now corrected to name the bare label AND to say Notes is the exception — copy that exists only because of the inconsistency and can be deleted with it. A stale docstring at library_notes_controller.py:5406 describes the Notes editor's 'Export…' action too.

Note for whoever takes this: Notes' select-mode export is a different control ('Export selected', shortened to 'Export' on a compact pane, library_notes_canvas.py:1657 and :2902) and must stay distinguishable from the browse action after any rename.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The four browse canvases spell the scoped-export action identically, and the chosen spelling is justified against this repo's ellipsis grammar rather than by majority
- [ ] #2 Notes' select-mode 'Export selected' / compact 'Export' remains distinguishable from the browse action at every pane width
- [ ] #3 The stale 'Export…' references in comments and docstrings that describe the Notes action are corrected in the same commit
- [ ] #4 import-and-export.md's two 'Notes is the exception' clauses are removed once the exception is gone, and the guide named by a pin so the copy cannot drift back
<!-- AC:END -->
