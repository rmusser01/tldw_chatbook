---
id: TASK-32540
title: >-
  Library Notes: Import once cannot be completed by keyboard — the picker opens
  with the tree focused, its buttons highlight by colour only, and the selection
  pane's Tab marks nothing before leaking into the rail
status: In Progress
assignee: []
created_date: '2026-09-13 06:46'
updated_date: '2026-09-14 18:32'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), both assessors, personas Sam and Jordan, Obsidian import workflow. D8 (B) + A's picker cell. Wave 3 fixed typed-path resolution (32251), memory (32174) and Ctrl+A (32229); focus was never in scope.

**What happened.** Add from files… → Import once → the picker opens at the configured `sync_directory` with the TREE focused: A's typed path went into the tree and Enter opened `..` (A 28); the File-name field had to be clicked (A 29, B 27). Open / Select folder / Cancel differ only by foreground colour 224 → 225/230/235 (B 27 ansi). After Select folder, the selection pane's Tab×6 marks none of Change selection / Clear / Check selection and ends in the rail's "Search Library…" box; Check selection and Import selected items were clicked (B 28). B's verdict for the Jordan · Obsidian cell: PARTIAL — three mandatory clicks. Captures: A 27, 28, 29, 30; B 24–28.

**Cause.** INFERRED. Docs contradicted: notes.md says the picker's File name field "can be typed into directly".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Import once picker opens with its path field focused: typed text lands in the field and Enter browses or resolves it
- [ ] #2 Open, Select folder and Cancel show a shape-based focus cue (not colour alone)
- [ ] #3 After Select folder, Tab from the confirmation walks Change selection → Clear → Check selection with a visible focus mark and does not leave the canvas
- [ ] #4 Import once completes from Add from files… to the receipt with the keyboard alone; the recipe is written in notes.md
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live on a scratch power profile with a 65-file vault (done: picker opens with the tree focused, typed text is swallowed; Open/Select folder/Cancel differ by colour+underline only; after Select folder Tab x5 reaches the rail search box and never marks the three pane actions).
2. RED pins in Tests/UI/test_library_notes_w4_import_keyboard.py against the real route.
3. Fixes: FileOpen._focus_initial_widget focuses the input bar field when offer_select_folder (FileSave precedent, task-1479); an APP-tier FileSystemPickerScreen Button:focus outline rule (widget DEFAULT_CSS loses to components/_buttons.tcss's Button:focus { outline: none }); focus the import canvas body when a selection lands, and scope Tab to #library-notes-canvas while the Notes view is 'import' (task-32246 mechanism).
4. GREEN + live keyboard-only walk at 235x52 and 100x30; notes.md recipe + stamps.
<!-- SECTION:PLAN:END -->
