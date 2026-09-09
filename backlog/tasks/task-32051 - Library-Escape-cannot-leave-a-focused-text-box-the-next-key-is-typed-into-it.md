---
id: TASK-32051
title: 'Library: Escape cannot leave a focused text box; the next key is typed into it'
status: Done
assignee: []
created_date: '2026-09-08 18:22'
updated_date: '2026-09-08 19:58'
labels:
  - library
  - ux
  - keyboard
  - critique-8
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With focus in the rail 'Search Library…' box or the Search/RAG query box, Escape is a no-op (footer stays 'typing in field') and the next printable key is inserted as text (`i` landed in the search box instead of opening Import). Only Tab/F6 leave the box, which is undocumented and breaks the keyboard-only journey. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Escape in the rail search box and in the Search/RAG query box moves focus to the canvas (or the documented target) without inserting text
- [x] #2 The footer hint reflects the new focus after Escape
- [x] #3 The next printable key after Escape performs its canvas action (for example `i` opens Import)
- [x] #4 The behaviour is documented in library.md's Keyboard section
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live (rail search box + RAG query box Escape)\n2. Failing test in Tests/UI/test_library_crit8_keyboard.py\n3. Add library_blur_text_field escape binding + check_action gate\n4. Green, live-verify, docs
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Escape in a Library text box now hands focus back to the canvas, so the next printable key is a canvas key.

Approach: one more `escape` binding, `library_blur_text_field`, declared AFTER every surface-back Escape (editor -> list, Ingest -> hub, ...) and BEFORE `library_list_focus_rail`. Position is the contract -- Textual runs the first binding whose check_action passes, so this can never steal a surface's own exit, and it beats the focus-rail hop, which was a no-op when the caret was already in the rail search box (the reported defect). `check_action` restates that order explicitly (derived from BINDINGS, not a literal list) because F1's help panel filters bindings through check_action alone and would otherwise show two Escape rows.

The action focuses the first control inside `#library-canvas` that is neither the field itself nor another text box, then re-registers the footer; nothing typed is cleared. Live on the seeded and fresh profiles: `/` -> footer 'typing in field', Escape -> footer switches to the canvas set, `i` opens Import instead of typing into the box.

Files: tldw_chatbook/UI/Screens/library_screen.py, Tests/UI/test_library_crit8_keyboard.py (new), Docs/User_Guide/library.md.
<!-- SECTION:NOTES:END -->
