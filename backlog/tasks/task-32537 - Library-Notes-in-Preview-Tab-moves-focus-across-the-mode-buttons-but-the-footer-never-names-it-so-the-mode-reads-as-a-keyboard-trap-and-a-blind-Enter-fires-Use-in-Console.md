---
id: TASK-32537
title: >-
  Library Notes: in Preview, Tab moves focus across the mode buttons but the
  footer never names it, so the mode reads as a keyboard trap and a blind Enter
  fires Use in Console
status: To Do
assignee: []
created_date: '2026-09-13 06:45'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), both assessors, persona Sam (keyboard-only, low vision). A rated it P1 as a trap; B's captures contradict the mechanics, so filed at P2 as a footer-chip defect.

**What happened.** Open a note, Tab to Preview, Enter, then Tab×6: the footer stays "pgup/pgdn scroll | esc back to notes" on every Tab (A 11, 13). A read Tab / Shift+Tab / F6 as inert and Enter as doing nothing, with Escape (to the list, losing the note) the only exit. B shows the heavy `┃ … ┃` box does move across the mode buttons and the sixth blind Enter fired Use in Console (B 13, 14, 15) — focus moves; what is missing is the chip. Captures: A 11, 13; B 13, 14, 15.

**Cause.** INFERRED: the `_LIBRARY_NOTE_EDITOR_ENTER_LABELS` chips task-32246 added are appended to the Edit and Info footer tiers, not the Preview tier; `Tests/UI/test_library_notes_w3_layout.py::test_activating_preview_focuses_its_scroll_owner` pins focus-on-open only and `test_library_notes_wave_editor_keys.py` pins chips for the delete prompt and Tab-out-of-body only. F6 from the preview region goes to the rail search box (`library_screen.py:1433`), which is why it looked inert. Docs contradicted: notes.md promises the footer names the focused control so focus is never unaccounted for.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With Preview open, every Tab and Shift+Tab stop shows its "enter …" chip in the footer exactly as the Edit and Info tiers do, while pgup/pgdn stays advertised
- [ ] #2 Escape from Preview returns to Edit first, or notes.md is corrected to say it returns to the list and the footer says so
- [ ] #3 A test pins the Preview-tier chip for ‹ Notes, Edit, Info, Save and Use in Console
<!-- AC:END -->
