---
id: TASK-32537
title: >-
  Library Notes: in Preview, Tab moves focus across the mode buttons but the
  footer never names it, so the mode reads as a keyboard trap and a blind Enter
  fires Use in Console
status: Done
assignee: []
created_date: '2026-09-13 06:45'
updated_date: '2026-09-14 19:25'
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
- [x] #1 With Preview open, every Tab and Shift+Tab stop shows its "enter …" chip in the footer exactly as the Edit and Info tiers do, while pgup/pgdn stays advertised
- [x] #2 Escape from Preview returns to Edit first, or notes.md is corrected to say it returns to the list and the footer says so
- [x] #3 A test pins the Preview-tier chip for ‹ Notes, Edit, Info, Save and Use in Console
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live on the power profile (done: Preview + Tab x6 keeps 'pgup/pgdn scroll | esc back to notes').
2. RED pin in Tests/UI/test_library_notes_w4_editor.py for the Preview-tier focus chip.
3. Append the focus chip to the 'preview' footer tier through the same helper the editor tier uses.
4. AC#2: re-derive the doc branch - Escape from Preview goes to the list, footer says 'esc back to notes', notes.md:430 already says so. Add the footer-names-every-Tab-stop sentence at :386.
5. GREEN + live at 235x52 and 100x30.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced live first at dev 2f97a42c9a on a scratch power profile, 235x52:
Preview open, Tab x6 walked the heavy focus box across ‹ Notes, Edit, Preview,
Info, Save and Use in Console while the footer stayed on "pgup/pgdn scroll |
esc back to notes" for every one of them (capture
`editor-00-32537-preview-tab6-footer-frozen-235x52`). B's reading of the
mechanics was right and A's was not: focus moves, the chip was missing.

**Fix.** The `region == "preview"` footer branch now appends the focused
control's own Enter action, exactly as the editor tier has since task-32246.
The append-don't-prepend reasoning (the narrow stage keeps only the leading
chips that fit, so a leading "enter …" evicts the exit) moved onto one shared
helper, `_with_library_notes_focus_chip`, which the editor, Preview and
Notes-list tiers all call — three copies of that rule would have drifted.
No new label table was needed: `_LIBRARY_NOTE_EDITOR_ENTER_LABELS` already
names every control Preview's Tab cycle reaches.

**AC#2 — re-derived, no code.** Escape from Preview goes to the LIST, not
back to Edit (verified live), the footer already says "esc back to notes",
and notes.md's "Escape" row already documented that. So the AC's second
branch was satisfied before this task; what was missing was the Preview row
saying so, which it now does. Nothing was loosened or deleted to reach that
conclusion — `::test_preview_footer_advertises_escape_to_the_list` pins the
chip AND the destination together, and passes on detached origin/dev.

**Tests.** `Tests/UI/test_library_notes_w4_editor.py::test_preview_tier_names_
every_tab_stop` walks all six stops through the real
`_library_notes_footer_shortcuts`. RED on detached origin/dev (no "enter"
key at all), GREEN here. Live at 235x52 and 100x30
(`editor-10-32537-preview-tab6-use-in-console-chip-235x52`,
`editor-10-32537-preview-chips-100x30`).

Modified: `tldw_chatbook/UI/Screens/library_screen.py`, the pin file,
`Tests/UI/test_library_honesty_accessibility.py` (its SimpleNamespace fake
needed the new shared helper bound, the same way it already binds
`_notes_footer_tier` and `_library_focus_enter_label`),
`Docs/User_Guide/library/notes.md`.
**Review round (Minor 5): the stamp now claims only what the captures show.**
It listed all six chips and said "at both sizes"; each saved capture is the
LAST stop of the Tab walk ("enter use in Console"), at 235x52 and at 100x30.
The stamp says so and names
`::test_preview_tier_names_every_tab_stop` for the other five stops.
<!-- SECTION:NOTES:END -->
