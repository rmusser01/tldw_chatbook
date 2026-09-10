---
id: TASK-32210
title: >-
  Library Media: the type chooser marks focus by a 1.09:1 background difference
  with no glyph
status: Done
assignee: []
created_date: '2026-09-10 14:52'
updated_date: '2026-09-10 19:23'
labels:
  - library
  - media
  - accessibility
  - critique-9
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In the type chooser the focused row is background rgb(30,30,30) against rgb(39,39,39) elsewhere (darker than its neighbours, no glyph); only the active value has `✓`. The footer has committed the user to a keyboard interaction whose cursor is invisible in a plain-text capture. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 6.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The focused chooser row carries the `█` left-edge bar the lists use; `✓` keeps marking the active value
- [x] #2 A painted-cell test pins the bar on the focused row and its absence on the others
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing painted-cell test in Tests/UI/test_library_crit9_media_list.py
2. Add LIBRARY_CHOICE_CURSOR + LibraryChoiceOptionList to library_choice_strip.py
3. Use it for both media choosers
4. Re-run the chooser pins
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
LibraryChoiceOptionList in tldw_chatbook/Widgets/Library/library_choice_strip.py keeps the house `█ ` cursor on exactly the highlighted option's prompt, on mount and on every highlighted change; both media choosers (#library-media-type-choices, #library-media-sort-choices) now construct it. CSS cannot do this -- option-list--option-highlighted is a Rich style, so it can recolour an option but not add a glyph -- hence the prompt rewrite; Option._set_prompt mutates in place so each option's choice_value payload survives, and the ✓ active marker is untouched (the highlighted active option reads '█ ✓ All types'). The on_mount override deliberately does NOT call super(): Textual dispatches every on_mount in the MRO subclass-first, so OptionList.on_mount's _update_lines() runs anyway and runs after this paint -- calling it explicitly double-fires it and trips Tests/UI/test_on_mount_mro_convention.py (backlog/docs/lessons-textual.md). Four prompt-reading assertions in Tests/UI/test_library_choice_strips.py moved to the new exact strings. New tests: Tests/UI/test_library_crit9_media_list.py -- the painted-cell leg asserts the bar reaches exactly ONE painted row, at the highlighted index, so absence is pinned in the paint and not only in the prompts. Live: 235x52 and 100x30 on the seeded power profile, the bar following each Down and never on two rows. Files: library_choice_strip.py, library_media_canvas.py, test_library_crit9_media_list.py, test_library_choice_strips.py, Docs/User_Guide/library/media-and-conversations.md.
<!-- SECTION:NOTES:END -->
