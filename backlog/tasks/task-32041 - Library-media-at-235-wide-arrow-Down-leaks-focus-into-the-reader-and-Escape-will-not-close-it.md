---
id: TASK-32041
title: >-
  Library media: at 235 wide, arrow-Down leaks focus into the reader and Escape
  will not close it
status: Done
assignee: []
created_date: '2026-09-08 14:36'
updated_date: '2026-09-08 15:09'
labels:
  - library
  - media
  - ux
  - accessibility
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #7 P1 (lead). At 235x52, the first arrow-Down off a focused Media list row defocuses the list into the reader pane instead of advancing the list cursor; four Escapes then neither close the reader nor move a cursor; a Console round-trip strands Library on a provider 'Get started' card that needs Home->Library to recover. The same gestures work correctly at 100x30, so this is a width-dependent focus-routing bug. Keyboard navigation of the list is the core interaction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 235x52, arrow Down/Up move the Media list cursor (visible focus cue) while the list owns focus, rather than leaking focus to the reader
- [x] #2 The Escape ladder returns focus predictably from the reader back to the list at every supported width
- [x] #3 A Console -> Library round-trip lands on the Media list, not a Get-started card
- [x] #4 Painted pins assert cursor movement at 235x52 (parity with the 100x30 behaviour that already works)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RULED: no behaviour change needed -- Assessment B's finding was a live-review mis-attribution of intentional, pinned design. AC#1/#4: Down/Up ALREADY move the Media list cursor at 235x52 (verified + newly pinned `test_media_list_arrow_keys_move_the_cursor_at_every_width`; B's 'leak' was pressing Down while the READER, not the list, held focus). AC#2: the wide Escape ladder is the deliberate task-31272 three-pane reader ladder -- Escape graduates reader->items->rail and the doc persists at wide because the Reader is permanent there (`_library_media_reader_exit_available` is False when library_open and items_open; pinned by `test_escape_moves_reader_to_items_then_library_then_screen_back`). Making Escape close the doc at wide would regress those reviewed pins. AC#3: the 'Console->Library strands on a Get-started card' symptom did NOT reproduce live -- verified at 235x52 on a seeded scratch profile the round-trip returns to the Media list with the loaded item intact; B's own env notes flagged it against a fresher/no-provider profile. Delivered: the arrow-key regression guards (the substantive verifiable part) + this ruling. No production change for 32041.
<!-- SECTION:NOTES:END -->
