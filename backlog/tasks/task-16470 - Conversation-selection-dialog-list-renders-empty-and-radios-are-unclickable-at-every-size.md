---
id: TASK-16470
title: Conversation selection dialog list renders empty and radios are unclickable at every size
status: To Do
assignee: []
created_date: '2026-08-14'
labels:
  - bug
  - ui
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The conversation selection dialog's list area draws EMPTY at every probed terminal size, and its radios cannot be clicked — which defeats the whole point of the dialog. Measured mechanism (TASK-15992 review, render evidence at 100x45 with three conversations loaded): the two `.form-row`s inside `.options-section` are `Horizontal`s with default `height: 1fr` while the section is `height: auto` (`tldw_chatbook/Widgets/conversation_selection_dialog.py:118-123`), so the options section swallows 28 of the container's 36 rows; `#conversations-list-container` (`height: 1fr`, `conversation_selection_dialog.py:85-89`) gets height 2. Inside it, each `ConversationItem` is a bare `Container` (default `height: 1fr`), so both items land on IDENTICAL coordinates outside the clip — `#conv-radio-11` and `#conv-radio-22` both at `Region(x=14, y=15, w=7, h=3)` — and zero conversation titles appear anywhere in the compositor strips. Consequence for a mouse user: `get_widget_at()` at a radio's own centre returns the options-section `Vertical`, and `await pilot.click("#conv-radio-11")` leaves `selected_conversation_id = None` with Generate disabled. A keyboard user CAN select (Tab reaches the radio, Enter selects) — but blind, since nothing is drawn. Same result at 80x30, 100x45, 100x50 and 120x60.

Pre-existing, not caused by TASK-15992: the review rebuilt the pre-15992 `compose()` (RadioSet wrapper restored, `.clear()` fixed) and rendered an identical empty 2-row box. The obvious one-line CSS patch (`.form-row { height: auto }` + `ConversationItem { height: auto }`) did NOT fix it, so this is genuinely task-sized layout work, not a quick patch. Render evidence and region tables are in the TASK-15992 review record (scratchpad `review15992.md`, section B1a).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Conversation titles are visible in the rendered list at 80x30 and 100x45 with multiple conversations loaded
- [ ] #2 `pilot.click` on a conversation radio selects it (selected_conversation_id set) and enables Generate
- [ ] #3 A render-evidence test pins the fix — compositor strips or widget-region assertions, not style probes (per this repo's style-probe-vs-render lesson)
<!-- AC:END -->
