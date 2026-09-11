---
id: TASK-32348
title: >-
  Library Media viewer: Find is unreachable by keyboard and inert to the click,
  and a failed attempt leaves 't trash' armed
status: Done
assignee: []
created_date: '2026-09-11 06:15'
updated_date: '2026-09-11 07:34'
labels:
  - library
  - media
  - keyboard
  - critique-10
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Find is neither in the Tab order (10 Tabs each way) nor opened by the same SGR click that its sibling buttons accept; typing a query after the failed click fired the t accelerator and armed 'Delete this media?' (B D4/D4a, caps 27/28). No Find key binding exists in the screen's bindings. Docs promise a focused search bar (media-and-conversations.md:529). Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Find opens from the keyboard (a binding with a footer chip) and is in the viewer's Tab order
- [x] #2 No single-key destructive accelerator fires while the viewer's Find gesture is pending
- [x] #3 Pinned
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Unit test: analysis_find_unavailable_reason refuses info/highlights
2. Implement the refusal in library_media_viewer_state
3. Extract the Find handler body, add ctrl+f binding + action + check_action gate + footer chip
4. Gate 't' (trash) off while find_open
5. UI tests + docs
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
One cause behind both halves of D4/D4a. analysis_find_unavailable_reason only ever considered the Analysis tab and returned '' for every other mode, but _compose_active_body mounts a Find bar on Read and Analysis ONLY. So on Info/Highlights the button was enabled, armed find_open, mounted nothing, and the focus call found no input -- focus stayed outside any Input, where the 't' of a typed 'token' reached the screen accelerator and armed 'Delete this media?'.

- library_media_viewer_state.py: the gate refuses info/highlights with 'This tab has no text to search · switch to Read or Analysis.' (renamed nothing -- 12 call sites reference the name). The disabled button carries that reason on screen and in its tooltip; the key drops from the footer with it.
- library_screen.py: the handler body is extracted to _toggle_library_media_find, so the new ctrl+f Binding -> action_library_media_reader_find and the Button press run ONE implementation and can never diverge. check_action gates the action on the same reason string, so the ('ctrl+f','find') footer chip can never advertise a refusal. ctrl+f is not printable, so it survives the focused-Input footer transform and still works from inside the Find field itself.
- AC#2: check_action refuses library_media_move_to_trash while find_open, and the 't trash' chip drops with it.

AC#1's Tab-order half was DISPROVED, not implemented: #library-media-reader-find is an ordinary focusable Button and is in screen.focus_chain -- the test now pins that. What it lacked was a key. No Tab-order change was made. (B's '10 Tabs each way' most likely walked the Items pane's own rows.)

Live (tmux, seeded profile, 235x52): Reader footer shows 'ctrl+f find'; ctrl+f mounts 'Search content…' focused and the footer becomes 'typing in field | esc close | after esc: / focus search · ] next item · l read later · c use in Console | ctrl+f find | F6 next pane' -- 't trash' absent. Typing 'token' filled the query, no delete confirmation. On Info and Highlights the button reads '○ Find', the chip is gone, and ctrl+f is inert.

Files: tldw_chatbook/Library/library_media_viewer_state.py, tldw_chatbook/UI/Screens/library_screen.py (BINDINGS, Reader route branch, check_action branch, Find handler), Tests/UI/test_library_crit10_viewer.py (new), Tests/UI/test_library_media_reader_flow.py (two test doubles updated for the new gate read and the extracted method, plus one new assertion), Docs/User_Guide/library/media-and-conversations.md (Find row, stamped).
<!-- SECTION:NOTES:END -->
