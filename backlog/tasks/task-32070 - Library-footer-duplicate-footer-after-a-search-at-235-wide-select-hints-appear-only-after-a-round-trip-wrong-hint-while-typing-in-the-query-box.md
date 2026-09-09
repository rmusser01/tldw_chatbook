---
id: TASK-32070
title: >-
  Library footer: duplicate footer after a search at 235 wide; select hints
  appear only after a round trip; wrong hint while typing in the query box
status: Done
assignee: []
created_date: '2026-09-08 18:26'
updated_date: '2026-09-08 19:56'
labels:
  - library
  - footer
  - ux
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After a rail search at 235x52 a wrapped 'vidence | F6 …' line painted above the real footer; Space/s hints in Media select mode appeared only after an Escape round trip; while typing in the Search/RAG query box the footer reads 'enter select evidence' although Enter runs the search. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 21.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The footer never paints twice
- [x] #2 Select-mode hints are present as soon as select mode is entered
- [ ] #3 The footer names the focused control's Enter action in the Search/RAG query box
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Probe the footer at 235x52 after a rail search (one AppFooterStatus? one row?) and reproduce live in tmux before changing anything.
2. RED/regression: exactly one footer widget, one row tall, its shortcut text inside its own slot.
3. Select-mode hints at 235x52 the moment select mode is entered (the 100x30 half is task-32060's gate fix).
4. 'enter run search' in the Search/RAG query box is another branch's change: assert it only if already present, otherwise skip that assertion and say so.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#2 was the real defect and is fixed by task-32060's gate in this same group: the select-mode hints came only after an Escape round trip because 's' itself was inert until focus had left and re-entered the list. With the gate fixed, pressing 's' from a focused Items row switches the footer to 'space toggle selection | s done selecting' in the same frame -- pinned at 235x52 and 100x30, and live-verified at both (caps/32070-235x52-select-hints-immediately.txt).

AC#1 (footer never paints twice): could NOT be reproduced, headlessly or live. At 235x52 after a rail search there is exactly one AppFooterStatus, one row tall, with its hint Static inside its own slot -- both in an app-test and live in tmux on the seeded profile (caps/32070-235x52-single-footer-after-rail-search.txt). The critique's wrapped 'vidence | F6 …' fragment is the same shortcut line wrapping, which needs a narrower footer slot than 235 columns gives it; I have pinned the invariant as a regression guard rather than change code I cannot show is wrong. If it recurs, the capture to take is the one that shows the slot width.

AC#3 is NOT done here and stays unticked: the 'enter run search' copy for the Search/RAG query box belongs to the keyboard branch of this wave, which owns that footer set. The test for it is written and skips with the current label ('enter select evidence' while typing) rather than duplicating the change; it starts asserting the moment that branch lands.

Files: Tests/UI/test_library_crit8_polish_media.py (three tests, committed with the sibling tasks in this group's shared test file), Docs/User_Guide/library/media-and-conversations.md (stamp).
<!-- SECTION:NOTES:END -->
