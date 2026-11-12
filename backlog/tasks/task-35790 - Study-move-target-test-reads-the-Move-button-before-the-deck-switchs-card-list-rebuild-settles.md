---
id: TASK-35790
title: >-
  Study move-target test reads the Move button before the deck switch's
  card-list rebuild settles
status: To Do
assignee: []
created_date: '2026-10-19 22:58'
labels:
  - study
  - tests
  - flaky-test
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_study_flashcards_screen.py::test_a_picked_move_target_stays_picked_and_enables_move` sometimes fails with `assert True is False` on `#move-selected-card-button.disabled`: the card was selected, the Move target Select holds "Chemistry", but Move is still disabled after one `pilot.pause()`.

Seen on 2026-10-19 while verifying TASK-34751: it failed in 2 of 4 whole-file runs on that branch (host load ~45 and ~97) and passed in all 3 whole-file runs on dev 8136af6f6f; run alone, it passed 9 of 9 times on each side. TASK-34751 does not touch the code this test exercises while the Study window is attached, so the difference is taken to be chance; nothing here proves that.

The likely cause, not yet confirmed: the `_open_flashcards_deck` helper returns once any listing has run and the list has rows, not once the deck switch's own rebuild has finished. If that rebuild lands after the test selects a card, it replaces the rows and the selection with them, and Move stays disabled. TASK-34751 fixed the same shape in the real-service contract test by waiting until no `study-*` worker is pending. The file is not in the UI PR gate (only `test_two_awaited_card_list_rebuilds_do_not_interleave` is), so this flake does not block merges.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the disabled Move button is shown with evidence, for example by forcing the deck switch's rebuild to land after the card selection
- [ ] #2 The test reads the Move button only once the card-list rebuilds it started have settled, and still fails if a picked Move target is reset (the original fix-round-4 regression)
- [ ] #3 The test passes repeatedly in whole-file runs under host load
<!-- AC:END -->
