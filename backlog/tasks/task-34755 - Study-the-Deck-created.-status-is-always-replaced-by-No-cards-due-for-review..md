---
id: TASK-34755
title: >-
  Study: the 'Deck created.' status is always replaced by 'No cards due for
  review.'
status: To Do
assignee: []
created_date: '2026-10-10 19:18'
labels:
  - study
  - ux
  - bug
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After Create Deck, the review status should confirm the new deck. create_deck selects the new deck, and the deck switch's own card-list rebuild (the Select.Changed worker) finishes last and resets the review panel to 'No cards due for review.', so the confirmation is never seen. Reproduced on dev 4de437a5c5 (2026-10-19): with TASK-33007's lock the deck switch's rebuild queues behind create_deck's own, still runs last, and the settled status reads 'No cards due for review.'.

A related stale write, found by reading the code during TASK-34751's review (not reproduced): when Flashcards is swapped out and back while a card-list rebuild is in flight, that rebuild's appends go to the old, pruned list and do nothing. But refresh_cards still sets current_cards, then writes 'Ready to review selected deck.' and the lifecycle controls, and create_deck/create_card then write their own status. All of these land on the newly composed panel. The new panel's own rebuild queues behind the lock and runs last, so this looks transient. _quiet_when_swapped_out only stops these actions from crashing; it does not stop these writes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After creating a deck, the review status reads that the deck was created (with its name, literally) once the deck switch settles
- [ ] #2 A later deck switch, refresh or view change does not resurrect a stale 'created' message
- [ ] #3 A test asserts the settled status after Create Deck
- [ ] #4 A card-list rebuild or create action that outlived its Flashcards panel writes no status, current_cards or lifecycle state to the panel composed in its place
<!-- AC:END -->
