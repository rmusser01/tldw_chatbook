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
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After creating a deck, the review status reads that the deck was created (with its name, literally) once the deck switch settles
- [ ] #2 A later deck switch, refresh or view change does not resurrect a stale 'created' message
- [ ] #3 A test asserts the settled status after Create Deck
<!-- AC:END -->
