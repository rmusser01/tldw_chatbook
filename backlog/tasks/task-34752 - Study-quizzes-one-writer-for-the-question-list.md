---
id: TASK-34752
title: 'Study quizzes: one writer for the question list'
status: To Do
assignee: []
created_date: '2026-10-10 19:17'
labels:
  - study
  - ui
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Study > Quizzes rebuilds #quiz-question-list in refresh_questions the way Flashcards rebuilt #card-list before TASK-33007 serialised those rebuilds (one lock in StudyFlashcardsController.refresh_cards) and made a Flashcards action end quietly when its view is swapped out: it clears the list, awaits the question fetch, then appends one row per question with an await per row. Selecting a quiz in code (refresh_quizzes sets #quiz-select.value) fires Select.Changed, whose handler rebuilds the list again while the caller may be rebuilding it too, so overlapping rebuilds can interleave and list a question more than once, or write into a quiz view that a sub-view switch has already torn down (NoMatches/MountError out of a worker exits the app). Found while fixing TASK-34751; still the shape on dev 4de437a5c5 (2026-10-19: no lock, no swap guard in quizzes_handler), not reproduced yet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Two overlapping question-list rebuilds leave exactly one row per question, in order, and current_quiz_questions matches the rows
- [ ] #2 A question-list rebuild still in flight when the Quizzes view is torn down (or the app shuts down) stops without writing, and no caller writes the attempt panel after it
- [ ] #3 A regression test fails on origin/dev for the stated reason and passes with the fix
<!-- AC:END -->
