---
id: TASK-4011
title: >-
  Study screen reached from Home falsely breadcrumbs Library and Escapes there
  instead
status: To Do
assignee: []
created_date: '2026-08-09 18:38'
labels:
  - home
  - study
  - navigation
  - ux-copy
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-2854 gave StudyScreen a truthful navigation identity (breadcrumb + Escape) for the ONE origin it considered: Library's staging canvas. Home's 'Review flashcards' button (Flashcards due: N row) navigates straight to the real Study screen at its flashcards section -- a genuinely honest first click, unlike the pre-task-2854 Library rail -- but StudyScreen's breadcrumb is hardcoded to 'Library ▸ Study' / 'Esc: back to Library', and its only Escape binding (action_study_back_to_library) always posts NavigateToScreen(TAB_LIBRARY, ...) regardless of where the user actually came from. A user who reached Study from Home and presses Escape lands on Library's Study-decks staging canvas, not back on Home -- the exact 'no hidden mystery navigation' violation task-2854 fixed for the Library origin, now present for the Home origin. Live-verified on dev @ 4d0232358 (task-3021's audit): seeded a due flashcard, clicked Home's 'Review flashcards', landed on Study with nav bar boxing nothing and header reading 'Library ▸ Study / Esc: back to Library'; pressing Escape boxed '⌃3 Library' and showed the Library 'Study decks' staging canvas -- Home was never involved in the return path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 StudyScreen's breadcrumb and Escape destination reflect the screen the user actually arrived from (Home vs Library), not a hardcoded Library assumption
- [ ] #2 Reaching Study via Home's Review flashcards and pressing Escape returns to Home, not to a Library staging canvas the user never visited
- [ ] #3 Existing Library-origin round trip (task-2854) is unaffected: Escape from Study reached via the Library staging canvas still returns there
<!-- AC:END -->
