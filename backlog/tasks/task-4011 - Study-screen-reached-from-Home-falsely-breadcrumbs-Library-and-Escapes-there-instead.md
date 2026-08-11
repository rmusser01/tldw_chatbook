---
id: TASK-4011
title: >-
  Study screen reached from Home falsely breadcrumbs Library and Escapes there
  instead
status: Done
assignee:
  - '@claude'
created_date: '2026-08-09 18:38'
updated_date: '2026-08-10 21:41'
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
- [x] #1 StudyScreen's breadcrumb and Escape destination reflect the screen the user actually arrived from (Home vs Library), not a hardcoded Library assumption
- [x] #2 Reaching Study via Home's Review flashcards and pressing Escape returns to Home, not to a Library staging canvas the user never visited
- [x] #3 Existing Library-origin round trip (task-2854) is unaffected: Escape from Study reached via the Library staging canvas still returns there
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add STUDY_ORIGINS + HandoffChannel.STUDY_ORIGIN to the existing pending-handoff seam (study_scope_models.py, pending_handoff_store.py) -- reuse the same single-slot channel pattern open_study_screen already uses for scope/initial-section; no new navigation-history mechanism.
2. TldwCli.open_study_screen gains origin: Optional[str] = None (None -> clear channel -> Library default, preserving every existing caller); open_home_flashcards_review passes origin='home'.
3. StudyScreen claims the origin in __init__ (screens are constructed fresh per navigation), derives breadcrumb title/subtitle, footer Esc hint, and the Escape target from it: home -> NavigateToScreen(TAB_HOME); library -> existing LIBRARY_NAV_CONTEXT_MODE 'study' staging-canvas return (task-2854 path untouched).
4. TDD: RED headless full-app test reproducing the observed lie (Home entry -> 'Library ▸ Study' breadcrumb + Escape lands on Library) before the fix; companion Library-origin round-trip test pins AC#3.
5. Live tmux verification of both round trips; update User Guide stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Threaded the actual origin through the EXISTING pending-handoff seam rather than any new nav-history mechanism: new single-slot HandoffChannel.STUDY_ORIGIN (normalized against STUDY_ORIGINS = {home, library} in study_scope_models.py), staged by TldwCli.open_study_screen's new origin kwarg (None clears -> Library default, so every existing caller is unchanged) and passed only by open_home_flashcards_review(origin='home'). StudyScreen claims it in __init__ (screens are built fresh per navigation) and derives the breadcrumb title/subtitle ('Home ▸ Study' / 'Esc: back to Home.'), the footer esc hint (_study_footer_shortcuts()), and the Escape target: action_study_back (renamed from action_study_back_to_library; binding description now origin-neutral 'Back') posts NavigateToScreen(TAB_HOME) for a home origin and keeps task-2854's Library staging-canvas return for everything else. TDD: Tests/UI/test_study_origin_navigation.py -- RED at HEAD reproduced the exact lie (title 'Library ▸ Study' from a Home entry; Escape landed on Library), plus a Library-origin pin (green before AND after) and a no-sticky-origin test (a later unlabelled entry breadcrumbs Library again). Live tmux (scratch profile sdd_hat1, seeded one due flashcard via sqlite): Home -> Review flashcards -> 'Home ▸ Study' + 'esc back to Home' footer -> Escape -> Home; Library -> Study decks -> Continue in Study -> 'Library ▸ Study' + 'esc back to Library' -> Escape -> Study decks staging canvas. Docs: home.md (quirk removed, control table updated, restamped), library.md (Home-origin parenthetical + stamp). Files: study_scope_models.py, pending_handoff_store.py, app.py, study_screen.py, library_screen.py (comment), Docs/User_Guide/{home,library}.md, Tests/UI/test_study_origin_navigation.py.
<!-- SECTION:NOTES:END -->
