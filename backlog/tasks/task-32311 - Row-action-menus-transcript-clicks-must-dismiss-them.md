---
id: TASK-32311
title: 'Row action menus: transcript clicks must dismiss them'
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-11 04:12'
updated_date: '2026-09-11 04:13'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clicking the transcript (most of the screen) left the conversation/workspace action menu mounted: the screen-level outside-click dismissal returns early for transcript targets, and the transcript's own pointer cleanup only knew its selection UI. The transcript's pointer press now folds both row-menu registries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A transcript press folds an open conversation action menu
- [x] #2 A transcript press folds an open workspace action menu
- [x] #3 Composer and rail outside-clicks keep folding the menus (regression)
- [x] #4 Menu-internal clicks (border/padding/buttons) still never fold it (regression)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Red tests in both menu suites: transcript press (pilot.click on #console-native-transcript) folds the open menu.
2. ConsoleTranscript.on_mouse_down folds the conversation + workspace action-menu registries (restore_focus=False; function-local imports per ADR-097).
3. Suites, lint, PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: the screen's outside-click dismissal (`_dismiss_console_selection_menus_outside_transcript`) returns early for any transcript-ancestor target -- "the transcript owns its in-area interaction" -- but the transcript's own pointer cleanup only knew its text-selection UI and the More menu, so presses on the transcript (the largest region of the Console) left the rail row-action menus floating. The transcript's `on_mouse_down` (the same earliest-seam precedent the More menu's press dismissal uses) now folds both row-menu registries with no opener focus-restore; menu-internal clicks are unaffected because the action menus mount on the screen, never inside the transcript.

Verification: new red-then-green tests in both menu suites (19 and 20 green); the compositor-tint and four More-menu transcript-suite failures are pre-existing on clean origin/dev (verified by stash-baseline: 7/7 fail without this change). Live-driver note: the PTY synthetic driver can no longer OPEN the rail menus on current dev (rail pointer machinery eats synthetic press pairs) and the seeded workspace tree listing is transient after re-sync, so the transcript-click dismissal is mounted-verified; composer-click dismissal of the same seam was live-verified in earlier UATs.
<!-- SECTION:NOTES:END -->
