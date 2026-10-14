---
id: TASK-35014
title: Settings drops the controls nothing reads and corrects copy about when changes apply
status: To Do
created_date: 2026-10-14 17:10
labels:
- settings
- ux
- copy-honesty
- settings-ux-review-2026-10-14
dependencies:
- TASK-33007
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Appearance offers two controls the app never reads: Palette limit (themes) and Smooth scrolling. The field guide even promises that the palette limit limits how many themes appear in the command palette, so users change them and nothing happens.

Other copy is false about when or how changes apply:
- Splash Screen's State line says changes take effect as you make them, but splash settings only show at the next launch.
- The Guardian and Dreams pages say 'Settings mode: Immediate global briefing-gate control' and 'Writes allowed: Yes - global gate only'. That wording was copied from Schedules and does not describe them.
- The Domain Defaults rail toggle says its pages are read-only, though six of its thirteen are editable.

Each of these tells the user something false about what a control does or when it takes effect. This task removes the two inert controls without touching the values users have already saved, and corrects the false copy.

Plan: `backlog/docs/plan-2026-10-14-settings-save-honesty.md`. Findings closed: D7, D-shell-design-M2 (Settings UX review, 2026-10-14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Appearance no longer shows Palette limit (themes) or Smooth scrolling. Neither appears in the field guide, the Appearance summary, '/' search results or the shipped config template.
- [ ] #2 A profile whose config already holds palette_theme_limit or smooth_scrolling opens Appearance and saves a change without error, and both keys keep their values.
- [ ] #3 Splash Screen's State line says its changes are saved as you make them and shown at the next launch.
- [ ] #4 Guardian's and Dreams' 'How this page works' rows describe what that page controls and writes. Schedules keeps its briefing-gate wording.
- [ ] #5 The Domain Defaults rail toggle no longer says that all of its pages are read-only.
- [ ] #6 Docs/User_Guide/settings.md's Appearance, Quirks, Splash, Guardian and Dreams text matches the app, including that Animations is read by character expression playback.
- [ ] #7 settings_screen.py is no longer than it was before this change, and every new test runs and passes in the UI PR gate lane.
<!-- AC:END -->
