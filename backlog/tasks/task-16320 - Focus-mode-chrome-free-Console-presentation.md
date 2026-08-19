---
id: TASK-16320
title: 'Focus mode: chrome-free Console presentation'
status: To Do
assignee: []
created_date: '2026-08-16 13:34'
updated_date: '2026-08-16 14:05'
labels:
  - ui
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a 'focus mode' that presents only the Console content — message stream, composer, and a one-line status bar — hiding the MainNavigationBar and workbench header. Recreates a claude-code/codex-style UI for zen coding on desktop and phone use over --serve without fine pointer/touch affordances. Zen-not-kiosk with one navigation rule: any navigation to a non-chat route exits focus; ctrl+shift+f re-enters from anywhere.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Console hides MainNavigationBar and DestinationHeader while focus mode is active; the one-line AppFooterStatus status bar remains visible
- [ ] #2 --focus CLI flag and [general] focus_mode config launch straight into the chrome-free Console (first-run onboarding still wins)
- [ ] #3 ctrl+shift+f is an app-level toggle with no conflicts with existing bindings; toggling on from a non-chat screen navigates to the Console and enters focus
- [ ] #4 Any navigation to a non-chat route (destination hotkey or palette) exits focus mode and the destination mounts with normal chrome
- [ ] #5 Footer shortcut context advertises the focus toggle truthfully in both states (focus / exit focus)
- [ ] #6 Default presentation is unchanged when focus mode is off (existing console contract tests pass)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Design approved + reviewed (2026-08-16). Spec: Docs/superpowers/specs/2026-08-16-focus-mode-design.md — ADR: backlog/decisions/067-focus-mode-chrome-free-console.md — Implementation plan: Docs/superpowers/plans/2026-08-16-focus-mode.md (7 TDD tasks). Ready to execute.
<!-- SECTION:NOTES:END -->
