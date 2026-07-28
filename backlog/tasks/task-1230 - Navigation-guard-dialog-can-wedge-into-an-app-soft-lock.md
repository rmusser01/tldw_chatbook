---
id: TASK-1230
title: 'Navigation-guard dialog can wedge into an app soft-lock'
status: To Do
assignee: []
created_date: '2026-07-28 09:30'
labels: [console, navigation, critical, uat]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expert UAT (Docs/superpowers/qa/fleet-ux-expert-review-2026-07-28, F1): with a busy fleet, navigate (guard dialog) -> Stay (works) -> navigate again -> click Leave at its rendered coordinates -> no effect, and thereafter the dialog answers to nothing (both buttons via 12-point click sweep, Escape, Tab, Enter, nav-bar clicks all inert). Only Ctrl+Q escapes. App log empty; mechanism undetermined (hypotheses in the report: post-confirm navigation failure leaving a painted-but-dead overlay; or a push_screen_wait interleaving the existing race test does not cover — that test queues a second NavigateToScreen message, not a Stay-then-renavigate-then-Leave human sequence). Task-1142 rhyme: the guard's tests click the button widget; nothing clicks rendered coordinates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The scripted repro (Stay, renavigate, Leave-by-coordinates) navigates cleanly with runs cancelled; no input-inert state is reachable.
- [ ] #2 The dialog is keyboard-operable (documented keys) and Escape maps to Stay.
- [ ] #3 A coordinate-honest regression test drives the full human sequence at rendered positions.
<!-- AC:END -->
