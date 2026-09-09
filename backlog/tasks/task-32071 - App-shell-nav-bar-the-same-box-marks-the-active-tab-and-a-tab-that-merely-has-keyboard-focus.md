---
id: TASK-32071
title: >-
  App shell nav bar: the same box marks the active tab and a tab that merely has
  keyboard focus
status: Done
assignee: []
created_date: '2026-09-08 18:26'
updated_date: '2026-09-08 20:04'
labels:
  - app-shell
  - ux
  - keyboard
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tabbing out of a Library canvas into the nav bar boxes '⌃1 Home' exactly as when Home is the active screen, so a keyboard user cannot tell 'focused' from 'active' and may press Enter expecting a Library action. Outside Library but surfaced by its Tab order. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 22.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A focused nav tab is visually distinct from the active tab by shape, not colour alone
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The finding did not reproduce. Driven live at 235x52 with capture-pane -e (caps/02-navbar-focus-vs-active-colour.txt): Tab into the nav bar paints the FOCUSED tab as an underlined label on the dark focus ground rgb(16,49,75) with no border, while the ACTIVE tab carries a drawn box (round rgb(1,120,212)) on the bright rgb(0,101,190) ground. The two differ by SHAPE (box vs no box) before colour enters it, which is what AC#1 asks for, so I made no style change: adding one would have been a change nobody needs and a risk to the app-wide focus grammar that Tests/UI/test_non_obscuring_focus_contract.py pins (NavigationButton:focus must keep 'text-style: bold underline').

Worth recording for whoever revisits this: an early attempt to pin the defect in a synthetic harness passed vacuously, because MainNavigationBar's own .nav-button rules live in the widget-defaults tier (widget_defaults_scoped.tcss, tie-breaker below every other default source) and lose to NavigationButton's DEFAULT_CSS 'border: none'. The nav bar's real paint can only be measured on the mounted bar under the app stylesheet, or live.
<!-- SECTION:NOTES:END -->
