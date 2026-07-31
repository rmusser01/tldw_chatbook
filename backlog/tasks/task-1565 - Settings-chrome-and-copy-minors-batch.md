---
id: TASK-1565
title: 'Settings: chrome and copy minors batch'
status: To Do
assignee: []
created_date: '2026-07-31 02:00'
labels: [settings, ux, P3]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Grouped P3 findings from the 2026-07-31 critique, each small but shipped:
- "Tokens: -- |" footer fragment intermittent across categories, always with a dangling pipe.
- Breadcrumb advertises "accounts"; no accounts category exists.
- Sidebar does not auto-scroll to the selected category (selection can be entirely off-viewport).
- "Duration 1.5" / "Animation 1.0" unitless on Splash Screen.
- RAG: "Clone it, then Set active, to edit." comma cadence; red Delete on a read-only built-in profile; description flyout overlaps the rail's lower rows.
- Guided-path chips ("Providers | Console | Privacy") disagree with sidebar names ("Providers & Models", "Console Behavior", "Privacy & Security").
- Two permanently visible filter-help lines say the same thing.
- Splash gallery mixes label casing and shows raw ids ("TLDW CHATBOOK v2.0 (digital_rain)").
- "WIP" appears in user-facing copy in six categories.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed item fixed or explicitly declined with a reason in notes.
- [ ] #2 No footer chrome renders dangling separators.
- [ ] #3 Category names are referenced identically everywhere they appear.
<!-- AC:END -->
