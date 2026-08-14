---
id: TASK-15991
title: 'ScraperBuilderWindow cannot be opened: FormBuilder.create_switch does not exist'
status: To Do
assignee: []
created_date: '2026-08-14 01:10'
labels:
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`UI/ScraperBuilderWindow.py:322` calls `self.form_builder.create_switch(...)`, but `FormBuilder` (Widgets/form_components.py) has no such method, so opening the window dies in compose with AttributeError. Pre-existing on dev well before the CSS consolidation (confirmed absent at `6b57458b8` too) — the screen is unopenable, which also means any historical evidence that 'measured' it opening measured a crash. Found during the TASK-15450 CSS-consolidation review (PR #1616, merged `c3ed2854a`); evidence in the session review record and `Docs/Design/2026-08-11-input-latency-audit.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening ScraperBuilderWindow composes without raising
- [ ] #2 A mounted regression test pins the open (born-red against the current crash)
- [ ] #3 Notes state whether create_switch was added to FormBuilder or the call sites changed, and why
<!-- AC:END -->
