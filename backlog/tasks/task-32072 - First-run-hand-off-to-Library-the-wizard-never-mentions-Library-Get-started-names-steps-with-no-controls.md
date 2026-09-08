---
id: TASK-32072
title: >-
  First-run hand-off to Library: the wizard never mentions Library; Get started
  names steps with no controls
status: In Progress
assignee: []
created_date: '2026-09-08 18:26'
updated_date: '2026-09-08 19:41'
labels:
  - library
  - onboarding
  - ux
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The setup Summary offers 'Explore Home' and never says where content lives; Get started reads '1 Add · 2 Find · 3 Use' but only the Add step has controls, so a first-timer has no path from 'I imported something' to 'find it' and 'use it in Console'. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 23.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The wizard Summary offers an action that lands in Library's Import (for example 'Add your first document')
- [ ] #2 Each Get started step is a live control that unlocks in sequence (Import a file, Find it, Use it in Console)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing tests: the wizard Summary never mentions Library; Get started names three steps with one control.
2. Add 'Add your first document' to the Summary, exiting on TAB_LIBRARY; app.py maps that route to Library's Import canvas.
3. Replace the '1 Add - 2 Find - 3 Use' line with three sequenced controls plus one reason line.
<!-- SECTION:PLAN:END -->
