---
id: TASK-998
title: >-
  Watchlists first run shows seven empty cards and dead-end Inspector guidance
status: To Do
assignee: []
created_date: '2026-07-27 22:00'
labels:
  - watchlists
  - ux
  - uat
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two first-run problems seen together in the clean-profile UAT (`origin/dev` `dbbb7de84`, 235x52).

**The Overview region is seven empty bordered cards.** It is the largest region on the screen and the first thing a new user sees, and it contains nothing. This was recorded during the original design work as one of the screen's defects and has never been addressed.

**The Inspector's empty state is a dead end.** It reads "Select a source, run, item, rule, or notification to see actions." — but on first run none of those exist, so the guidance names five things the user cannot do. The right-hand rail is a third of the screen and spends it telling a new user to do something impossible.

Together these mean a first-time user's screen is mostly empty boxes and instructions that do not apply. The tree's `New` and the centre's `Create source` are the only real affordances, and neither is where the eye goes.

Worth treating as one piece of work: it is the same question — what should this screen say when there is nothing in it yet.

Evidence: `Docs/superpowers/qa/watchlists-uat-2026-07-27/notes.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Overview region shows real content, or is not shown, on a profile with no watchlists and no sources
- [ ] #2 The Inspector's first-run text points at an action the user can actually take
- [ ] #3 A first-run capture from a clean profile is attached, showing no empty bordered cards
- [ ] #4 The populated states of both regions are unchanged
<!-- AC:END -->
