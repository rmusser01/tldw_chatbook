---
id: TASK-102
title: Fix dead .--highlight selectors across the stylesheet
status: To Do
assignee: []
created_date: '2026-06-11 20:28'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Textual sets -highlight (single dash) on highlighted ListItems; several bundle rules use .--highlight and never match. Sweep and fix non-personas occurrences.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No dead --highlight selectors remain in non-personas TCSS source files,Bundle regenerated after fixes
<!-- AC:END -->
