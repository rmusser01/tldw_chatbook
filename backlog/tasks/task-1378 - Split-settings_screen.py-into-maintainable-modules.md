---
id: TASK-1378
title: Split settings_screen.py into maintainable modules
status: To Do
assignee: []
created_date: '2026-08-06 00:09'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-critique structural finding: settings_screen.py is ~10.8k lines in one file — a maintainability risk where the three-pane contract and commit models (ADR-033) are enforced only by convention. Split the screen into cohesive modules (e.g. category renderers, commit-model/draft machinery, providers & models section, advanced config) without behavior change. This is a refactor: preserve all existing tests green and keep the public screen contract intact.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 settings_screen.py reduced to a coordinator with category/section logic in sibling modules,No behavior change: full settings test suites pass unmodified,ADR-033 commit-model labeling preserved (copy constants stay importable by widgets),Ruff/lint clean on new module layout
<!-- AC:END -->
