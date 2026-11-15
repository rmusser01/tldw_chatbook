---
id: TASK-1378
title: Split settings_screen.py into maintainable modules
status: In Progress
assignee: []
created_date: 2026-08-06 00:09
updated_date: '2026-11-15T10:06:00Z'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-critique structural finding: settings_screen.py is ~10.8k lines in one file — a maintainability risk where the three-pane contract and commit models (ADR-303) are enforced only by convention. Split the screen into cohesive modules (e.g. category renderers, commit-model/draft machinery, providers & models section, advanced config) without behavior change. This is a refactor: preserve all existing tests green and keep the public screen contract intact.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 settings_screen.py reduced to a coordinator with category/section logic in sibling modules,No behavior change: full settings test suites pass unmodified,ADR-303 commit-model labeling preserved (copy constants stay importable by widgets),Ruff/lint clean on new module layout
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-11-15T10:06:00Z

---

TASK-1378.3 completed the bounded Speech category ownership migration after Network and Providers & Models. Targeted mounted checks and independent review are recorded in Docs/superpowers/qa/2026-11-14-textual-pattern-followups/verification.md. Remaining category/coordinator decomposition and the parent acceptance scope stay open; no full Settings sweep or parent completion is claimed.

---
<!-- COMMENTS:END -->
