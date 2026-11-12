---
id: TASK-34883
title: TemplateSelectorDialog._load_templates imports non-existent Evals symbol
status: Done
assignee: []
created_date: '2026-09-19 21:21'
updated_date: '2026-09-20 06:04'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pre-existing bug: TemplateSelectorDialog._load_templates imports a symbol tldw_chatbook.Evals no longer exports, so the template list is always empty. Documented in .superpowers/wave2-report.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Template selector lists available templates
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in PR #2751 (commit 3aa76926af): imports from Evals.eval_templates with full spec carried through (Qodo fix 9ecac0898c). Merged to dev as 9929b48be52.
<!-- SECTION:NOTES:END -->
