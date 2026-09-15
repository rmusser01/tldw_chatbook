---
id: TASK-32594
title: Correct the contained inline-form example in the gallery and catalog
status: To Do
assignee: []
created_date: '2026-09-15 00:19'
labels:
  - design-system
  - ui
  - audit
dependencies: []
references:
  - Docs/superpowers/reports/2026-09-14-component-first-ui-audit.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The new Temperature exemplar places an 11-cell label beside an input requesting the full row width, clipping its right border at both 80 and 120 columns. The catalog also describes an inline composition with a plain Container. The canonical examples should demonstrate a composition that downstream features can safely copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The gallery Temperature label and input, including their borders, stay within their parent at 80 and 120 columns in dark and light themes.
- [ ] #2 The documented executable form skeleton and rendered gallery use the same clearly described inline or stacked layout contract.
- [ ] #3 Rest and focus states preserve the displayed value and complete field edges; a targeted containment/render regression covers the example.
- [ ] #4 Any new layout value or class follows the existing token, ownership, catalog and gallery governance.
<!-- AC:END -->
