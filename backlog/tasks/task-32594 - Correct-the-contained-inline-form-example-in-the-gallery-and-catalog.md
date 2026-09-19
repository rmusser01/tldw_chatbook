---
id: TASK-32594
title: Correct the contained inline-form example in the gallery and catalog
status: Done
assignee:
  - '@codex'
created_date: '2026-09-15 00:19'
updated_date: '2026-09-15 01:07'
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
- [x] #1 The gallery Temperature label and input, including their borders, stay within their parent at 80 and 120 columns in dark and light themes.
- [x] #2 The documented executable form skeleton and rendered gallery use the same clearly described inline or stacked layout contract.
- [x] #3 Rest and focus states preserve the displayed value and complete field edges; a targeted containment/render regression covers the example.
- [x] #4 Any new layout value or class follows the existing token, ownership, catalog and gallery governance.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md (existing)
Reason: repair the canonical inline example with existing sizing vocabulary; no new token, stylesheet ownership, or architecture.

1. Add a mounted regression for Temperature label/input containment, readable value and painted field edges at 80/120 columns, dark/light themes, rest/focus; confirm it fails on the existing full-row input.
2. Compose the gallery inline input with the existing w-fill utility; coordinate the matching Horizontal/form-input w-fill catalog skeleton and contract with the root reviewer.
3. Run focused regression and gallery snapshots, refresh intended snapshot changes, inspect captures, and record lint/verification evidence before closure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The Temperature inline field now composes form-input w-fill to take the width remaining beside its label. The catalog uses Horizontal and documents the same inline/stacked contract. This reuses existing ADR-150/161 vocabulary without new CSS or tokens.

The new production-styled mounted regression failed in all eight cases before the fix (11-cell overflow) and passes all eight afterward. It checks containment, non-overlap, value retention, focus and painted field edges at 80/120 columns in dark/light themes, rest/focus. Both gallery snapshots reproduce after the intended edge correction and blank-line normalization; all eight SVG/PNG captures were inspected. Final shared governance/token/bundle/budget/gallery checks: 34 passed. Fatal Ruff, test formatting and full branch whitespace checks pass; independent cross-review found no actionable issue.

Files: Widgets/pattern_gallery.py, Tests/UI/test_pattern_gallery_layout.py, gallery snapshot fixtures and backlog/docs/component-patterns.md. Evidence: Docs/superpowers/reports/2026-09-14-component-audit-fixes.md and Docs/superpowers/qa/2026-09-14-component-fixes/ (gallery tests, geometry and selected SVG/PNG captures). SVG whitespace normalization preserves parsed elements, attributes and visible text.

ADR required: no new ADR; existing backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/161-component-pattern-library.md apply.
<!-- SECTION:NOTES:END -->
