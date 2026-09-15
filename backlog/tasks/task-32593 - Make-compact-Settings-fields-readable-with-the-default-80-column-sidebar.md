---
id: TASK-32593
title: Make compact Settings fields readable with the default 80-column sidebar
status: Done
assignee:
  - '@codex'
created_date: '2026-09-15 00:18'
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
Shared Settings rows reserve 24 columns for labels at 80-column width. Providers model and endpoint fields have only about seven editable content cells, while Network policy names truncate to Verif or Cust. Users can type values but cannot comfortably inspect them in the default layout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With the normal sidebar visible at 80x24, Providers Model and Endpoint and Network CA bundle path have at least 20 editable content columns.
- [x] #2 The closed Network Certificate verification selector displays each complete selected policy name at 80 columns.
- [x] #3 Field labels remain complete and unambiguous, and keyboard focus stays visible through row adaptation and 120-to-80-to-120 resizing in both themes.
- [x] #4 Provider values and typed CA paths survive resizing; Network validation still reports a specific actionable error for an invalid path.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: scoped responsive-layout repair under existing design-token and component-pattern decisions; no behavior or persistence boundary changes.
1. Add production-styled mounted regressions for Providers/Network at 80 columns, full TLS policy paint, and dark/light resize retention/focus. Confirm failure before implementation.
2. Reuse the existing Settings compact-workbench class to stack affected field labels above controls with existing tokens, preserving wide layouts and values.
3. Coordinate generated CSS rebuild with integration owner, run dedicated and adjacent targeted tests, and record evidence and remaining native verification needs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the compact Providers Connect and Network field-row repair with existing Settings classes and tokens. At 100 columns or fewer, direct rows place complete labels above full-width controls; wider layouts retain the 24-column label column. The closed Network policy selector releases one padding cell so all policy names fit at 80 columns. Settings behavior, validation, persistence and token values are unchanged.

Changed files: css/features/_settings.tcss, generated styles, Tests/UI/test_settings_compact_fields.py and backlog/docs/component-patterns.md. Six new production-styled mounted regressions failed before the repair and pass afterward: real typing, at least 20 editable cells, complete labels and all three policy names, visible focus, retained values/focus over 120-to-80-to-120 resizing in both themes, and an actionable invalid-CA-path notification. Ten adjacent Settings tests pass. The suffix-paint assertion presses End after resizing; value retention is checked independently.

Final integration verification: 34 component/token/bundle/budget/gallery-snapshot checks pass, generated CSS reproduces, boot CSS is 613053/634050 bytes, and fatal Ruff, test formatting and full design-branch whitespace checks pass. Native scratch-profile checks confirm Providers Model focus and Endpoint rendering at 80x24, restoration at 120x45, and the Network policy/path at 80x24 in both themes. No full test suite or external model request was run.

Evidence and remaining review boundaries: Docs/superpowers/reports/2026-09-14-component-audit-fixes.md and Docs/superpowers/qa/2026-09-14-component-fixes/. Native 120-column Network still truncates the default policy when the Scope Inspector is visible; this existing wide layout is explicitly retained for a later dedicated review. All compact-task acceptance criteria are met.

ADR required: no new ADR. Existing backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/161-component-pattern-library.md govern the scoped repair.
<!-- SECTION:NOTES:END -->
