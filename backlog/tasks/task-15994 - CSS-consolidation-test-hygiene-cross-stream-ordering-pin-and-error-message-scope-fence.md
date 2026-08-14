---
id: TASK-15994
title: 'CSS consolidation test hygiene: cross-stream ordering pin and error-message scope fence'
status: To Do
assignee: []
created_date: '2026-08-14 01:10'
labels:
  - tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two weaknesses in `Tests/UI/test_widget_css_consolidation.py` found in review. (1) `test_base_class_blocks_precede_their_subclasses` (~:222-256) builds its order from the CONCATENATION of the self and scoped sheets, but the two streams carry different tie-breakers (0 vs -1,000,000) that decide precedence regardless of position — so a cross-stream index comparison pins nothing, and the test also only inspects direct syntactically-named bases (a grandparent inversion passes). (2) The mounted dialog test (~:414) fences its scope with `"clear" not in str(raised)` — a substring match on an error message; the load-bearing StylesheetParseError assertion runs first so the pin holds, but the fence is fragile (dissolves entirely once TASK-15992 fixes the underlying dialog crash). Found during the TASK-15450 CSS-consolidation review (PR #1616, merged `c3ed2854a`); evidence in the session review record and `Docs/Design/2026-08-11-input-latency-audit.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The ordering test compares only within a stream (or asserts the tie-breaker relation directly) and covers transitive bases
- [ ] #2 The mounted-dialog test's exception tolerance is either removed (after TASK-15992) or keyed on exception type and site, not message substring
- [ ] #3 Both tests still born-red against a seeded violation
<!-- AC:END -->
