---
id: TASK-830
title: Scope the unscoped stylesheet assertion in test_console_persistent_rails
status: To Do
assignee: []
created_date: '2026-07-26 22:07'
labels:
  - tech-debt
  - css
  - tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
test_console_persistent_rails.py::test_generated_console_stylesheet_includes_rail_rules has been failing on dev for some time. Line 278 asserts globally over the entire stylesheet that 'border: thick $ds-action-focus;' appears nowhere. It is the only unscoped assertion in a function whose every other check uses _css_block(css, selector). Unrelated RAG-settings work later added that declaration to .settings-rag-profile-modal and .settings-library-rag-starter-panel, tripping a Console rail test from Settings. The Lab frame work had to route around this red test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The assertion is scoped to the Console rail selectors it was written to protect, matching its sibling checks,Adding an unrelated 'border: thick' rule elsewhere in the stylesheet no longer fails this test,The test still fails if a Console rail selector regains a heavy border,test_console_persistent_rails passes on dev with no other change
<!-- AC:END -->
