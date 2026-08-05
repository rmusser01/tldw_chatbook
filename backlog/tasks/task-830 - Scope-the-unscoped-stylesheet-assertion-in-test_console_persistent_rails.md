---
id: TASK-830
title: Scope the unscoped stylesheet assertion in test_console_persistent_rails
status: Done
assignee: []
created_date: '2026-07-26 22:07'
updated_date: '2026-07-27 20:26'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Obsolete, not fixed: the file this task describes no longer exists.

Tests/UI/test_console_persistent_rails.py was deleted on dev by 9d036b902 (TASK-650: remove legacy Chat root state), taking test_generated_console_stylesheet_includes_rail_rules and its unscoped line-278 assertion with it. There is nothing left to scope.

Verified while closing: no surviving test guards Console rail selectors against a heavy border, and no Console rail selector in the generated bundle currently carries 'border: thick'. So the invariant the assertion protected still holds, but is now unguarded. Deliberately not re-adding a guard here -- the containing file was removed on purpose as part of a legacy-subsystem retirement, and re-introducing a rule for it would be re-adding weight that removal was meant to shed. If a heavy border ever lands on a Console rail selector, the right response is a scoped check next to the rules it protects, not a revival of this file.
<!-- SECTION:NOTES:END -->
