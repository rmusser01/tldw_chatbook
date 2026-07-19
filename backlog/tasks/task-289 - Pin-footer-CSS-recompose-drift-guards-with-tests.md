---
id: TASK-289
title: Pin footer CSS/recompose drift guards with tests
status: Done
assignee:
  - '@claude'
created_date: '2026-07-17 15:17'
updated_date: '2026-07-17 20:34'
labels:
  - ux
  - testing
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-264's final review left two comment-only invariants unguarded: (1) AppFooterStatus.DEFAULT_CSS must stay a faithful subset of the live bundle source (css/components/_widgets.tcss 'Window Footer Widget' block) — drift silently diverges harness geometry from production; (2) PersonasScreen's footer hints use the non-persisting set_shortcut_context path, which is only safe while PersonasScreen has no recompose path — a future recompose=True reactive would silently drop its hints. Make both a red test instead of a comment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test fails when AppFooterStatus.DEFAULT_CSS core rules diverge from the _widgets.tcss footer block
- [x] #2 A test fails if PersonasScreen gains a screen-level recompose path while still using the non-persisting footer registration
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. AST-based guard: any call with recompose=True in personas_screen.py while it still uses the non-persisting set_shortcut_context path fails. 2. CSS-subset guard: parse AppFooterStatus.DEFAULT_CSS and the _widgets.tcss footer block; every DEFAULT_CSS declaration must appear in the matching bundle-source block (and in the built tldw_cli_modular.tcss, catching forgot-to-rebuild). 3. Mutation-check both tests actually fail when the invariant breaks; run the footer suites.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two drift guards in Tests/UI/test_screen_footer_hints.py (the footer contract file; AST-guard style precedent per test_console_workbench_parity_matrix.py). CSS: parsed-declaration subset comparison of AppFooterStatus.DEFAULT_CSS against BOTH the _widgets.tcss footer section and the built tldw_cli_modular.tcss (catches forgot-to-rebuild); section markers use the full comment text (bare markers sit inside comments — splitting there leaves a dangling */ corrupting the first selector). Personas: AST guard — recompose=True inside BaseAppScreen subclasses OR module-level .refresh(recompose=True) helpers (library_screen precedent), disarmed only by an actual .register_footer_shortcuts() call (AST match, not substring — a comment cannot disarm it); child-widget classes excluded (their recompose never touches the screen footer). All rules mutation-verified in both directions. Review (sonnet): 'with fixes (optional)' — both Importants applied (substring hatch, module-level scope gap) + parser-limitation comment. Commits bab438cd/87e0f110 on claude/footer-drift-guards-289.
<!-- SECTION:NOTES:END -->
