---
id: TASK-3801
title: Bundled-CSS harness test pins the nav ghost rule's width-neutrality
status: To Do
assignee: []
created_date: '2026-08-09 15:51'
labels:
  - tests
  - navigation
  - tech-debt
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-3225 (review round 4) fixed a real regression: the .nav-button-clip-ghost rule in main_navigation.py's DEFAULT_CSS once declared a four-edge solid border, which was NOT geometry-neutral -- it widened a ghosted nav button by 2 cells versus its un-ghosted rendering, silently reflowing the strip. The bundle-tier stylesheet has its own, separately-maintained copy of this rule (css/components/_navigation.tcss's .nav-button.nav-button-clip-ghost:disabled override, needed because CSS_PATH stylesheets outrank widget DEFAULT_CSS regardless of specificity -- see main_navigation.py's own docstring at the rule site). Only the DEFAULT_CSS-tier copy has a reflow-regression test (test_ghosting_a_button_never_reflows_the_strip in Tests/UI/test_master_shell_navigation.py, built on a bare App harness that never loads the CSS_PATH bundle); nothing exercises the bundle tier's rule the same way, so a reintroduced box-model property there would be invisible to any existing suite. This is the same class of gap the MCP inspector's InspectorAppWithBundledCSS harness (Tests/UI/test_mcp_inspector.py's test_disabled_action_buttons_stay_legible_with_bundled_css) was built to close for a different disabled-styling rule in the same bundle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test exercises MainNavigationBar under a harness that actually loads the app's CSS_PATH bundle (mirroring InspectorAppWithBundledCSS), not just DEFAULT_CSS
- [ ] #2 That test fails if the bundle tier's .nav-button.nav-button-clip-ghost:disabled rule (or any future bundle-tier override of it) reintroduces a non-zero border/padding/margin delta between a button's ghosted and un-ghosted box model
- [ ] #3 The test passes against the current, already-fixed bundle rule
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Filed, not fixed, per the fix/library-polish-batch final review's ask: name the incident (round 4's border -> 2-cell reflow -> drift-back) and cite the MCP-inspector InspectorAppWithBundledCSS precedent.
<!-- SECTION:NOTES:END -->
