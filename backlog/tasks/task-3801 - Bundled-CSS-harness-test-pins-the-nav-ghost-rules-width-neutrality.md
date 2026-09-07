---
id: TASK-3801
title: Bundled-CSS harness test pins the nav ghost rule's width-neutrality
status: Done
assignee:
  - '@claude'
created_date: '2026-08-09 15:51'
updated_date: '2026-08-09 18:19'
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
- [x] #1 A test exercises MainNavigationBar under a harness that actually loads the app's CSS_PATH bundle (mirroring InspectorAppWithBundledCSS), not just DEFAULT_CSS
- [x] #2 That test fails if the bundle tier's .nav-button.nav-button-clip-ghost:disabled rule (or any future bundle-tier override of it) reintroduces a non-zero border/padding/margin delta between a button's ghosted and un-ghosted box model
- [x] #3 The test passes against the current, already-fixed bundle rule
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the MCP inspector precedent (InspectorAppWithBundledCSS / test_disabled_action_buttons_stay_legible_with_bundled_css) and the DEFAULT_CSS-tier sibling (test_ghosting_a_button_never_reflows_the_strip in test_master_shell_navigation.py) to mirror both the bundled-CSS harness shape and the geometry-comparison shape.
2. Add a harness App subclass with CSS_PATH set to the real generated tldw_cli_modular.tcss, mounting MainNavigationBar with auto-ghosting disabled (mirroring _NoAutoGhostBar) so the test's manual ghost/un-ghost isn't raced by the widget's own settle pass.
3. Write the test: capture a button's region before ghosting, ghost it by hand (add_class + disabled=True), capture region again, assert equality.
4. Verify the test passes against the current (already-fixed) bundle rule.
5. Mutation evidence: reintroduce a box-model property (border) into the SOURCE tcss (css/components/_navigation.tcss), regenerate the bundle via build_css.py, confirm the test goes RED (with -B / no stale pycache) and reproduces the exact task-3225 shape (2 cells wider). Then restore the source, regenerate the bundle again, and verify check_bundle_sync reports clean.
6. Run the test 5x on the restored, correct bundle.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Filed, not fixed, per the fix/library-polish-batch final review's ask: name the incident (round 4's border -> 2-cell reflow -> drift-back) and cite the MCP-inspector InspectorAppWithBundledCSS precedent.

**Implementation (2026-08-09).** Added `test_ghost_rule_is_width_neutral_under_the_bundled_stylesheet` to `Tests/UI/test_master_shell_navigation.py`, right after its DEFAULT_CSS-tier sibling `test_ghosting_a_button_never_reflows_the_strip`. New `_NavAppWithBundledCSS` (`App` subclass, `CSS_PATH` = the real generated `tldw_cli_modular.tcss`, mirroring `InspectorAppWithBundledCSS`) mounts a `_NoAutoGhostBarWithBundledCSS` (a `MainNavigationBar` subclass whose `_ghost_clipped_buttons` no-ops, same reasoning as the sibling's `_NoAutoGhostBar`). The test snapshots `#nav-workflows`' region, manually ghosts it (`add_class("nav-button-clip-ghost")` + `disabled = True`), and asserts the region is unchanged.

**First attempt was vacuous and caught by mutation, not by inspection.** The first version of this test used a plain `MainNavigationBar(active="home")` (no auto-ghost override) inside the bundled-CSS App. It PASSED against a deliberately reintroduced `border: solid $background;` mutation in `css/components/_navigation.tcss` -- which should have gone red. Root cause: at the test's 200-col width `#nav-workflows` genuinely does not straddle the viewport, so the widget's own `_ghost_clipped_buttons` settle pass silently un-ghosted the button again (`add_class`/`disabled=True` were both reverted) before the assertion read the region -- a probe of `victim.classes`/`victim.disabled` after the manual ghost call showed the ghost class and disabled flag both gone. This is the exact race `_NoAutoGhostBar` in the DEFAULT_CSS-tier sibling test already exists to prevent; the bundled-CSS harness needed the same override and didn't have it. Adding `_NoAutoGhostBarWithBundledCSS` fixed it.

**Mutation evidence (after the fix).** Edited `css/components/_navigation.tcss`'s `.nav-button.nav-button-clip-ghost:disabled` to add `border: solid $background;`, regenerated the bundle (`python tldw_chatbook/css/build_css.py`), confirmed the mutated rule landed in `tldw_cli_modular.tcss`, and ran the new test with `-B` (no stale bytecode, per the "green result is not evidence" lesson). It failed exactly as expected, reproducing the task-3225 incident's own shape almost to the cell: `Region(x=91, y=0, width=14, ...) -> Region(x=91, y=0, width=16, ...)` -- 2 cells wider, ghosted vs un-ghosted, under the real bundle. Restored `_navigation.tcss` to its original content, regenerated the bundle again (only the `Generated:` timestamp line differs from the pre-mutation committed bundle -- confirmed via `git diff`), and ran `python tldw_chatbook/css/check_bundle_sync.py`: "CSS bundle reproduces from its source modules." (exit 0). Ran the new test 5/5 on the restored, correct bundle; also ran the full `test_master_shell_navigation.py` (33 tests) and `test_css_bundle_sync_guard.py` (3 tests) afterward with no regressions.

**Files changed:** `Tests/UI/test_master_shell_navigation.py` (new imports, `_BUNDLED_CSS_PATH`, `_NoAutoGhostBarWithBundledCSS`, `_NavAppWithBundledCSS`, `test_ghost_rule_is_width_neutral_under_the_bundled_stylesheet`); `tldw_chatbook/css/tldw_cli_modular.tcss` (regenerated via `build_css.py` after the mutation round-trip -- timestamp-only diff versus the pre-task committed bundle, `check_bundle_sync` clean). No change to `css/components/_navigation.tcss` (restored to its original content) or any other production source.
<!-- SECTION:NOTES:END -->
