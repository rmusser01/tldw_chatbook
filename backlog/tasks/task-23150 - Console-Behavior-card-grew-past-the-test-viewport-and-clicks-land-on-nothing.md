---
id: TASK-23150
title: Console Behavior card grew past the test viewport and clicks land on nothing
status: Done
assignee:
  - '@codex'
created_date: '2026-08-28'
updated_date: '2026-09-17 22:35'
labels:
  - tests
  - settings
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
All 3 tests in `Tests/UI/test_settings_console_rail_labels.py` fail. The production save path is
intact — the decisive evidence is inside the third test, where values set **programmatically**
stage correctly while the one reached by `pilot.click` does not, and the checkbox's own `.value`
stays `False`, so the widget never received the toggle.

The Console Behavior card grew roughly 46 lines above that checkbox across two 2026-08-26 commits,
at the test's 190x55 viewport.

**One caveat carried over from the diagnosis and deliberately not closed:** "below the fold" was
inferred from the two commits plus the zero-effect click, *not* asserted against the viewport. The
first step of this task is to verify it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The checkbox's region is asserted to be inside the visible container **before** anything is
  changed, confirming or refuting the below-the-fold diagnosis
- [x] #2 The tests scroll or focus the control into view (or drive it by key) rather than clicking a
  fixed position
- [x] #3 A visibility assertion fails loudly on future layout growth instead of silently clicking air
- [x] #4 If the card genuinely no longer fits a realistic terminal, that is filed as a separate UX task
  rather than absorbed here
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md (existing governance)
Reason: Repair test isolation and interaction evidence without changing production behavior or architectural boundaries.
1. Restore the existing rail-label cases under the established private-profile helper so configuration ownership remains valid for one interpreter lifetime.
2. Record the checkbox and viewport regions before focus or scrolling, reproducing or refuting the original missed-click report.
3. Drive the affected toggles through focus and keyboard with explicit post-scroll visibility and painted-content assertions; retain original persistence, failure, runtime and revert assertions.
4. Run the targeted rail-label file, compare lint/format against the baseline, self-review and record evidence. Track a separate UX follow-up only if real production styling makes controls unreachable.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Confirmed the pre-focus checkbox region (41,79,36,3) lies below the visible Settings pane (37,29,111,24). Restored the nine existing cases using exact-node private profiles; the three missed-click cases now focus, assert viewport and compositor visibility, and press Space while preserving original save/runtime/failure/revert checks. A separate production-style compact overflow was filed and repaired as TASK-32759. Evidence and exact test/native limits: Docs/superpowers/qa/2026-09-17-settings-console-rail/README.md. Ruff check/format, targeted verification and independent read-only review pass. Existing ADR-150/161 apply; no new ADR required.
<!-- SECTION:NOTES:END -->

## Evidence

Production seams all present and unchanged: handler `settings_screen.py:19612`, key in
`CONSOLE_BEHAVIOR_SAVE_ORDER` (`:915`), save branch `:22520`, adapter call `:23110` (the
monkeypatched name). Checkbox at `settings_screen.py:13345`, now below an exchange-capture block and
a thinking-visibility block.

Growth blames to `c6218918d1` (32 lines, "Codex/full semantic capture (#2126)") and `4aa87159ee`
(14 lines, "feat: add Console thinking visibility and history controls"), both 2026-08-26. The test
file was last updated 2026-08-24 (`a0d61d9957`).
