---
id: TASK-31263
title: >-
  Command palette test
  test_palette_library_skills_command_opens_hidden_starter_route fails on clean
  dev
status: Done
assignee:
  - '@claude'
created_date: '2026-09-04 13:47'
updated_date: '2026-09-04 20:51'
labels:
  - tests
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_command_palette_providers.py::TestTabNavigationProvider::test_palette_library_skills_command_opens_hidden_starter_route fails identically on clean origin/dev (verified at 2516735cfd during PR #2374 work, run in its own process). It is in the same hidden-starter-route family as the six Library failures tracked by task-31249 but lives in a different file and is not covered by that task's list. Every PR touching command-palette or theme code has to hand-verify this failure is baseline, which is how real breaks hide.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test passes on dev, or is rewritten/removed with the reason recorded in this task (no bare skip markers)
- [x] #2 Root cause identified and recorded (production code vs test contract)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root-caused by verified bisect (first bisect run was polluted by a lost transcript + wrong first-bad; re-bisected the dev-only range and verified parent-good/child-bad by hand): 45a13eb2c8 'feat(library): migrate Skills to adaptive reader (#2134)' (2026-08-27). The migration left _reconcile_library_entry_state gating the Skills canvas behind the Library SOURCE snapshot: with _library_lookup_error set, the reconciler swapped in the #library-canvas-error Static, so the palette deep-link never mounted #library-skills-canvas. compose_content's Skills mount had already been un-gated with a comment recording exactly this decision (Skills state is registry/trust-worker owned, independent of the source snapshot); the reconciler was the un-fixed twin. Fix: drop the lookup-error guard in the reconciler's skills branch and exclude skills from the error-surface swap. Test env note: app_factory stubs make the source lookup FAIL, which is why the test (and a real user with a broken source DB) hit this. Evidence: pinning test green; sibling suites' failure set identical to clean-dev baseline (heals 1, breaks 0).
<!-- SECTION:NOTES:END -->
