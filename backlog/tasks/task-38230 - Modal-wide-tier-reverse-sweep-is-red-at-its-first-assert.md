---
id: TASK-38230
title: 'Modal wide-tier reverse sweep is red at its first assert'
status: Done
assignee: []
updated_date: '2026-11-02 21:52'
created_date: '2026-10-30 18:43'
labels:
  - tests
dependencies: []
references:
  - Tests/UI/test_modal_wide_tier.py
  - Tests/UI/modal_wide_tier_registry.py
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
test_every_modalscreen_class_is_tiered_or_explicitly_skipped fails at its first assert on dev today (stale skip names), which masks its "missing" assert. Roleplay frame B2b added a skip row for its RoleplayChoiceModal and does not fix the stale names. Found during Roleplay frame B2b (TASK-33910.4, plan Docs/superpowers/plans/2026-10-26-roleplay-b2b-toolbar-new-import.md).

**Update 2026-11-02 (dev 88af9f9379).** TASK-36382 (#3227, merged 2026-10-31) removed the stale skip names, so the first assert now passes. The test is still red, now at its "missing" assert. Seven ModalScreen classes added since have neither a registry anchor nor a skip row:
- `ConsoleCodeReviewModal` (Widgets/Console/console_code_review_modal.py)
- `NewCardModal` (UI/Kanban/kanban_board_screen.py)
- `PluginAuthorityDialog`, `PluginAuthorityResetModal`, `PluginReviewModal`, `PluginStatusModal` and `PluginVariableDialog` (Widgets/plugin_review_modal.py)

`Tests/UI/test_modal_wide_tier.py` with `-n 4` through the scratch-profile wrapper gave 1 failed, 46 passed. AC#2 already holds: these seven turn the sweep red.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The reverse sweep passes on dev with every stale skip name removed or re-pointed
- [x] #2 A new ModalScreen subclass with no tier and no skip row turns it red
<!-- AC:END -->

## Implementation Notes

Already fixed on dev; this PR (TASK-33910.36 burn-down PR 7) changes nothing for it. The seven modals named in the 2026-11-02 update got skip rows in `Tests/UI/modal_wide_tier_registry.py` (lines 66-72) in `1770ac7f19` (PR #3346); the stale names were removed earlier by TASK-36382 (#3227).

Re-verified 2026-11-05 on origin/dev `29592ecc9b`: `Tests/UI/test_modal_wide_tier.py::test_every_modalscreen_class_is_tiered_or_explicitly_skipped` gives `1 passed` (read from the pytest summary line, 65.8 s). AC#2 holds by the test's own `missing` assert, which the seven unregistered modals turned red before `1770ac7f19`.
