---
id: TASK-36522
title: Ten modals added after wide-tier wave 2 keep their narrow fixed widths on full-screen terminals
status: To Do
created_date: 2026-10-24 00:26
labels:
- follow-up
- ui
- modals
- css
priority: low
references:
- Tests/UI/modal_wide_tier_registry.py
- Tests/UI/test_modal_wide_tier.py
dependencies:
- TASK-36382
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36382 (PR #3227, 2026-10-23). The wide-tier contract's reverse sweep had been hiding 15 ModalScreen classes behind three stale skip names. TASK-36382 triaged them by base width. Five are tiny or already wider than the tier. The other ten carry fixed widths that the tier's ladder would relax (base width at most 80 gets a cap of 120, 84-96 gets 150, 100 or more gets 170, at `WIDE_VIEWPORT_COLUMNS` = 150 and above; the registry already gives base 100 the 170 cap, as for `#notes-recovery-dialog` and `ChatbookTemplatesWindow`). They went into `MODAL_WIDE_TIER_SKIPPED` marked "follow-up wave", which the registry comment calls a queued gap, not a verdict:
- `ConsoleGoalSetupModal`, `ConsoleGoalHistory` (base 76) and `ConsoleGoalStatus` (90)
- `DreamsGoalsModal` and `DreamsStoryModal` (76)
- `GuardianRuleEditModal` (76)
- `MCPRecoveryReviewDialog` (80)
- `FileNotesHistoryDialog` (90)
- `WorkflowRunConfirmation` and `WorkflowRunSetup` (90% / 100)

The app is used full-screen (211x44 and 235x52), where these open at their base widths.

Not in scope: the older module-level "follow-up wave" entries in `MODAL_WIDE_TIER_SKIPPED_MODULES` (Research workspace, scheduling forms and others).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 211x44 and 235x52, each of the ten modals opens at the wide-tier cap that the ladder gives its base width, or the registry records why it stays at its base width
- [ ] #2 Below the wide viewport, each modal's geometry is unchanged
- [ ] #3 Each tiered modal has a registry anchor and no skip entry, and the wide-tier coverage contract passes
- [ ] #4 The boot CSS byte budget still passes
<!-- AC:END -->
