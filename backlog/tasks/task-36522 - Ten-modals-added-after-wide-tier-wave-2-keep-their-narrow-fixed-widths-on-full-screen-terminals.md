---
id: TASK-36522
title: Ten modals added after wide-tier wave 2 keep their narrow fixed widths on full-screen terminals
status: Done
assignee:
- '@claude'
created_date: 2026-10-24 00:26
updated_date: 2026-11-03 20:56
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
- [x] #1 At 211x44 and 235x52, each of the ten modals opens at the wide-tier cap that the ladder gives its base width, or the registry records why it stays at its base width
- [x] #2 Below the wide viewport, each modal's geometry is unchanged
- [x] #3 Each tiered modal has a registry anchor and no skip entry, and the wide-tier coverage contract passes
- [x] #4 The boot CSS byte budget still passes
<!-- AC:END -->

## Implementation Plan

1. Tier each modal by its base width, as wave 2 and TASK-38641 did: one `App.-wide-viewport <anchor> { width: $ds-percent-85; max-width: $ds-size-N; }` rule per anchor in the matching cap group of the wide-tier block in `css/components/_agentic_terminal.tcss`; rebuild the bundle with `build_css.py`.
   - cap 120 (base <= 80): `ConsoleGoalSetupModal > .goal-dialog-panel`, `ConsoleGoalHistory > .goal-dialog-panel` (76), `#dgm-dialog` (DreamsGoalsModal, 76), `#dsm-dialog` (DreamsStoryModal, 76), `#grm-dialog` (GuardianRuleEditModal, 76), `MCPRecoveryReviewDialog #confirmation-dialog` (80).
   - cap 150 (base 84-96): `ConsoleGoalStatus > .goal-dialog-panel` (90), `#file-notes-history-dialog` (FileNotesHistoryDialog, 90).
   - cap 170 (base 100): `.workflow-run-dialog`, the one class both workflow run modals share (90% / 100).
   - The goal panel class is shared by three modals with different bases, and `#confirmation-dialog` by every ConfirmationDialog, so those anchors are type-scoped.
2. Registry: add the nine anchors with their owner modules; delete the ten "follow-up wave" rows from `MODAL_WIDE_TIER_SKIPPED`.
3. Tests (`Tests/UI/test_modal_wide_tier.py`): add the ten modals to the spot representatives with their base rules and caps (both workflow modals measured, through a type-scoped query for the second), built idle with no database, service or provider; measure them with `run_test(size=...)` at 200x50, 211x44 and 235x52 (cap), at 149x44 and 120x40 (base), and across live resizes. Show the wide cases red before the CSS lands.
4. Check the coverage contract (the ten names leave the skip list, nothing new joins the sweep's output), the boot CSS byte budget, the CSS build/governance tests, bundle sync via preflight, and the size ratchets.

## Implementation Notes

All ten modals joined the wide tier at the cap the ladder gives their base width. None needed a skip: every one sizes its content to the dialog (fill-width inputs, editors and option lists, `1fr` grid columns, fill-width or compact buttons), so nothing has a fixed inner grid that extra width would leave as dead space. Each anchor has one `App.-wide-viewport` rule (`width: $ds-percent-85; max-width: $ds-size-N`) in its cap group of the wide-tier block in `css/components/_agentic_terminal.tcss`, and the bundle was rebuilt with `build_css.py`. Below 150 columns nothing changes.

| Modal | Base width | Cap | Anchor |
|---|---|---|---|
| ConsoleGoalSetupModal | 76 / 100% | 120 | `ConsoleGoalSetupModal > .goal-dialog-panel` |
| ConsoleGoalHistory | 76 / 100% | 120 | `ConsoleGoalHistory > .goal-dialog-panel` |
| ConsoleGoalStatus | 90 / 100% | 150 | `ConsoleGoalStatus > .goal-dialog-panel` |
| DreamsGoalsModal | 76 / 96% | 120 | `#dgm-dialog` |
| DreamsStoryModal | 76 / 96% | 120 | `#dsm-dialog` |
| GuardianRuleEditModal | 76 / 96% | 120 | `#grm-dialog` |
| MCPRecoveryReviewDialog | 80 / 96% | 120 | `MCPRecoveryReviewDialog #confirmation-dialog` |
| FileNotesHistoryDialog | 90 / 95% | 150 | `#file-notes-history-dialog` |
| WorkflowRunSetup, WorkflowRunConfirmation | 90% / 100 | 170 | `.workflow-run-dialog` |

- Anchors: ids that are unique to their dialog are used bare. Three goal modals share `.goal-dialog-panel` with two different bases, and every ConfirmationDialog composes `#confirmation-dialog`, so those anchors are type-scoped. Both workflow run modals compose the same `.workflow-run-dialog` class, and nothing else does, so one rule covers both. Each tier rule outranks its base rule by specificity, wherever the base lives: the bundle (goal, MCP), the widget defaults (Dreams, Guardian) or a split sheet that loads later (File Notes history, workflows).
- Registry (`Tests/UI/modal_wide_tier_registry.py`): nine anchors with their owner modules, and the ten "follow-up wave" skip rows are deleted.
- Tests (`Tests/UI/test_modal_wide_tier.py`): the ten modals are spot representatives with their base rules and caps. Each is built idle, with no database, goal store, notes service or provider behind it. The second workflow modal is keyed by a type-scoped query (`WorkflowRunSetup .workflow-run-dialog`). Every entry from the portrait dialogs on is measured with `run_test(size=...)` at 200x50, 211x44 and 235x52 (cap), at 149x44 and 120x40 (base), and across live resizes in both directions: 70 new cases.
- Red/green: before the CSS, the 40 wide cases (4 per modal) failed with the base width against the cap (`76 == 120` x20, `80 == 120` x4, `90 == 150` x8, `100 == 170` x8). The 30 narrow cases passed and still pass, which is the AC#2 evidence that base geometry is unchanged. After the CSS, the whole file passes (131 cases, the reverse sweep included).
- Boot CSS: 597,451 -> 598,380 B (+929 B) against the 608,090 B limit. Two of the rules (File Notes history, about 95 B, and the workflow dialog, about 80 B) style dialogs whose base rules live in split sheets off the boot path. They ride the boot bundle like every other tier rule, because a `-wide-viewport` token pins a rule to the bundle under those splits.
- Not caused by this change: 26 covering tests are red on this branch. They fail identically on dev (eba3ad14b5) with the same command and the same messages: MCP recovery controls and review (19), two goal-resume controls, two 60x20 workflow approval cases, the class-coverage contract, the base-class tie order check and the split-sheet harness check.
- User Guide: no page describes these modals' widths, so no change.
- The new mounted spot cases are not on a PR lane, the same as every earlier spot case. The census runs this file's static contracts, which cover AC#3 on the lane.
