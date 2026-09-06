---
id: TASK-31912
title: Move pure Console rewind and settings draft policy to their existing owners
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 20:05'
updated_date: '2026-09-05 20:16'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove redundant screen ownership of pure rewind admission and settings draft construction while preserving exact policy and explicit controller boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Rewind summary eligibility and initial settings draft results remain identical for all existing cases.
- [x] #2 The pure helpers live on their existing Commands and SettingsNavigation owners, with redundant constructor callbacks retired and all runtime/test callers correctly bound.
- [x] #3 Normalized function AST bodies and arguments match the baseline, affected whole-file checks pass or pre-existing failures are separately evidenced, and no architecture ceilings increase.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Implements existing DESIGN.md section7 and the root-approved Docs/superpowers/plans/2026-09-05-console-private-delegate-cleanup-proposal.md. 1. Read task and census the two helper imports/callers/constructor protocols; characterize ownership RED and baseline focused behavior. 2. Move exact static helpers to Commands and SettingsNavigation and remove redundant callback parameters/assignments/wiring. 3. Retarget the screen restore caller and three direct fixture references without assertion changes. 4. Verify AST equivalence, entire affected policy/rewind/settings files, lint and exact screen counts. 5. Root reviews before scoped commit; separate delegate task follows under ID coordination.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline model-apply/rewind wholefiles46passed2failed107.43s: existing bare model-popover fake lacks _commit_console_settings_submission_live (naturally addressed by31750receiverretarget); rewind E2E gateway lacks prepare_chat_request/complete_auxiliary and is separately tracked31753. New owner characterization was RED2expectedfailures and is nowGREEN2. Both moved helpers compare ast.dump-identical to pre-move HEAD, including staticmethod decorator, signature and complete body; only owner/location/imports and redundant callback plumbing changed. No diagnostic calls in either helper. Whole7-file769-case verification running; no full-green claim.

Root reviewed full purehelper diff and new ownership regression and approved scoped commit on focused22passed44.87s plus exact AST equality, with two independently reproduced baseline failures and ongoing769case fullrun explicitly qualified. Root requested preservation of unrelated baseline formatting and unused session assignment: those incidental cleanups were reverted; the existing Chat session-settings F841 at line3068 remains baseline, not extraction-introduced. Production files and new/retargeted ownership code lint clean. Final screen17202lines/569methods (64line/2method reduction); unchangedcap still requires329lines/10methods. This task establishes policy ownership only; overallConsole repair remainsInProgress.
<!-- SECTION:NOTES:END -->

## PR 2427 rebase renumbering provenance

Review-owned TASK-31749 was renumbered to TASK-31912 on 2026-09-06
while rebasing PR 2427 onto dev c4d45c0926. The user approved preserving
upstream task identities and renumbering review-created collisions only.
Original creation dates, task history, and literal verification artifact paths
are retained. See backlog/docs/pr-2427-rebase-reconciliation.md for the mapping.
