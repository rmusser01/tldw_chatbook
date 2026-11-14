---
id: TASK-40722
title: Restore plugin backup coverage in private SQLite CI inventories
status: Done
created_date: 2026-11-13 18:06
assignee:
- '@codex'
priority: high
updated_date: 2026-11-14 04:36
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current dev fails two required architecture-review checks because the plugin backup owner added by TASK-40132 is missing from the behavioral backup-owner matrix and static capture-call census.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The current plugin registry backup owner participates in the existing behavioral copy-backup cases.
- [x] #2 The exact plugin capture call is represented in the census without weakening detection of extra calls.
- [x] #3 The two reproduced failing checks and plugin-owner behavioral cases pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Test-only correction reflecting existing TASK-40132 policy and documented B41 inventory; no runtime or storage boundary changes.

1. Reproduce both current-dev failures and inspect TASK-40132, the actual plugin capture call, and existing B41 inventory.
2. Add the existing plugin owner to the copy-backup test matrix and its exact capture call to the census.
3. Run the two failing checks and the newly included behavioral parameter cases; record evidence.
4. During integration with dev531acb8c, retain its equivalent plugin census/matrix entries exactly once, remove duplicate additions from this branch, and rerun the affected inventory and plugin behavior checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Restored the existing recovery.plugins.registry owner to COPY_BACKUP_OWNER_IDS, so the generic real-SQLite backup behavior also exercises it, and added the exact Plugins/backup::_Plugins.capture::copy_private_sqlite census entry already documented as B41. Bound the census loop predicate's expected value explicitly to satisfy Ruff B023; behavior is unchanged. No production code, inventory exclusions, or test coverage reductions.

Both original failures reproduced on dev1455ceeb. After the correction, 5 affected cases passed (407 deselected); the final census rerun after the loop-binding correction passed. Ruff check passes for both edited test files; diff whitespace clean. Independent review found no actionable issues. Existing B41 documentation already matches production, so no inventory-doc change was needed. ADR required: no (test-only correction of existing TASK-40132 policy). Full suite not run.
Integration with dev531acb8c on 2026-11-14 retained dev's equivalent plugin matrix/census additions and removed duplicate entries introduced by the rebase. The exact census reproduced expected count two versus one actual plugin capture call (/tmp/tldw-ci-dev-rebase-inventory-red.log). Removing only the duplicate initializers retained the lambda binding correction and all behavioral coverage. The census and plugin owner cases then passed four tests in23.49s (/tmp/tldw-ci-dev-rebase-inventory-green.log). Independent rebase follow-up review cleared the duplication fix; Ruff and whitespace checks pass.
The backup-owner completeness check also passed (one case,0.45s; /tmp/tldw-ci-dev-rebase-backup-owners.log), completing the focused inventory/behavior checks after reconciliation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
