---
id: TASK-41033
title: "Three Canvas policy-worker recovery tests are red on dev: an accepted native policy read is not retained through backup or disposal"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- backup-recovery
- canvas
priority: medium
references:
- Tests/Backup_Recovery/test_canvas_policy_worker.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On dev ecad404bd4 (2026-11-15), run serially and at `-n 4`, three tests in `Tests/Backup_Recovery/test_canvas_policy_worker.py` fail:
- `test_backup_and_disposal_retain_accepted_native_policy_read[watcher_cancel]`: "accepted native read was not retained". The maintenance drain succeeds while the accepted read is still running.
- `test_backup_and_disposal_retain_accepted_native_policy_read[dispose_cancel]`: "dispose abandoned an accepted native read".
- `test_cancelled_watcher_reuses_pending_native_read_when_restarted`.

The `[maintenance]` and `[drain_cancel]` params pass. The 2026-11-12 verification of the Phase 3 merge chain saw the same three red on dev 4b90eb860b.

If the product is wrong rather than the tests, a backup can start while a Canvas policy read that it should wait for is still running. Only three other nodes of this file run on a PR lane (`derived-artifacts.yml`), so CI does not see these.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The task notes say whether the product or the tests are wrong, and name the commit that changed the behaviour
- [ ] #2 All four params of test_backup_and_disposal_retain_accepted_native_policy_read and test_cancelled_watcher_reuses_pending_native_read_when_restarted pass on dev, and a fixed product defect is shown red first
- [ ] #3 These nodes run on a PR lane (through TASK-40307's Backup_Recovery lane work), or the task notes say why not
<!-- AC:END -->
