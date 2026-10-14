---
id: TASK-34951
title: The Watchlists backend-switch draft test still flakes after its fix
status: To Do
created_date: 2026-10-14 14:04
labels:
- tests
- flaky
- watchlists
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Watchlists/test_watchlists_bulk_source_authoring.py::test_backend_switch_preserves_open_source_draft_widget_and_focus[size1] failed in at least eight required-check attempts across six branches on 2026-10-13 and 2026-10-14, including a docs-only branch and a run that passed on a same-commit re-run. Commit db3a0c3183 ('await tree choices before backend-switch draft') reached dev with PR #3010 on 2026-10-13 18:29Z to fix it, and it failed again about five and a half hours later. A flaky test in the required check can evict a good pull request from the merge queue. Found while re-measuring the required check for TASK-34789.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the remaining intermittent failure is identified, in the test or the product
- [ ] #2 The test passes 20 consecutive runs in CI
- [ ] #3 If the product was at fault, a test that fails deterministically without the fix pins it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
