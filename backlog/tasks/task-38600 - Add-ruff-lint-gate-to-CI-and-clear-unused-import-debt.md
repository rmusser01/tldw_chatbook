---
id: TASK-38600
title: Add ruff lint gate to CI and clear unused-import debt
status: In Progress
created_date: 2026-11-02 14:28
labels:
- chore
- ci
priority: medium
updated_date: 2026-11-02 14:51
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
None of the 24 CI workflows runs a linter and the package carries 1219 findings under ruff F B ASYNC PLE rules including 126 unused imports; full list in /tmp/ruff_audit_full.txt. See bug-sweep-2026-11-02-dev.md section 4.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A CI workflow runs ruff with at least F B ASYNC and PLE selected
- [ ] #2 Unused imports are fixed so the gate starts green
- [ ] #3 Remaining findings are triaged or baselined explicitly
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
