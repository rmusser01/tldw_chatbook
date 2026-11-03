---
id: TASK-38600
title: Add ruff lint gate to CI and clear unused-import debt
status: Done
created_date: 2026-11-02 14:28
labels:
- chore
- ci
priority: medium
updated_date: 2026-11-03 02:56
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
None of the 24 CI workflows runs a linter and the package carries 1219 findings under ruff F B ASYNC PLE rules including 126 unused imports; full list in /tmp/ruff_audit_full.txt. See bug-sweep-2026-11-02-dev.md section 4.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A CI workflow runs ruff with at least F B ASYNC and PLE selected
- [x] #2 Unused imports are fixed so the gate starts green
- [x] #3 Remaining findings are triaged or baselined explicitly
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
F-class findings fixed 146 to 0 across 65 files with reasoned noqas; ruff ratchet gate added (lint.yml, baseline 1064 findings over 26 rules, pinned 0.16.6) plus ADR-287. Commits d0bfa5a5fe and d4f8e61b94.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
