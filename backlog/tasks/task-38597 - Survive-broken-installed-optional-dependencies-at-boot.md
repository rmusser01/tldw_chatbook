---
id: TASK-38597
title: Survive broken-installed optional dependencies at boot
status: Done
created_date: 2026-11-02 14:28
labels:
- bug
- utils
priority: medium
updated_date: 2026-11-03 02:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
check_dependency only catches ImportError so an installed-but-broken optional package such as a failed torch dlopen crashes eager boot instead of degrading to feature disabled. See bug-sweep-2026-11-02-dev.md section 1.7.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 check_dependency catches non-ImportError failures such as OSError
- [x] #2 Eager dependency check degrades to feature disabled instead of crashing boot
- [x] #3 Test covers a raising import
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
check_dependency catches non-ImportError failures as broken-unavailable with the failure kind logged so missing versus broken stays distinguishable. Commit 7bd2ad0ad2.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
