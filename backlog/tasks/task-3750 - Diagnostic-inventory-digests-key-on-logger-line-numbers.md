---
id: TASK-3750
title: Diagnostic inventory digests key on logger line numbers
status: To Do
assignee: []
created_date: '2026-08-08 21:06'
labels:
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Docs/security/production-diagnostic-inventory.json stores a per-file digest that changes when logger call LINE NUMBERS move, not only when the diagnostics themselves change. Any refactor that shifts lines in a file containing logging therefore fails Tests/Architecture/test_persistent_diagnostic_inventory.py and needs a review-and-regenerate cycle -- this cost one cycle per task across decomposition waves 4 and 5, every time with call_count unchanged and the sink topology byte-identical. Keying the digest on diagnostic CONTENT (message, level, owner) rather than position would keep the security signal the file exists for while removing a per-refactor chore that trains people to regenerate it without reading the diff.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Moving a logger call within a file without changing it does not change its digest
- [ ] #2 Adding, removing, or editing a diagnostic still changes the digest and fails the test
- [ ] #3 The existing reviewed inventory is migrated to the new keying in one deliberate commit
<!-- AC:END -->
