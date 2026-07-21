---
id: TASK-421
title: Skills trust - remediation guidance for non-reviewable quarantine states
status: To Do
assignee: []
created_date: '2026-07-21 15:19'
labels:
  - skills
  - ux
  - trust
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P2 from the 2026-07-21 Skills UX/NNG review. quarantined_manifest_error and quarantined_unsupported_path are trust_blocked but excluded from review eligibility and not unlockable, so every trust button stays disabled and the panel offers no way forward. NNG heuristic 9 (help users recognize, diagnose, and recover from errors).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both non-reviewable quarantine states render guidance explaining what happened and what to do next,The on-disk skill path is surfaced so the user can inspect or fix files externally,No trust state leaves the panel with zero enabled remediation or guidance,Covered by tests
<!-- AC:END -->
