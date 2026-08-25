---
id: TASK-21146
title: Fold model-lists consent into wizard summary
status: To Do
assignee: []
created_date: '2026-08-25 06:15'
labels:
  - ux
  - wizard
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT finding H-1 (findings.md): finishing the wizard lands in Console which immediately opens the 'Check model lists online?' consent modal - a fourth decision at the moment of first chat, and setup itself already contacted the provider. Surface the consent as an unchecked-by-default option on the wizard Summary; Console must respect the recorded answer and not re-ask. Privacy default must not weaken: no consent means no online checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Wizard Summary offers the model-lists consent, default off
- [ ] #2 After completing the wizard, Console does not show the consent modal (either answer)
- [ ] #3 Users who skip the wizard still get the existing Console consent flow
<!-- AC:END -->
