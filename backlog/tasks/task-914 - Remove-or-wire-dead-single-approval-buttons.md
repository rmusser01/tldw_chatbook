---
id: TASK-914
title: 'Remove or wire the dead single-approval card buttons'
status: To Do
assignee: []
created_date: '2026-07-27 03:55'
labels: [console, dead-code]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ChatApprovalCard's single-approval body renders "Allow once" (#approval-allow-once) and "Deny" (#approval-deny) buttons that are not wired in on_button_pressed and can never emit ApprovalDecided — pre-existing dead UI confirmed during the parallel-agents train review. All production flows use the batch body.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The single-approval buttons either resolve their round correctly or the dead body is removed.
- [ ] #2 No unreachable button handlers remain on the card.
<!-- AC:END -->
