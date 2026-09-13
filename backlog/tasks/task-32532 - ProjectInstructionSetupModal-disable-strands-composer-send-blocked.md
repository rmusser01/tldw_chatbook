---
id: TASK-32532
title: 'ProjectInstructionSetupModal Disable strands composer send-blocked while readiness says Ready'
status: To Do
assignee: []
created_date: '2026-09-12 07:55'
labels:
  - console
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pre-existing defect found during TASK-32477 live verification (finding D2, evidence /tmp/apr-live/probe3_readiness_delta.py and probe3-app.log; reproduced on master @ 33dbeccc22-era code, NOT introduced by the routing branch). Disabling project instructions via ProjectInstructionSetupModal leaves the Console composer send-blocked even though the readiness indicator reports Ready — the user cannot send until some other state change unblocks it. Investigate the send-readiness projection vs the modal's disable path; the composer block and the readiness indicator disagree, so one of them is reading stale state. Adjacent (already-Done) items task-31553 (setup-refusal pending-send release) and task-16475 (stale-default surfacing) touch nearby code but do not cover this path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reproduce: after Disable in ProjectInstructionSetupModal, composer send state and readiness indicator agree
- [ ] #2 Root cause identified (stale state read in send gating or readiness projection)
- [ ] #3 Fix with a pilot test driving modal→Disable→composer state
<!-- AC:END -->
