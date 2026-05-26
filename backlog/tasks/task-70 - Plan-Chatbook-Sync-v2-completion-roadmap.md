---
id: TASK-70
title: Plan Chatbook Sync v2 completion roadmap
status: Done
labels:
- sync
- sync-v2
- planning
- local-first
- roadmap
priority: high
documentation:
- backlog/tasks/task-59.1 - Add-Chatbook-Sync-v2-profile-summary.md
- backlog/tasks/task-69 - Surface-Chatbook-Sync-v2-profile-status.md
- backlog/tasks/task-60.4.2 - Post-release-write-sync-promotion-tranche.md
- Docs/superpowers/specs/2026-05-20-workspace-operating-context-handoff-prd-design.md
- Docs/superpowers/plans/2026-05-22-post-release-deferred-feature-tranches.md
- Docs/superpowers/specs/2026-05-26-chatbook-sync-v2-completion-roadmap-design.md
modified_files:
- backlog/tasks/task-70 - Plan-Chatbook-Sync-v2-completion-roadmap.md
- backlog/tasks/task-70.1 - Add-Notes-Sync-v2-outbox-producer-plan.md
- backlog/tasks/task-70.2 - Add-Chat-Sync-v2-outbox-producer-plan.md
- backlog/tasks/task-70.3 - Add-manual-Sync-v2-preview-and-execution-plan.md
- backlog/tasks/task-70.4 - Add-Sync-v2-conflict-review-and-recovery-plan.md
- backlog/tasks/task-70.5 - Add-Sync-v2-Notes-and-Chat-restore-QA-plan.md
- backlog/tasks/task-70.6 - Close-out-manual-Sync-v2-milestone-with-actual-use-QA.md
- Docs/superpowers/specs/2026-05-26-chatbook-sync-v2-completion-roadmap-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and track the complete Chatbook Sync v2 roadmap through polished multi-device sync: content-producing outbox paths, manual sync execution, restore/new-device flows, conflict review, user-private encryption/key recovery, scheduled/background sync, workspace-scoped datasets, and major-domain coverage. The first implementation milestone is manual reliable sync for the active server profile's personal dataset, covering both Notes and Chat messages before workspace-scoped datasets or background automation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Define the first content-producing Sync v2 scope and explicitly exclude background automation until manual sync is safe.
- [x] #2 Design the client-side boundaries for Notes outbox production, manual sync execution, conflict/status reporting, and restore compatibility.
- [x] #3 Break the effort into Backlog child tasks suitable for separate PRs after the spec is approved.
- [x] #4 Record design decisions, verification strategy, risks, and open questions in a committed spec document.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-read the current Sync v2 display-only baseline and workspace handoff PRD.
2. Draft a roadmap spec that identifies the first manual Notes+Chat milestone, exclusions, phases, risks, and verification gates.
3. Create child Backlog tasks for PR-sized implementation slices.
4. Run focused documentation/backlog verification and diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created the Chatbook Sync v2 completion roadmap spec and split the next work into child tasks for Notes outbox production, Chat outbox production, manual sync execution, conflict recovery, restore QA, and manual milestone closeout. The plan keeps write sync manual and active-profile/personal-dataset-only until Notes and Chat content-producing paths are verified with restore and actual-use QA evidence.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
`TASK-70` now records the Sync v2 completion path from the current read-only/dry-run baseline to reliable manual Notes+Chat sync, with background/workspace/broader-domain sync intentionally deferred behind QA gates.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Roadmap spec committed
- [x] #3 Child tasks created
- [x] #4 Verification recorded
<!-- DOD:END -->
