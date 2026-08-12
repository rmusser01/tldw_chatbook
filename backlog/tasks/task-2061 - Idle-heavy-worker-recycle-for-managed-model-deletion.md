---
id: TASK-2061
title: Idle heavy-worker recycle for managed model deletion
status: In Progress
assignee: []
created_date: '2026-08-03 20:11'
updated_date: '2026-08-12 20:49'
labels:
  - stt
  - artifacts
  - architecture
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-08-12-task-2061-idle-worker-recycle-design.md
  - Docs/superpowers/plans/2026-08-12-task-2061-idle-worker-recycle.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Split from TASK-596 AC #5. The browser's deletion flow reports lease blockers honestly and never bypasses an active lease or cancels an active job (delivered and tested), but the other half of that criterion -- 'deletion CAN REQUEST an idle heavy-worker recycle' -- has no mechanism: nothing in ModelArtifactService or the heavy-worker pool can ask a worker to unload an idle resident model (grep recycle = 0). This is a cross-subsystem design against the parse/STT heavy-worker pool owner, not browser UI work, which is why it was deferred by the TASK-596 spec (Docs/superpowers/specs/2026-08-01-task-596-model-artifact-browser-design.md, 'Mutations') and never fit the browser PRs. Design first: the request/ack contract between the deletion flow and the pool, idle detection, and what 'refused' looks like in the UI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A deletion blocked only by an idle resident model can request a recycle and proceed once the worker unloads
- [ ] #2 An active (non-idle) lease is never bypassed and an active job is never silently cancelled
- [ ] #3 The browser's blocked-deletion UI shows recycle-requested state distinctly from hard-blocked
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Test-drive worker-confirmed resident lease-closure reporting.
2. Test-drive exact idle-only executor recycling with active and nonmatching refusal.
3. Bind the existing app-owned executor into the Installed model view without lazy construction.
4. Test-drive one lease-enforced deletion retry with policy recheck and distinct path-private UI states.
5. Run focused tests, required mutations, static checks, Ponytail/correctness review, and close the task only after all criteria pass.
<!-- SECTION:PLAN:END -->
