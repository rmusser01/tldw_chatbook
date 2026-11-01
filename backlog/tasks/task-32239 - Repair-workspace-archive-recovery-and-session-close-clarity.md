---
id: TASK-32239
title: Repair workspace archive recovery and session close clarity
status: Done
assignee:
  - '@codex'
created_date: '2026-09-10 15:35'
updated_date: '2026-09-10 16:36'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Workspace archive is discoverable and reversible in Console and Settings, including name collisions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Archived rows visibly identify their state and can be restored from Console.
- [x] #2 Archive provides Undo and View archived; restoration provides explicit feedback.
- [x] #3 Name conflicts can be resolved while restoring and close copy distinguishes saved history from actual losses.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/144-conversation-archive-and-exact-resume.md
Reason: implement shared lifecycle and recovery design.

Inspect lifecycle and close flows; pin failing collision/status/feedback tests; implement reversible workspace controls and accurate close impact; run targeted mounted checks.

Plan: Docs/superpowers/plans/2026-09-10-console-archive-recovery.md
Spec: Docs/superpowers/specs/2026-09-10-console-archive-recovery-design.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented atomic workspace Restore as without activation; Console Show archived, Restore, Restore as and a protected Undo/View archived receipt; Settings literal archived rows, restore naming, success feedback and persistent recovery actions. Workspace archive uses the shared live/queued/unsaved guard before and after confirmation. Close impact now includes draft and attachment loss and separates saved history from temporary/unsaved messages. ADR: backlog/decisions/144-conversation-archive-and-exact-resume.md. Initial RED tests reproduced missing restore keyword/controls/literal status, missing recovery receipt, archive confirmation draft race, and silent draft-only close. Targeted intermediate checks: 60 registry/Settings/workspace checks passed; six close checks passed; receipt and compact/wide restore checks passed after separating modal mount from Console refresh. Final integrated targeted verification and parent review pending; keep In Progress.

Final integrated evidence: Docs/superpowers/qa/console/2026-09-10-archive-recovery.md. User guide updated at Docs/User_Guide/console/sessions-tabs-workspaces.md. ADR-144 linked. Compact/wide mounted flows and final cancellation-safe lifecycle run pass (29 checks); targeted FTS/recovery run passes (75); six close tests pass. Independent final review reports no remaining blockers. No full suite, live provider calls, commit or unrelated checkout cleanup.
<!-- SECTION:NOTES:END -->
