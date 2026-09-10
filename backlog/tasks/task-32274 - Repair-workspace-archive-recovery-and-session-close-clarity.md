---
id: TASK-32274
title: Repair workspace archive recovery and session close clarity
status: Done
assignee:
  - '@codex'
created_date: '2026-09-10 15:35'
updated_date: '2026-09-10 19:51'
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
ADR path: backlog/decisions/147-conversation-archive-and-exact-resume.md
Reason: implement shared lifecycle and recovery design.

Inspect lifecycle and close flows; pin failing collision/status/feedback tests; implement reversible workspace controls and accurate close impact; run targeted mounted checks.

Plan: Docs/superpowers/plans/2026-09-10-console-archive-recovery.md
Spec: Docs/superpowers/specs/2026-09-10-console-archive-recovery-design.md

PR #2576 review corrections: reproduce stale Console composer guards, workspace browser cache reuse, and stale Settings Undo receipts; capture drafts using their visible owner before both guard checks; invalidate cache after lifecycle writes; retire matching restored receipts; verify targeted guard and mounted lifecycle tests. Confirm empty current-owner composer text clears stale stored drafts without clearing another session during a transition.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added atomic workspace Restore as, visible archived rows, Show archived, and Undo/View archived receipts in Console and Settings. Restore does not activate the workspace. Close copy distinguishes retained saved history from unsaved messages, drafts, attachments and live work; current dev tree-wide loss accounting is preserved.

ADR: backlog/decisions/147-conversation-archive-and-exact-resume.md. User guide: Docs/User_Guide/console/sessions-tabs-workspaces.md. Targeted verification and limitations: Docs/superpowers/qa/console/2026-09-10-archive-recovery.md.

Integrated onto an isolated branch from current dev; original checkout changes are excluded. Task and ADR IDs were reassigned to avoid published collisions.

PR #2576 review corrections (#3, #8, #11): shared conversation/workspace archive guards capture covered or retained Console composer text using its visible session owner before the initial and confirmation-time checks. An empty composer clears a stale stored draft only for a settled active owner; a session transition preserves the prior owner's draft. Console workspace archive/restore invalidates persisted browser rows before refresh. Settings Restore removes only the matching archive receipt, preserving unrelated Undo targets.

Verification: observed 7 failing review regressions before fixes; 28 distinct targeted tests then passed. The empty-composer follow-up reproduced 1 failure, then 26 targeted actions, lifecycle and mounted draft-confirmation tests passed (including 2 new empty-owner cases). Ruff checks and diff whitespace checks passed; self-review completed. Existing dependencies emit non-failing requests-version and pytest temporary-directory cleanup warnings. No full suite was run. Existing ADR-147 applies; no new architectural decision.
<!-- SECTION:NOTES:END -->
