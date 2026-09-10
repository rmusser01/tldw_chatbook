---
id: TASK-32274
title: Repair workspace archive recovery and session close clarity
status: Done
assignee:
  - '@codex'
created_date: '2026-09-10 15:35'
updated_date: '2026-09-10 21:08'
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
PR #2576 wave 2 backend: use shared Pydantic name validation matching the existing strict-text/nonblank WorkspaceRecord contract; reject concurrent zero-row restore updates without publishing mutation success; verify a real SQLite restore race and name compatibility.

PR #2576 second review: reproduce registry-read error escapes and UI-thread storage in Console Restore/Settings Undo; dispatch recovery through asynchronous storage calls, fence delayed UI publication, retain failed receipts, and assert replacement rename-input identity before the next action. Verify blocking-storage responsiveness and targeted mounted lifecycle flows.
PR #2576 third review: preserve deleted conversation discovery independently of archive scope; prevent failed new mutations from exposing prior Undo; recheck durable state before existing-session Resume; serialize Unicode name checks with restore writes; align workspace-archive action copy and navigation contracts. Add focused regressions and verify affected integrations. ADR required: no new ADR; implements existing ADR147 lifecycle/recovery boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added atomic workspace Restore as, visible archived rows, Show archived, and Undo/View archived receipts in Console and Settings. Restore does not activate the workspace. Close copy distinguishes retained saved history from unsaved messages, drafts, attachments and live work; current dev tree-wide loss accounting is preserved.

ADR: backlog/decisions/147-conversation-archive-and-exact-resume.md. User guide: Docs/User_Guide/console/sessions-tabs-workspaces.md. Targeted verification and limitations: Docs/superpowers/qa/console/2026-09-10-archive-recovery.md.

Integrated onto an isolated branch from current dev; original checkout changes are excluded. Task and ADR IDs were reassigned to avoid published collisions.

PR #2576 review corrections (#3, #8, #11): shared conversation/workspace archive guards capture covered or retained Console composer text using its visible session owner before the initial and confirmation-time checks. An empty composer clears a stale stored draft only for a settled active owner; a session transition preserves the prior owner's draft. Console workspace archive/restore invalidates persisted browser rows before refresh. Settings Restore removes only the matching archive receipt, preserving unrelated Undo targets.

Verification: observed 7 failing review regressions before fixes; 28 distinct targeted tests then passed. The empty-composer follow-up reproduced 1 failure, then 26 targeted actions, lifecycle and mounted draft-confirmation tests passed (including 2 new empty-owner cases). Ruff checks and diff whitespace checks passed; self-review completed. Existing dependencies emit non-failing requests-version and pytest temporary-directory cleanup warnings. No full suite was run. Existing ADR-147 applies; no new architectural decision.

PR #2576 wave 2 backend corrections: Restore as uses a shared Pydantic validator matching WorkspaceRecord (strict text, trimmed and nonblank), rejecting accidental numeric/bytes coercion while preserving established long Unicode and internal-character names. No undocumented length or character restriction was introduced. Conditional restore requires one updated row and raises WorkspaceNotFound before mutation generation advances when another restore already committed. A deterministic race with independent real SQLite connections proves the losing name cannot report success. Registry/name/import run: 72 passed; existing import/service neighbor run: 72 passed. New test files pass Ruff; existing-file diagnostic multisets are unchanged, diff whitespace passes, and self-review completed.

PR #2576 wave 2 UI corrections (#3983108330, #3983108367, #3983108378, #3983108393): Console Restore and its Show archived recovery opener run registry reads/writes through storage_call in app-owned workers. Request/current-screen checks prevent delayed modal or refresh publication after navigation; successful writes still invalidate cached rows. Settings Undo protects initial reads and writes, keeps receipts on storage failures, and preserves newer receipts/selections while earlier work settles. Rename verification now fails explicitly unless a different replacement input mounts.

UI verification: 4 focused reproductions failed before implementation and then passed. Final targeted runs cover 29 distinct cases: 25 recovery/lifecycle tests passed, plus 4 Settings archive/restore/rename cases in the preceding 22-pass run. Coverage includes blocked storage with a responsive event loop, read/write failure feedback, delayed navigation, newer receipt preservation, and mounted Restore as at both viewport sizes. New/changed ranges pass Ruff (existing unrelated file diagnostics remain), range formatting and diff whitespace checks pass, and self-review completed. This implements ADR-147 without changing the registry ownership boundary.
Final integrated review verification and baseline limits are recorded in Docs/superpowers/qa/console/2026-09-10-archive-recovery.md. All modified archive flows pass their targeted tests; the unrelated compact Overview assertion reproduces with the prior Settings implementation. Started-write cancellation preserves storage completion publication. Existing ADR147 applies.
PR #2576 third review: fixed Trash archive independence, failed-mutation Undo ownership, durable existing-tab Resume checks, serialized Unicode restore names, workspace recovery labels, and archive navigation contracts. Targeted real SQLite, recovery and mounted checks pass; third-review evidence and temporary host-disk interruption are recorded in the QA report. ADR147 applies and documents deletion-oriented scope. Self-review and scoped static checks complete.
<!-- SECTION:NOTES:END -->
