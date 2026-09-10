---
id: TASK-32275
title: Build searchable conversation archive review and bulk recovery
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
Saved chats can be searched, reviewed, archived and restored individually or in batches from Library.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Active Archived and All scopes provide exact paged results and preserve query through review.
- [x] #2 A bounded read-only transcript with match navigation supports review before resume.
- [x] #3 Single and bulk archive restore and undo act on captured identities and expose partial failures.
- [x] #4 Resume and Use as source are distinct actions with responsive keyboard-operable layout.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/147-conversation-archive-and-exact-resume.md
Reason: implement shared lifecycle and recovery design.

Inspect Library paging and reader seams; add failing lifecycle/preview tests; implement scoped search, bounded reader and bulk receipts; verify keyboard/compact/wide behavior.

Plan: Docs/superpowers/plans/2026-09-10-console-archive-recovery.md
Spec: Docs/superpowers/specs/2026-09-10-console-archive-recovery-design.md
Review corrections for PR #2576: verify comments #1, #2, #7, #9 and #13 against current dev; add regressions for every newly added Reader child being absent, separate conversation/workspace lifecycle assertions, and invalid scope boundaries; use portable test artifacts and the shared strict scope validator; document handler contracts; run only the affected tests and compare lint to HEAD. Existing ADR-147 applies; no new storage or UX contract is introduced.
Wave 2 review: preserve honest partial recovery across the separate workspace/conversation stores using fresh confirmation preflight and explicit retry feedback; fence existing-session activation and handoff settlement against navigation/supersession; attach safe identity context to failures; parameterize recovery record types. Add narrow failing regressions before fixes and document the partial-completion policy in ADR-147.
PR #2576 third review: preserve deleted conversation discovery independently of archive scope; prevent failed new mutations from exposing prior Undo; recheck durable state before existing-session Resume; serialize Unicode name checks with restore writes; align workspace-archive action copy and navigation contracts. Add focused regressions and verify affected integrations. ADR required: no new ADR; implements existing ADR147 lifecycle/recovery boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added Active/Archived/All search scopes, workspace/archive labels, captured-ID single and bulk archive/restore, and version-checked Undo receipts. The current dev Library Reader remains the canonical read-only transcript/Find surface, with distinct Resume, Restore only and Use as source actions. Existing adaptive panes and single-page pager suppression are preserved.

ADR: backlog/decisions/147-conversation-archive-and-exact-resume.md. User guide: Docs/User_Guide/console/sessions-tabs-workspaces.md. Targeted verification and limitations: Docs/superpowers/qa/console/2026-09-10-archive-recovery.md.

Integrated onto an isolated branch from current dev; original checkout changes are excluded. Task and ADR IDs were reassigned to avoid published collisions.
PR #2576 review corrections: screenshots now use pytest's tmp_path. Reader synchronization resolves all added Find, source, archive and restore children inside the existing incomplete-tree guard, retaining incoming state for recomposition; the missing Find-position regression failed before the fix. Lifecycle tests now exercise all four conversation/workspace archive combinations with exact Active/Archived labels. A strict Pydantic scope validator is shared by Library controls, restored state/navigation and the DB scope parser; invalid values cannot mutate recovery scope. New public event handlers document their event arguments and effects.

Validation: 45 targeted tests passed (49 unrelated cases deselected), including both mounted terminal sizes, eight missing-child cases, the lifecycle matrix, and scope utility/event/recovery boundaries. Compilation and git diff --check pass; compared Ruff findings against HEAD with no new diagnostics in modified production/existing test files. Self-review confirmed every added sync_state child lookup is protected and the prior Reader/source eligibility contracts remain intact. No full suite was run.
Wave 2 review #3983108358: Library recovery metadata now uses tuple[Mapping[str, Any], ...] at the public annotation seam and worker return, with explicit list/mapping element types and dict[str, Any] for the lifecycle wrapper. Existing Library recovery tests pass as part of the 34-case targeted boundary/Library run; Ruff and compilation pass.
PR #2576 third review: fixed Trash archive independence, failed-mutation Undo ownership, durable existing-tab Resume checks, serialized Unicode restore names, workspace recovery labels, and archive navigation contracts. Targeted real SQLite, recovery and mounted checks pass; third-review evidence and temporary host-disk interruption are recorded in the QA report. ADR147 applies and documents deletion-oriented scope. Self-review and scoped static checks complete.
<!-- SECTION:NOTES:END -->
