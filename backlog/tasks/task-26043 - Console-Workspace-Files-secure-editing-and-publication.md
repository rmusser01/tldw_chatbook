---
id: TASK-26043
title: Console Workspace Files secure editing and publication
status: To Do
assignee: []
created_date: '2026-08-31 16:25'
updated_date: '2026-08-31 16:28'
labels: []
dependencies:
  - TASK-26042
references:
  - Docs/superpowers/specs/2026-08-31-workspace-files-inspector-design.md
  - >-
    backlog/decisions/079-workspace-file-inspector-direct-user-authority-and-save-publication.md
  - task-26042
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users deliberately edit and save ordinary safely publishable workspace text files without mixing manual changes into overlapping agent change review. Deliver one recoverable draft buffer, explicit edit-lease ownership, exact conflict detection, honest publication outcomes, and graceful quit behavior on top of the read-only inspector.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A read/write binding can enter explicit Edit for an eligible ordinary UTF-8 file, while read-only, unsafe, unsupported, or oversized files remain view-only with a specific adjacent reason and no try-anyway path.
- [ ] #2 Editing provides visible Save, Undo, Redo, Revert, Copy draft, and Done editing behavior with correct clean/dirty transitions and one inline guard for every draft-replacing navigation, dismissal, or quit intent.
- [ ] #3 One app-scoped canonical-root coordinator uses component-aware platform path semantics, atomically excludes overlapping manual and agent mutation windows, and releases every manual and agent lease on its defined terminal path.
- [ ] #4 Save is single-flight, briefly freezes typing, is visibly cancellable only before publication, and keeps the inspector mounted until the single terminal outcome after publication wins.
- [ ] #5 Exact baseline validation includes content, stable file identity, type/link facts, parent identity, BOM/newline/final-newline policy, mode, and all promised supported metadata; identical-byte replacement or metadata change conflicts rather than being overwritten.
- [ ] #6 Eligible Save uses an exclusive same-directory temporary file, supported durability steps, atomic replacement, metadata preservation, and final byte/identity/metadata verification without unsafe automatic retry.
- [ ] #7 The UI distinctly reports not published, published durable, and published with confirmation incomplete, preserving the draft and required recovery actions for conflicts, binding changes, failures, and uncertain publication.
- [ ] #8 Graceful Ctrl+Q reuses the dirty guard and waits for an already active Save; Chatbook exits only after a durable verified clean state, while force-termination limitations remain explicit and no draft is persisted.
- [ ] #9 Targeted service, real-filesystem race/platform, root-coordinator, production-shaped Textual, privacy, performance, and live scratch evidence proves both publication behavior and prohibited side effects.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 All acceptance criteria are checked only after their behavior and prohibited side effects are evidenced.
- [ ] #2 The task is moved to In Progress before an Implementation Plan is added, and that plan records ADR required: yes, ADR-079, and the reason.
- [ ] #3 Targeted automated tests, relevant static checks, and git diff --check pass; a full suite is run only after explicit user approval.
- [ ] #4 Production-shaped Textual evidence and an isolated live scratch verification cover the user-facing path and preserve unrelated Console and profile state.
- [ ] #5 Relevant documentation and concise Implementation Notes identify the approach, trade-offs, files changed, verification, and any plan deviation.
- [ ] #6 A self-review confirms security, privacy, accessibility, performance, licensing, task dependencies, and no unrelated regression before the task is set to Done.
<!-- DOD:END -->
