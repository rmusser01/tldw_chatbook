---
id: TASK-24613
title: Add human-reviewed Agent Lesson promotion proposals
status: To Do
assignee: []
created_date: '2026-08-30 01:17'
labels:
  - notes
  - agents
  - security
dependencies:
  - TASK-24309
documentation:
  - >-
    Docs/superpowers/specs/2026-08-29-agent-lessons-notes-organization-sync-design.md
  - >-
    Docs/superpowers/plans/2026-08-29-agent-lesson-promotion.md
  - backlog/decisions/104-human-reviewed-agent-lesson-promotion.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Turn verified Agent Lessons into small reviewable proposals for authorized user-owned instructions while keeping lesson content untrusted and every application human-controlled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A foreground primary can present one exact promotion proposal from independently verified lesson evidence
- [ ] #2 A repository-instruction proposal identifies the writable binding, target path, current effective instruction chain, exact resulting content, and current target-state precondition before asking for approval
- [ ] #3 Repository instruction application uses one existing file-mutation seam with an atomic expected-digest or expected-absent check at the write boundary, so stale proposals fail without mutation and unrelated user edits are preserved
- [ ] #4 Chatbook-managed local skills remain proposal-only in Console; the user manually applies an approved proposal through the existing Library skill editor/service and its version plus re-trust boundary
- [ ] #5 Subagents can return evidence and candidate text but cannot present an approval request or apply a promotion change
- [ ] #6 Ineligible targets, missing authority, changed bindings, changed effective instruction chains, and stale content fail without mutation and report a non-sensitive reason
- [ ] #7 Promotion outcomes are recorded only through separately approved ordinary Agent Lesson Note updates and never authorize later writes
- [ ] #8 Targeted deterministic tests and scripted behavioral evaluations cover proposal quality, exact review, stale-state refusal, role enforcement, and the untrusted-data boundary
<!-- AC:END -->
