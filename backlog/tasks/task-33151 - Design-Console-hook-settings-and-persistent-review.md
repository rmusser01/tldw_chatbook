---
id: TASK-33151
title: Design Console hook settings and persistent review
status: In Progress
assignee:
  - codex
created_date: '2026-09-28 01:40'
updated_date: '2026-09-28 01:45'
labels:
  - hooks
  - design
  - console
dependencies: []
references:
  - backlog/decisions/197-console-hook-configuration-review.md
documentation:
  - Docs/superpowers/specs/2026-09-27-console-hook-settings-and-review-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define native Console access to hook permissions and guided hook configuration in canonical Settings, with one-time review before existing, new, or changed commands can run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The written design matches the approved native Console toolbar, review modal, and three-pane Settings layout.
- [x] #2 The design defines next-Send review, preserved drafts, persistent exact-definition approval, revocation, and runtime enforcement across foreground and background paths.
- [x] #3 A new ADR records configuration ownership, consent storage, and the preserved legacy hook contract; the spec and task link it.
- [ ] #4 The written spec passes a consistency and link review and is presented for user review before implementation planning.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/197-console-hook-configuration-review.md
Reason: persistent hook consent changes execution admission, private state ownership, and canonical Settings navigation.
1. Trace actual Console controls, Send/queue admission, settings persistence, and shared hook execution; preserve ADR-148 runtime rules.
2. Record the approved native ASCII layout and exact review behavior in the written design spec.
3. Write ADR-197 for standalone user-hook consent and settings ownership; link the task and spec.
4. Check scope, consistency, local links, and documentation diffs; commit only this design work.
5. Present the written spec for user review before writing an implementation plan.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Recorded the approved native Console toolbar, expandable review modal, and canonical three-pane Hooks Settings design. ADR-197 defines local exact-definition consent, next-Send admission, shared-runtime launch checks, and preserved ADR-148 semantics. The written spec awaits user review; no product implementation or implementation plan has started. Documentation link and placeholder checks pass. Changed only the design spec, proposed ADR, ADR index, and this task. Renumbering provenance: the CLI assigned TASK-33097; the live all-ref/history and 86-worktree sweep found TASK-33150 as the maximum, so this new task was moved to TASK-33151 before cross-references were added. No existing task was changed.
<!-- SECTION:NOTES:END -->
