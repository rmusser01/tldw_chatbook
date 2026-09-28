---
id: TASK-33151
title: Design Console hook settings and persistent review
status: Done
assignee:
  - codex
created_date: '2026-09-28 01:40'
updated_date: '2026-09-28 02:50'
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
- [x] #4 The written spec passes a consistency and link review and is presented for user review before implementation planning.
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
Authored the approved native Console Hooks toolbar action, expandable review modal, and canonical three-pane Settings design. The user reviewed the written spec positively and requested an audit on 2026-09-27. Traced the actual parser, runtime singleton, async subprocess launch, notification queue, prompt-queue refusal state, config locks, and structured save outcomes. Corrected legacy duplicate grant inheritance; consent-versus-spawn ordering; revoke/disable failure fencing; authoritative lossless config refresh; cancelled/delayed modal continuation; stale cross-process decisions; frozen notification targets; invalid/recovery row visibility; contextual Enable/Disable; safe command rendering and narrow-toolbar requirements. ADR-197 is accepted and linked from the spec, task, and index. All 12 local document links, placeholder/fence checks, scoped whitespace checks, and self-review passed. This is documentation-only verification; targeted runtime races remain required for implementation. Product files were not changed and no full suite was run. Renumbering provenance: the CLI assigned TASK-33097; the all-ref/history and 86-worktree sweep found TASK-33150 as maximum, so this new task became TASK-33151 before references were added; no existing task was changed.
<!-- SECTION:NOTES:END -->
