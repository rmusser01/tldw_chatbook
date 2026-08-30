---
id: TASK-24406
title: Build Personal Context interview and review UI
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-30 02:28'
updated_date: '2026-08-30 02:29'
labels:
  - personal-context
  - interviews
  - ui
dependencies:
  - TASK-24405
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide a keyboard-first Textual interview and final-review flow that discloses provider use and commits only user-approved Personal Context changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Personal and workspace interviews render fixed or adaptive questions with provider and model disclosure before answer entry.
- [ ] #2 Users can answer, skip, retry, use the fixed fallback, finish early, cancel, and resume without exceeding coordinator authority.
- [ ] #3 Final review exposes every deterministic change with editable payload and privacy controls plus independent selection.
- [ ] #4 Save only and Save and use with agents commit only selected reviewed changes and report runtime or cleanup recovery honestly.
- [ ] #5 Settings exposes Run interview again for the selected eligible scope without adding legacy settings surfaces.
- [ ] #6 Production-shaped Textual tests, CSS bundle checks, targeted regressions, and independent review pass.
<!-- AC:END -->

## Implementation Plan

1. Add RED production-shaped Textual tests for provider disclosure, question controls, fixed fallback, cancellation, narrow layouts, final-review editing/selection, honest partial-success recovery, and the Settings re-interview action.
2. Implement a keyboard-first `ProfileInterviewScreen` as a thin coordinator client with explicit state labels, safe Escape/finish behavior, provider errors, bounded progress, and no hidden destructive bindings.
3. Implement `PersonalContextReviewModal` with independently selectable structured rows, editable supported payload values, syncability and visibility controls, Save only / Save and use with agents actions, and explicit cleanup/runtime recovery outcomes.
4. Connect the canonical Personal Context Settings panel to launch personal or selected mapped-workspace re-interviews; keep setup/workspace handoff wiring for the next task.
5. Add the dedicated modular stylesheet, rebuild consolidated CSS, run production-shaped UI/CSS and Personal Context regressions, perform the bounded Impeccable detector/review pass, and obtain independent specification/code-quality approval.

ADR required: no

ADR path: `backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md`

Reason: ADR-102 already governs interview review authority, privacy controls, runtime-local enablement, and the long-lived Settings experience; this task implements that approved UI contract without changing those boundaries.
