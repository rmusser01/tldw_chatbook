---
id: TASK-24406
title: Build Personal Context interview and review UI
status: Done
assignee:
  - '@codex'
created_date: '2026-08-30 02:28'
updated_date: '2026-08-30 04:02'
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
- [x] #1 Personal and workspace interviews render fixed or adaptive questions with provider and model disclosure before answer entry.
- [x] #2 Users can answer, skip, retry, use the fixed fallback, finish early, cancel, and resume without exceeding coordinator authority.
- [x] #3 Final review exposes every deterministic change with editable payload and privacy controls plus independent selection.
- [x] #4 Save only and Save and use with agents commit only selected reviewed changes and report runtime or cleanup recovery honestly.
- [x] #5 Settings exposes Run interview again for the selected eligible scope without adding legacy settings surfaces.
- [x] #6 Production-shaped Textual tests, CSS bundle checks, targeted regressions, and independent review pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED production-shaped Textual tests for provider disclosure, question controls, fixed fallback, cancellation, narrow layouts, final-review editing/selection, honest partial-success recovery, and the Settings re-interview action.
2. Implement a keyboard-first `ProfileInterviewScreen` as a thin coordinator client with explicit state labels, safe Escape/finish behavior, provider errors, bounded progress, and no hidden destructive bindings.
3. Implement `PersonalContextReviewModal` with independently selectable structured rows, editable supported payload values, syncability and visibility controls, Save only / Save and use with agents actions, and explicit cleanup/runtime recovery outcomes.
4. Connect the canonical Personal Context Settings panel to launch personal or selected mapped-workspace re-interviews; keep setup/workspace handoff wiring for the next task.
5. Add the dedicated modular stylesheet, rebuild consolidated CSS, run production-shaped UI/CSS and Personal Context regressions, perform the bounded Impeccable detector/review pass, and obtain independent specification/code-quality approval.

ADR required: no

ADR path: `backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md`

Reason: ADR-102 already governs interview review authority, privacy controls, runtime-local enablement, and the long-lived Settings experience; this task implements that approved UI contract without changing those boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a keyboard-first fixed/adaptive Personal Context interview with provider/model and retention disclosure, safe answer/skip/retry/fallback/finish/cancel/resume behavior, and explicit expired, cleanup-failed, and commit-outcome-unknown recovery states.
- Added a deterministic final-review modal that exposes every proposal, supports typed payload/privacy edits and independent selection, and commits only checked changes through Save only or Save and use with agents.
- Hardened the coordinator with validated compare-and-swap review rewrites, secret rejection, stable review identities, a durable ``committing`` fence, and honest runtime/cleanup recovery without duplicate commits.
- Added the canonical Settings re-interview action for the selected linked personal or workspace scope. Setup and workspace-creation handoff remain deliberately assigned to the next approved plan task.
- Added production-shaped Textual coverage and modular CSS. Verification passed with 128 targeted tests, Ruff check/format, CSS bundle reproduction, ``git diff --check``, an empty Impeccable detector result, and independent specification/code-quality approval. The repository-wide suite was not run, per the targeted-test policy.
- ADR-102 remains the governing decision. The threaded-commit cancellation incident and its verified mitigation are recorded in ``backlog/docs/lessons-testing-evidence.md``.
<!-- SECTION:NOTES:END -->
