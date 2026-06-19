---
id: TASK-89.7
title: Clarify Library study flashcards and quizzes handoffs
status: Done
dependencies:
- TASK-89.1
labels:
- library
- study
- flashcards
- quizzes
- handoffs
priority: medium
parent_task_id: TASK-89
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the Library study-related modes explain and preserve source context when users move from sources into Study, Flashcards, or Quizzes. The goal is not to rebuild those modules, but to make Library handoffs predictable and recoverable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Study, Flashcards, and Quizzes modes expose source-context actions that clearly state what will be carried forward and what remains WIP.
- [x] #2 When a source or source set is selected, the destination handoff preserves enough context for the receiving screen to show where the study material came from or why it cannot be used yet.
- [x] #3 Empty or unsupported states explain whether the user needs sources, generated study material, provider readiness, or a later implementation slice.
- [x] #4 Keyboard-only navigation covers selecting the mode, selecting source context, and activating the primary handoff.
- [x] #5 The handoff behavior follows TASK-89.1 content-hub ownership and does not duplicate Study, Flashcards, or Quizzes internals inside Library.
- [x] #6 Focused regression coverage verifies route targets, payload/context preservation, and recovery copy for unavailable paths.
- [x] #7 Actual CDP QA captures the rendered study/flashcards/quizzes handoff states and receives screenshot approval before completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: this is a UI/handoff clarity slice under the existing TASK-89.1 Library content-hub contract; it does not change storage, sync policy, provider/runtime boundaries, service contracts, or persistence schemas.

1. Add failing mounted regressions for Study, Flashcards, and Quizzes mode handoff copy, empty-state recovery, source-context preservation, and keyboard activation.
2. Add a small Library-screen helper for study handoff state and render mode-specific copy/actions in study-related modes.
3. Preserve existing StudyScopeContext and open_study_screen routing; do not duplicate Study, Flashcards, or Quizzes internals in Library.
4. Run focused Library/Study tests and git diff checks.
5. Capture actual rendered CDP/Textual-web screenshots for Study, Flashcards, and Quizzes handoff states and get approval before PR.
6. Update TASK-89.7 acceptance criteria and implementation notes after verification.
<!-- SECTION:PLAN:END -->

## Non-Goals

<!-- SECTION:NON_GOALS:BEGIN -->
- Do not rebuild Study, Flashcards, Quizzes, or generation workflows in Library.
- Do not add study content persistence or provider-backed generation beyond explicit handoff context.
<!-- SECTION:NON_GOALS:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added Library mode-specific handoff detail panels for Study, Flashcards, and Quizzes so the center pane now explains the purpose, carried source snapshot, ownership boundary, WIP limits, and recovery path.
- Kept Library as the source-preparation surface and preserved existing StudyScopeContext/open_study_screen routing instead of duplicating Study, Flashcards, or Quizzes internals.
- Added mounted regressions for source-context handoff copy, empty/no-source recovery, keyboard activation, and mode-specific recovery copy.
- Captured approved actual Textual-web/CDP screenshots:
  - `Docs/superpowers/qa/library-study-handoffs/library-study-handoff-seeded-cdp-2026-06-19.png`
  - `Docs/superpowers/qa/library-study-handoffs/library-flashcards-handoff-seeded-cdp-2026-06-19.png`
  - `Docs/superpowers/qa/library-study-handoffs/library-quizzes-handoff-empty-cdp-2026-06-19.png`
- Verification: `python -m pytest -q Tests/UI/test_product_maturity_phase3_library_study_context.py Tests/UI/test_library_content_hub.py --tb=short` passed with 23 tests; `git diff --check` was clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Library Study, Flashcards, and Quizzes modes now present explicit destination-native handoff copy and recovery paths while preserving the existing Study ownership boundary and source-context routing.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
