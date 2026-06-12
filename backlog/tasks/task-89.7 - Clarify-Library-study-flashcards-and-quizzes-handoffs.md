---
id: TASK-89.7
title: Clarify Library study flashcards and quizzes handoffs
status: To Do
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
- [ ] #1 Study, Flashcards, and Quizzes modes expose source-context actions that clearly state what will be carried forward and what remains WIP.
- [ ] #2 When a source or source set is selected, the destination handoff preserves enough context for the receiving screen to show where the study material came from or why it cannot be used yet.
- [ ] #3 Empty or unsupported states explain whether the user needs sources, generated study material, provider readiness, or a later implementation slice.
- [ ] #4 Keyboard-only navigation covers selecting the mode, selecting source context, and activating the primary handoff.
- [ ] #5 The handoff behavior follows TASK-89.1 content-hub ownership and does not duplicate Study, Flashcards, or Quizzes internals inside Library.
- [ ] #6 Focused regression coverage verifies route targets, payload/context preservation, and recovery copy for unavailable paths.
- [ ] #7 Actual CDP QA captures the rendered study/flashcards/quizzes handoff states and receives screenshot approval before completion.
<!-- AC:END -->

## Non-Goals

<!-- SECTION:NON_GOALS:BEGIN -->
- Do not rebuild Study, Flashcards, Quizzes, or generation workflows in Library.
- Do not add study content persistence or provider-backed generation beyond explicit handoff context.
<!-- SECTION:NON_GOALS:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
