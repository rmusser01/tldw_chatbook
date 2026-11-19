---
id: TASK-35586
title: Dreams question-track UI entry and tracked-row interaction
status: Done
assignee:
  - '@plexo'
created_date: '2026-09-28 20:23'
updated_date: '2026-09-29 01:22'
labels:
  - dreams
  - phase2
  - ui
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 2 shipped the question-tracking service (track_question/run_track_check, judged change runs, rebaseline) with no production caller - the user guide deliberately words it as planned (final-review ruling P11). Add the one-keystroke entry on the story modal pre-filled from the story query, plus make the focusable-but-inert Tracked rows on the Artifacts screen do something (open a detail view or drop can_focus) - both are the same modal/screen surface.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Story modal exposes question-tracking pre-filled from the story query,Tracked rows are no longer a keyboard dead-end,Guide question-tracking paragraph returns to present tense
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add DreamsDB.get_story(story_id) targeted read (Tracked rows resolve origins the recent-stories window may have dropped)
2. Modal: w/watch binding + action_watch reusing track_question (query verbatim, kind-derived intent, tracked feedback, cap notice)
3. Artifacts screen: route artifacts-dream-track-row-* click/Enter to story modal when origin resolves, else dismissible summary notice
4. Guide: question-tracking paragraph to present tense; stamp refresh
5. Tests: bare-App watch action trio + full-harness tracked-row interaction + get_story
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped: w/watch question-track entry on the story modal (query verbatim, kind-derived intent, tracked feedback); Tracked rows open their origin story modal or a summary notice; guide present tense. Gate 238 passed x2.
<!-- SECTION:NOTES:END -->

ADR check: no new ADR required — targeted read + UI wiring inside the Phase 2 Track design (ADR-196 already governs); reviewer concurred in task-33164-report.md.

## Renumbering provenance

Renumbered from TASK-33164 to TASK-35586 during PR3162 qualification on 2026-10-18. The archive follow-up ownership task created at 2026-09-28 05:42 retains TASK-33164; this younger Dreams task was created at 2026-09-28 20:23 and added in commit 0085d96ab2. The completed implementation and original review evidence remain unchanged; the historical task-33164-report.md reference above identifies that original review. The new ID was reserved after scanning all 876 refs and 41 worktrees (maximum TASK-35585).
