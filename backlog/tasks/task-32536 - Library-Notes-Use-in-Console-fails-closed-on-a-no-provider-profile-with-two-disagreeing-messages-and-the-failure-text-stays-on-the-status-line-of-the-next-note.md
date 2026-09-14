---
id: TASK-32536
title: >-
  Library Notes: "Use in Console" fails closed on a no-provider profile with two
  disagreeing messages, and the failure text stays on the status line of the
  next note
status: To Do
assignee: []
created_date: '2026-09-13 06:45'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), both assessors, personas researcher/student and Alex, Edit workflow. P1 (A) / D1 + D2 (B).

**What happened.** Fresh profile (the default first-run state, no provider): Tab to "Use in Console" (footer chip "enter use in Console") → status "Use in Console failed — check Console readiness and try again. · Next: Review the error, then keep editing." plus a toast "Copy or link blocked Library sources into the active workspace before using them in Console." (B 14; A 55/56). No link to set up a provider, no statement that nothing was staged, and no error anywhere on screen to review. Escape, open note B, edit, let it save: fifteen minutes later B's status line reads "Saved 06:03 · Use in Console failed — check Console readiness and try again. · Next: Review the error, then keep editing." and the list status line carries "…failed — check Console readiness and…" too (B 33, 35, 53; A 68 on the power profile). Captures: A 55, 56, 68; B 14, 33, 35, 53.

**Cause.** Strings PROVEN: `tldw_chatbook/UI/Library_Modules/library_notes_controller.py:4270` (`failure_next_action="check Console readiness and try again"`, reported whenever the hand-off returns false) and `tldw_chatbook/Workspaces/display_state.py:644` (a tooltip reused as the toast, naming a different blocker — workspace linking). Persistence INFERRED: `_finish_library_notes_operation` (`library_notes_controller.py:1597-1625`) stores the terminal failed state and renders it; no clear-on-open was found. The actual blocker on a no-provider profile is INFERRED. B: "the two messages describe two different blockers; a first-run user should get one sentence naming the real one." Docs contradicted: notes.md says Use in Console hands the note to Console with the suggested prompt. Improvement idea in the critique-3 ideas task: readiness-independent hand-off.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On a profile with no ready model, Use in Console either stages the note and lands on Console's own setup card, or shows one sentence naming the real blocker with the remedy action inline (for example Set up provider)
- [ ] #2 Exactly one message is shown for a failed hand-off (status line or toast, not both), and "Next: Review the error" is never offered when no error is on screen
- [ ] #3 Opening another note, or saving the current one, clears a previous action failure from the editor status line, and the list status line never carries it
- [ ] #4 A regression test performs a failed hand-off, opens a second note and saves it, and asserts the second note's status line reads Saved with no failure text
<!-- AC:END -->
