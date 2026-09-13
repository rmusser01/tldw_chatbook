---
id: TASK-20979
title: 'Flashcards viewing/SRS surface — decide and build what makes real flashcards rows visible'
status: To Do
assignee: []
created_date: '2026-08-23'
updated_date: '2026-08-23'
labels:
  - flashcards
  - notes
  - ui
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The flashcards data layer exists (`decks`, `flashcards`, `flashcards_fts` —
verified in the student-workflow design spec), but it has **no screen route**:
nothing in the app renders or reviews those rows. That is why the
student-workflow spec's ruling §7.3 (§6, "Flashcards — the deliberate
decision") made the sub-project's flashcard output **Q/A markdown inside
notes** — visible in the notes screen the moment it lands — and explicitly
kept the real-rows path open "for whenever a flashcards viewing/SRS surface
exists", filing this task at implementation close-out per its §8 ruling 11.

This task decides, and where the decision is "build", builds the surface that
would make real-rows flashcard output viable: a place a student can see their
decks and review cards. Until such a surface exists, agent tooling that
writes real flashcards rows would ship output the student cannot see anywhere
— the invisible-output problem is the whole reason the Q/A-in-notes ruling
was taken, and it should be re-cited (not re-litigated) if this task is
closed as "not now".

Source spec: `Docs/superpowers/specs/2026-08-23-student-workflow-design.md`
§6 and §8 ruling 3/11. Discovery record:
`.superpowers/sdd/2026-08-23-student-workflow/task-2-report.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] A recorded decision (ADR if it sets a long-lived UX/data boundary) on the flashcards surface: what is built now, or an explicit "not now" that cites the invisible-output ruling rather than restating it
- [ ] If built: a screen route renders decks and their flashcards rows from the existing data layer (create/read at minimum; review/SRS flow scoped by the decision), covered by tests against real SQLite
- [ ] If built: the agent-output question is re-visited in a follow-up filing — with rows visible, does an agent write path (e.g. a flashcards target beside `library_save_note`) get built, and under which policy action
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
(To be added when the task is picked up.)
<!-- SECTION:PLAN:END -->
