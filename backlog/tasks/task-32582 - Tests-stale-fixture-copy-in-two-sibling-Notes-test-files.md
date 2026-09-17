---
id: TASK-32582
title: 'Tests: stale fixture copy in two sibling Notes test files'
status: To Do
assignee: []
created_date: '2026-09-14 22:47'
labels:
  - notes
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Left alone deliberately by wave-4 group 7 so that a sibling pin file stayed out of its diff — the stale strings are fixture INPUTS, not assertions, so nothing was green-for-the-wrong-reason. But a fixture that feeds copy the app no longer emits is the seed of the next self-supplied-string pin: the moment somebody asserts on it, the test proves something about a string with no producer. Wave 4 found three such dead pins by grepping the changed strings across Tests/, and this is the same material one step earlier.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both files' fixture strings are re-derived from the shipped producers
- [ ] #2 A grep confirms no assertion anywhere reads the stale spellings
- [ ] #3 The changed-string grep that found them is recorded in the task notes so the next sweep can repeat it
<!-- AC:END -->
