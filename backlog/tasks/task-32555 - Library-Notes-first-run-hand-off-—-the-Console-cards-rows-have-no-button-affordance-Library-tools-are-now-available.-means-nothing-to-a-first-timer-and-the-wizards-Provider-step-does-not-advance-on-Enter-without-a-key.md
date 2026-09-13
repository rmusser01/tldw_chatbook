---
id: TASK-32555
title: >-
  Library Notes: first-run hand-off — the Console card's rows have no button
  affordance, "Library tools are now available." means nothing to a first-timer,
  and the wizard's Provider step does not advance on Enter without a key
status: To Do
assignee: []
created_date: '2026-09-13 06:48'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor A, persona Jordan, the first-run path into Notes. Task-32245 fixed the destination (Blank note focused); these are the residuals on the way there.

1. The Console first-run card's "Write a note in Library" and "Set up provider" rows are visually identical text lines two rows apart; A's first click hit "Set up provider" and landed in Settings ▸ Providers (A 03, 04).
2. On the first note the toast "Library tools are now available." fires — it describes the rail graduation in the product's words, not Jordan's (A 05).
3. Wizard step 2 (Provider) does not advance on Enter without a key; Jordan wanted notes, not a provider (A 01, 02).

**Cause.** INFERRED.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Console first-run card's actions render as buttons with a shape-based focus cue
- [ ] #2 The rail-graduation toast says what changed in the user's words, or is dropped
- [ ] #3 Enter on the wizard Provider step with no key advances with a visible skip, or the step states why it cannot
<!-- AC:END -->
