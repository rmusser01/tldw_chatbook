---
id: TASK-32555
title: >-
  Library Notes: first-run hand-off — the Console card's rows have no button
  affordance, "Library tools are now available." means nothing to a first-timer,
  and the wizard's Provider step does not advance on Enter without a key
status: In Progress
assignee:
  - '@claude'
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

## Implementation Plan

1. Reproduce live on the fresh profile (captures `handoff-11-card-before.txt` / `handoff-11-card-focus-before.txt`: the card's actions are text lines whose only focus cue is bold+underline; `handoff-13-toast-before.txt`: the first note raises "Library tools are now available."; `handoff-12-wizard-before.txt`: Enter in the empty key field does nothing).
2. RED pins: the card actions render as bordered buttons with an outline focus cue under the real Console sheet; STARTER->GRADUATED raises no toast (existing pins in `test_library_shell.py` / `test_library_entry_compose_once.py` updated to the new truth); Enter in an empty `#setup-provider-api-key` advances past the Provider step and the Summary's Provider row reads not configured.
3. Fix: drop `compact=True` from the card's action buttons (Textual's compact class forces `border: none !important`), give `.console-setup-modal-action` a round border and a heavy left/right outline on focus in the Console sheet; delete the graduation notify; the wizard's key-field Enter with no key clears the provider choice and advances (the key hint says so).
4. Bundle sync + byte budget, GREEN, live re-verify (`handoff-11-card-focus.txt`, `handoff-12-wizard-skip.txt`, `handoff-13-no-toast.txt`), guide pages `console.md` / `library.md` / `First_Run_Setup.md` + stamps.
