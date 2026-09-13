---
id: TASK-32464
title: 'Console: the library-access modal paints a blocked glyph for an unselected radio'
status: To Do
assignee: []
created_date: '2026-09-12 00:16'
labels:
  - console
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-32235 gave the Library one meaning per glyph: ☑/☐ for a selection the user makes, ○ for something blocked or disabled. task-32303 moved the Console RAG settings modal's source toggles onto the shared constants, and its Decision was deliberately scoped to that modal — because one Console selection surface still contradicts the legend.

`Widgets/Console/console_library_access_modal.py:45,53` (`ConsoleAccessRadioButton`) sets `self.BUTTON_INNER = "●" if self.value else "○"`: an unselected radio painted with the glyph that means "you cannot use this" everywhere in Library. It is the per-conversation Library access modal — the surface the Console guide's RAG paragraph points the reader to two sentences after describing the ☑/☐ toggles, so the two sit side by side in one reading.

A radio is not a checkbox, and ●/○ is the standard pair for one — so this is a real product decision, not a mechanical swap: either the legend claims radios too (and the pair changes), or radios are carved out explicitly and the legend says so. Whichever way it goes, the glyphs should come from the shared constants rather than a third set of literals in this file. Swept for others: `console_onboarding_state.py:23` (step progress) and `console_cost_tracker.py:388` (cold cache) also use ○ but neither is a selection, so this is the only remaining case.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A decision is recorded on which glyph pair an unselected radio wears, and whether the Library legend covers radios at all
- [ ] #2 The modal reads its glyphs from the shared constants rather than local literals
- [ ] #3 A test pins the painted radio glyphs so a future legend change cannot silently pass this surface by
- [ ] #4 The Library glyph legend's own documentation states where radios sit, so the next audit finds the answer instead of the counterexample
<!-- AC:END -->
