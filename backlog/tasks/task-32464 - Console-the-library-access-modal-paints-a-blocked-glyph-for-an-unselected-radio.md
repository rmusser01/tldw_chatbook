---
id: TASK-32464
title: >-
  Console: the library-access modal paints a blocked glyph for an unselected
  radio
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-12 00:16'
updated_date: '2026-09-14 14:25'
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
- [x] #1 A decision is recorded on which glyph pair an unselected radio wears, and whether the Library legend covers radios at all
- [x] #2 The modal reads its glyphs from the shared constants rather than local literals
- [x] #3 A test pins the painted radio glyphs so a future legend change cannot silently pass this surface by
- [x] #4 The Library glyph legend's own documentation states where radios sit, so the next audit finds the answer instead of the counterexample
<!-- AC:END -->

## Decision

**AC#1 — radios keep ●/○ and the Library legend carves them out** (the
controller's call, recorded verbatim; revisitable):

> RULING on AC#1 (mine — record it verbatim under `## Decision` in the task,
> attributed as the controller's call, and say in the docs that it can be
> revisited): **radios keep ●/○ and the Library legend carves them out
> explicitly.** A radio is a single-choice control; painting it ☑/☐ would
> promise multi-select it does not offer, which is a worse lie than the glyph
> collision. So: selection glyphs are ☑/☐ for a multi-select list, ●/○ for a
> radio group (exactly one sibling is ● at all times, which is what tells the
> reader it is a chooser), and a bare ○ outside a radio group keeps its
> blocked/disabled meaning.

What ships matches: the pair lives beside `LIBRARY_GLYPH_SELECTED` /
`LIBRARY_GLYPH_UNSELECTED` in `Library/library_shell_state.py` as
`LIBRARY_GLYPH_RADIO_SELECTED` / `LIBRARY_GLYPH_RADIO_UNSELECTED`, the modal
reads them instead of its own literals, and `Docs/User_Guide/library.md`'s
legend states the carve-out and says it can be revisited.

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the controller's ruling: radios keep the paired dot glyphs; the Library legend carves them out.
2. Add the radio pair beside LIBRARY_GLYPH_SELECTED/UNSELECTED in Library/library_shell_state.py.
3. Point ConsoleAccessRadioButton at the shared constants.
4. Red-first pin on the painted radio glyphs.
5. Docs: the Library legend row for radios + the Console RAG page's open question.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The glyphs stay ●/○ (see `## Decision`); what changed is where they come from
and what the legend says about them.

- `Library/library_shell_state.py` gains `LIBRARY_GLYPH_RADIO_SELECTED` /
  `LIBRARY_GLYPH_RADIO_UNSELECTED` beside the existing checkbox pair, with the
  carve-out reasoning in the comment.
- `ConsoleAccessRadioButton._button` reads those instead of its two literals.
- `Docs/User_Guide/library.md`'s **State glyphs** legend gains a radio row and
  a paragraph stating the carve-out, that a bare `○` outside a radio group
  still means blocked, and that the decision is revisitable;
  `Docs/User_Guide/console/context-and-rag.md`'s open question is closed.

Evidence: two pins in `Tests/UI/test_console_library_access_modal.py`. The
literal pin is red against the pre-fix module and green after; the painted-
glyph pin was mutation-tested by pointing the modal at the checkbox pair
(1 failed, 9 passed). Value equality alone cannot catch AC#2 here — the ruling
keeps the same two characters — which is why one pin reads the module source.

Not fixed here, filed as a rider instead: `SetupRadioButton`
(`UI/Wizards/FirstRunSetupWizard.py:146`) is the same widget with the same two
literals, and paints the same pair live (capture
`00-wizard-radios.txt`: "▐●▌ Quick setup" / "▐○▌ Full setup").

Modified: `tldw_chatbook/Library/library_shell_state.py`,
`tldw_chatbook/Widgets/Console/console_library_access_modal.py`,
`Tests/UI/test_console_library_access_modal.py`,
`Docs/User_Guide/library.md`, `Docs/User_Guide/console/context-and-rag.md`.
<!-- SECTION:NOTES:END -->
