---
id: TASK-14826
title: Make the ingest option-error gate locatable and clearable
status: Done
assignee:
  - '@claude'
created_date: '2026-08-10 21:00'
updated_date: '2026-08-10 21:42'
labels:
  - library
  - ingest
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P2 of the 2026-08-10 re-critique. An invalid option value blocks Start with a message the user cannot act on, and the block outlives the selection that caused it.

Observed live: setting `#opt-generic-chunk_size` to `7` produces `Fix the highlighted options to start: Chunk size must be between 100 and 5000.` The message names no group; collapsing the panel highlights nothing (the `-ingest-option-invalid` class is applied to the Input, which is inside the collapsed body); the collapsed title shows `Chunk size: 7` with no error marker. Pressing Clear resets the path and restores the intro but LEAVES the block in place; leaving Ingest and re-entering still blocks. The value is not persisted to config, so the block dies on restart — which also makes it irreproducible in a bug report.

The repo already has the better pattern: Settings' field-level search lands focus ON the offending field.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The gate message names the group and field that is invalid
- [x] #2 A collapsed panel containing an invalid value is visibly marked as such
- [ ] #3 The gate offers a way to reach the offending field (focus lands on it), rather than only describing it
- [ ] #4 Clearing the staged selection does not leave an unreachable block behind
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Mark a collapsed panel that contains an invalid value, in the one place a collapsed panel still renders: its title.
2. Use TEXT, not a CSS class -- the screen's in-place update assigns ``Collapsible.title`` and nothing else, so a class-based marker would drift the moment a value changes without a recompose.
3. Report the gate-wording, focus-the-field and Clear-clears-the-block halves, whose root causes are in sibling-held files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Done here (AC#2).** ``build_type_group_title`` now validates each
ENABLED field and leads the title with ``⚠ {field label} needs fixing``
(plus ``(+N more)``), and drops the invalid value's own ``name: value``
pair so the panel cannot report ``Chunk size: 7`` as a setting. The marker
is text, deliberately: the screen's in-place receipt update assigns
``Collapsible.title`` and nothing else, so a CSS-class marker would drift
out of sync, and a glyph survives monochrome (the house rule). Mutation:
disabling the validation branch restores the live title
``Plain text & HTML — Chunk size: 7, …`` with no ``⚠``.

**Not done here -- root cause in a sibling-held file (verbatim
handoff):**
- AC#1 (gate names the group and field): ``build_library_ingest_state``
  in ``Library/library_ingest_state.py`` builds
  ``start_quiet_line = f"Fix the highlighted options to start:
  {option_errors[0][2]}"``. ``option_errors`` entries are
  ``(group, field, message)``, so the group and field are already in hand
  -- render them: "Fix Plain text & HTML ▸ Chunk size to start: must be
  between 100 and 5000."
- AC#3 (focus lands on the offending field): needs an action on
  ``UI/Screens/library_screen.py`` that expands
  ``#type-group-{option_errors[0][0]}`` and focuses
  ``#opt-{group}-{field}``, bound from the gate line (Settings' field-level
  search is the incumbent pattern to copy).
- AC#4 (Clear does not leave the block behind): the Clear-path handler in
  ``library_screen.py`` resets the path but not the staged
  ``form.type_options``, so ``option_errors`` survives a cleared
  selection and blocks the next one. Clear must either reset the option
  values it is clearing the selection for, or the gate must only consider
  option errors for groups the CURRENT pre-flight staged.

Files touched: ``Widgets/Library/library_ingest_canvas.py``,
``Tests/UI/test_library_ingest_canvas.py``,
``Docs/User_Guide/library/import-and-export.md``.
<!-- SECTION:NOTES:END -->
