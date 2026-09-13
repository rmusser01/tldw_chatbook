---
id: TASK-32235
title: >-
  Library: one state-glyph legend — `○` means disabled, unchecked and skipped;
  leading `▸` means selected row and disclosure
status: In Progress
assignee: []
created_date: '2026-09-10 14:52'
updated_date: '2026-09-10 19:02'
labels:
  - library
  - ux
  - accessibility
  - design
  - critique-9
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`○ Export selected` (disabled), `○ Media (0)` in the Search/RAG Sources panel (unchecked) and `○ skipped · weird.xyz` (a settled outcome) share one glyph; a leading `▸` marks the selected rail row and, one row below, an expandable node. The guide codifies `○` as the disabled marker and `▸` as the selected row; neither holds. The inline blocked-reason line already carries the meaning for gated actions. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 4.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A documented legend: `▸/▾` trailing for disclosure, `█` leading for the keyboard cursor, `☐/☑` for selection, `✓/✗/–` for settled outcomes; blocked actions keep their inline reason and drop the glyph
- [x] #2 Every Library canvas uses the legend; the guide's glyph sentences match
- [x] #3 Captures at 235x52 and 100x30 pin one meaning per glyph
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add LIBRARY_GLYPH_* constants to library_shell_state.py (one meaning per glyph)
2. Point the Search/RAG source toggle, the ingest canvas toggle label and _GLYPH_SKIPPED at them
3. Update the pins that assert the old collided glyph
4. Legend table in Docs/User_Guide/library.md + the search-and-rag / import-and-export sentences
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
One meaning per glyph, with one deliberate deviation from AC#1: `○` KEEPS
its meaning (a blocked or disabled action) rather than being dropped.
`LIBRARY_DISABLED_ACTION_MARKER` is the only non-colour cue on dozens of
gated Library buttons -- task-4023 AC#1 (RC-07) added it precisely because
those buttons were distinguishable by dimming alone, and most carry their
reason in a tooltip rather than inline, so dropping the glyph would
re-introduce the colour-only meaning the house rules forbid. The other two
meanings moved off it instead, which is what removes the collision:
selection to `☐`/`☑`, the never-attempted outcome to `–`.

Five constants in `library_shell_state.py` are now the legend's one home.
Three call sites changed: the Search/RAG Sources toggle
(`library_search_rag_panel.scope_toggle_label`), the Import canvas's
`_toggle_label`, and `library_ingest_state._GLYPH_SKIPPED` (an aliased
import, since that line was the only one this branch owned in that file).

Pins updated because they asserted the collision: the Sources toggle
labels in `Tests/UI/test_library_shell.py` (7 assertions across 4 tests)
and `Tests/UI/test_library_content_hub.py` (1); the skipped queue row in
`Tests/Library/test_library_ingest_state.py`,
`Tests/UI/test_library_ingest_canvas.py` and
`Tests/UI/test_library_crit8_recovery_copy.py` (1 each). Every other `○`
pin in Tests/ asserts a DISABLED action and is unchanged -- that is the
point.

AC#2 IS LEFT UNTICKED for one known site:
`tldw_chatbook/Widgets/Library/library_note_import_canvas.py:42` still
renders its per-item membership toggle as `✓`/`○`. That file is outside
this branch's file set (the wave gives source-file ownership exclusively),
so it needs one line there plus the matching pin in
`Tests/Widgets/Library/test_library_note_import_canvas.py:438`. The task
stays In Progress until that lands. The Console RAG settings modal
(`console_rag_settings_modal.py`) also mirrors the old pair, but it is not
a Library canvas.

AC#3: captures at both sizes in `<scratch>/crit9/wave/grammar/caps/` --
`rag-sources-100x30.txt` and `notes-select-glyphs-235x52.txt`, the latter
showing `█` (cursor), `☑` (selection) and `○` (blocked "Export selected")
on ONE screen, each meaning exactly one thing.

Files: `Library/library_shell_state.py`, `Library/library_ingest_state.py`
(line 325), `Widgets/Library/library_ingest_canvas.py` (`_toggle_label`),
`Widgets/Library/library_search_rag_panel.py`,
`Tests/UI/test_library_crit9_grammar.py` (new), `Docs/User_Guide/library.md`
(new "State glyphs" section), `Docs/User_Guide/library/search-and-rag.md`,
`Docs/User_Guide/library/import-and-export.md`.
**Fix round 1.** Two guide defects the review found:
`Docs/User_Guide/library/media-and-conversations.md:190` described the
disabled marker as "the same ✓/○ pair the ingest toggles use" -- a pair
this branch retired -- so it now points at the legend instead (that page
gets its own stamp); and the legend's "each glyph means exactly one thing"
claim omitted three glyphs the Import queue paints, so `●` (still
working), `≡` (already in your Library) and `⊘` (cancelled) are now rows in
it. The carve-out import at `library_ingest_state.py:325` is relative and
98 chars instead of 190, still one line.

### AC#2 close-out (hand-off to the critique-9 `notes` branch)

The one canvas Task 5 could not reach was the note-import review, whose
`_choice_label` (`library_note_import_canvas.py`) still rendered a selection
as `✓`/`○` -- colliding with the settled-outcome `✓` one row above it in the
same receipt and with the blocked-action `○` on the buttons beside it. It now
uses `LIBRARY_GLYPH_SELECTED`/`LIBRARY_GLYPH_UNSELECTED`, so every Library
canvas is on the legend. The guide's legend row already named "Import type
toggles" under `☐`/`☑`, so no guide sentence needed changing.

Pins updated: `Tests/Widgets/Library/test_library_note_import_canvas.py` and
`Tests/UI/test_library_note_import_flow.py` (both now assert the constants,
not a literal).
<!-- SECTION:NOTES:END -->
