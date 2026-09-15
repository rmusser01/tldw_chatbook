---
id: TASK-32607
title: >-
  Library Notes: Info's footer says enter run action for every control including
  Delete, and two stops show no focus at all
status: In Progress
assignee: []
created_date: '2026-09-15 06:38'
updated_date: '2026-09-15 17:58'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A P1, personas Sam and Jordan, edit workflow.

What happened. Everywhere else in the editor the footer names the focused control ('enter save note', 'enter use in Console', 'enter undo delete'). Inside Info, eleven consecutive Tab presses produced the same generic chip for '‹ Notes', Keywords, Copy, Export Markdown, Export text and Delete (A cap 11 plus 11 Tab probes), and two of those stops showed no focus indicator anywhere on screen (A caps 12, 13). A keyboard user pressing Enter in Info is choosing blind between 'copy to clipboard' and 'delete this note'. The ANSI decode also shows Delete's label rendered grey #a5a5a5 against its siblings' white #e1e1e1, so the destructive control reads as the disabled one.

Cause, PROVEN. Of the four Notes footer tiers, navigator, preview and editor are wrapped in _with_library_notes_focus_chip (UI/Screens/library_screen.py:8224, :8236, :8256); the context tier -- Info -- is not (:8241-8245), so every Info stop renders the tier's literal ('enter', 'run action') from LIBRARY_NOTES_CONTEXT_SHORTCUTS (:1553-1556). Wave 4's task-32537 (#2683) added the chip to Preview and left this tier alone.

Docs contradicted: notes.md says 'The footer names whichever editor control has focus as an enter … chip, so focus is never unaccounted for.'
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every Info control names itself on the footer when focused, in the same grammar the editor tier uses
- [x] #2 Every Info tab stop paints a visible focus indicator that survives a monochrome capture
- [x] #3 Delete in Info is styled as destructive rather than dimmer than its neighbours
- [x] #4 A test pins the focus chip for each Info control by name, so a new control cannot ship chip-less
- [x] #5 The Notes list footer's keyboard map is true of the screen in every navigator focus state, and extends it with n (new note), g (go to folder tree) and e (export selected)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Wrap the context (Info) footer tier in the shared focus chip, replacing the generic 'run action' in place so the narrow chip budget is unchanged; drop the enter chip where no Info control owns Enter.
2. Add Enter labels for the Info stops that have none (keywords field, backlink rows).
3. Give #library-note-context-keywords the field + focus treatment the dead #library-note-keywords rule was spending on an undisplayed twin.
4. Style Info's Delete with the readable error role instead of $ds-text-muted.
5. Extend the navigator tier: drop the lie, add n / g / e -- n already fires, so only g (focus the folder tree) and e (export selected) need bindings.
6. Pin each Info control's chip by name, plus the two CSS blocks and the three new keys.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Info was the one Notes footer tier never wrapped in the shared focus chip, so it rendered its literal 'enter run action' on every stop. The tier now replaces that label IN PLACE with the focused control's own (the lasting_add pattern -- this tier already spends the narrow chip budget on 'enter') and DROPS the chip entirely where the control owns no Enter: the Keywords field has no Input.Submitted handler, and a generic label there is the same lie one control smaller. That matches the editor tier's grammar, where #library-note-title gets no chip either. Info's backlink rows carry no DOM id at all (`_backlink_buttons` identifies them by a note_id attribute and a class), so `_library_focus_enter_label` gained a class branch ahead of its id lookup.

AC#2, measured rather than assumed: every stop of the navigator, the editor, Preview and Info was focused in turn and its computed (text-style, outermost edge type) pair diffed against its blurred pair -- the half of a focus cue a monochrome capture keeps. Two Info stops changed nothing. `#library-note-context-region` swapped an accent colour into a border that was already `solid`, with the reset's `*:focus { outline: solid }` painting the same glyphs over the same cells; it (and its Preview twin) now takes a `heavy` border plus `outline: none`, one cell wide like solid, so geometry is untouched. `#library-note-context-keywords` was given the same field treatment: the `#library-note-title, #library-note-keywords` rules were being spent on a twin inside `#library-note-wide-utilities`, which apply_session_state sets `display = False` unconditionally, so the live field was styled unlike the Title field directly above it. Adding the live id to both selectors was preferred to renaming, so the existing `#library-note-keywords:focus` pin stays honest about the widget that is still composed. **Fix round 1 correction:** the first write-up of this claimed the live field "had no field styling at all" and no monochrome-visible cue. That is FALSE and was never measured. With both selector additions reverted in place, `#library-note-context-keywords` still changes border type `tall` -> `solid` on focus from the generic `Input:focus`, and all three parametrisations of `test_every_note_pane_tab_stop_paints_a_shape_change_on_focus` stay green without them -- the only revert of seven that did not go red. The change is kept as a CONSISTENCY change (the live field now matches the Title field, and its monochrome cue becomes `text-style: bold` rather than the border-type swap); it is deliberately not pinned, because there is no defect to pin. The `[context]` red originally attributed to this hunk belongs entirely to the reading-region hunk.

AC#3: `.library-media-action-danger`'s whole treatment is `color: $ds-text-muted`, which is right where danger sits inside a neutral strip (the media Reader's More row) and wrong where the control IS the section -- Info's own 'Danger' heading already says that. Scoped to `#library-note-context-delete`, using the same readable token as #library-media-trash-delete-confirm.

AC#5 (folded-in idea 10): the map was made true before it was extended. 'n' already fired in the navigator (on_key shares library_notes_new's gate since task-32138), so advertising it cost no code; 'g' (focus the folder tree, resuming on the selected placement) and 'e' (press the composed Export selected button, so the mutation fence and empty-selection guard stay single-authority) are new bindings gated in check_action the same way '/' is. 'e' lives on the select-mode tier because that is the only state its button is composed in, and the footer asks each key's own gate whether to advertise it rather than keeping a second copy of the predicate. Every printable chip is dropped while a text field holds focus (the task-32609 half of the same hunk).

Files: UI/Screens/library_screen.py, css/components/_agentic_terminal.tcss (+ generated screen_agentic_library.tcss / widget_defaults_*), Tests/UI/test_library_notes_w5_kbd_focus.py, Tests/UI/test_screen_footer_hints.py (its navigator tuple; its editor tuple had been stale since task-32247 and was red on dev -- re-pinned to what ships), Docs/User_Guide/library/notes.md.
<!-- SECTION:NOTES:END -->
