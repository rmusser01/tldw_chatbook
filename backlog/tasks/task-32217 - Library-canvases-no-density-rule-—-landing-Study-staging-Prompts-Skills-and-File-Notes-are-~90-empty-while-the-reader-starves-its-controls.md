---
id: TASK-32217
title: >-
  Library canvases: no density rule — landing, Study staging, Prompts, Skills
  and File Notes are ~90% empty while the reader starves its controls
status: In Progress
assignee: []
created_date: '2026-09-10 14:54'
updated_date: '2026-09-10 20:19'
labels:
  - library
  - layout
  - design
  - critique-9
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The landing hub is ten lines in a 190x44 canvas; Study staging six lines; Prompts and Skills give a 48-cell list a 145-cell 'select something' pane; File Notes two lines on the full screen. Meanwhile the Media reader's Analysis actions sit ~28 blank rows below the text they act on and the note editor's Body gets an 11-row box in a 45-row pane. Media already proved 'a pane with nothing open gives its columns to its sibling'; no sibling canvas got that rule. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 14.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The empty-pane widening rule is applied to Prompts, Skills, Collections and Conversations
- [ ] #2 The note editor Body and the reader content box grow to fill their pane; Analysis actions sit under the content
- [x] #3 The landing either earns its space (recent items, last import, pending review sets) or is narrowed to a readable measure
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing measured tests: at 235x52 with nothing open, each of Prompts/Skills/Collections/Conversations gives its list ~50 cells and its empty work pane ~132.
2. Root cause: resolve_adaptive_reader_layout already has the empty-work-pane branch (task-31979); only Media and Notes pass reader_has_item. Pass it from the four remaining controllers.
3. Density block in _agentic_terminal.tcss: cap the landing at a readable measure (AC#3); record that the vertical half already ships for the surfaces this branch owns.
4. Update the two width pins written against the old behaviour (skills reader collapse, collections collapse).
5. Live-verify at 235x52 / 100x30 / 60x24; docs + stamps.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The empty-pane widening rule already existed in the shared resolver
(`resolve_adaptive_reader_layout`'s task-31979 branch: a work pane with no
document keeps its own floor and the surplus goes to the list). Only Media and
Notes ever passed `reader_has_item`, so the rule was invisible on four
destinations. Prompts, Skills, Collections and Conversations now pass it too --
one keyword argument per controller resolve site, no new mechanism, and nothing
per-canvas that can drift.

AC#1 measured at 235x52 (list / work pane, before -> after):
  Prompts        50 / 132  ->  134 / 48
  Skills         50 / 132  ->  134 / 48
  Collections    50 / 132  ->  134 / 48
  Conversations  50 / 132  ->  138 / 44
Live captures on the seeded profile confirm the same at 235 and 100 columns
(scratchpad crit9/wave/shell/caps/32217-*).

AC#3: `#library-landing-canvas { max-width: 96 }` in the density block appended
to `_agentic_terminal.tcss` -- measured 192 -> 96 cells. Not centred: Textual
has no auto margin and the only container that could centre it
(`#library-canvas-route-content`) hosts every other non-adaptive route, so
centring would have to be bought with a shared-selector change. Left-aligned at
a capped measure is the readable half of the decision; say the word if the
centring matters and it can be done with a route class.

AC#2 NOT ticked, and the reason is worth recording. Both of its named
selectors already carry `height: 1fr` on dev: `#library-prompt-editor-content`
and `#library-media-viewer-content` (the latter from task-31237, with its own
measured rationale in the CSS comment). What the critique actually saw --
"Analysis actions ~28 blank rows below the text they act on" -- is CAUSED by
that rule, not by its absence: a 6-line document in a 1fr box paints ~36 blank
rows inside the border and puts the action row under all of them (live capture
32217-media-reader-235.txt). Fixing that means giving the box `height: auto`
with a ceiling, which reverses a documented decision and belongs to whoever
owns `library_media_viewer.py`. The note-editor Body half is Task 7's by the
plan's own Step 6, and is deliverable from the shared TCSS
(`#library-note-body`, currently `height: auto; max-height: 20`) without
touching that peer's Python.

Files: library_prompts_controller.py, library_skills_controller.py,
library_collections_controller.py, library_conversation_reader_controller.py,
css/components/_agentic_terminal.tcss (+ regenerated bundle),
Tests/UI/test_library_crit9_shell.py, and two width pins updated to the new
rule (Tests/UI/test_library_skills_reader.py,
Tests/UI/test_library_collections_capture_reader.py).
<!-- SECTION:NOTES:END -->
