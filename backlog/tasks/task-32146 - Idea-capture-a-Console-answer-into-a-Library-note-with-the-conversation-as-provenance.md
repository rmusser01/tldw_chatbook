---
id: TASK-32146
title: >-
  Idea: capture a Console answer into a Library note with the conversation as
  provenance
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-08 21:39'
updated_date: '2026-09-11 16:56'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - idea
  - console
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Improvement pitched by the design assessor: a `/note` command or message action that turns a Console answer into a Library note, the reverse of the existing 'Use in Console'. Notes currently implements only one direction of the product's own loop. Size M. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design agreed with the user before implementation
- [ ] #2 The created note records the conversation and message it came from
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reuse the existing Console note seam: a new overflow action `capture-note` ("Save answer as note") on completed ASSISTANT rows, beside TASK-31759's two span note actions.
2. Pure derivations in Chat/console_save_targets.py: first-line title (bounded by CONSOLE_SAVE_TITLE_MAX_CHARS) + provenance keywords (console, conversation:<id>, message:<id>).
3. Dispatch in UI/Console_Modules/message.py through handle_console_message_action (button-id prefix table + branch), writing through app.notes_scope_service.save_note under notes_user_id.
4. Receipt: reuse Widgets/confirmation_dialog.py with Open note -> NavigateToScreen(TAB_LIBRARY, {note_id}) (the existing Home deep-link contract).
5. Fix the sibling Save as... > Note owner-id bug it exposes (current_user is never set anywhere -> notes saved under default_user are invisible in Library > Notes).
6. RED->GREEN tests on the real action route in Tests/Chat/test_console_note_span_actions.py + pure helper units.
7. Live capture 235x52 and 100x30; guides in Docs/User_Guide/console/ and library/notes.md with stamps.
<!-- SECTION:PLAN:END -->
