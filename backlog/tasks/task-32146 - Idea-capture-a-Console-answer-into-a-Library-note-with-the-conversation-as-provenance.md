---
id: TASK-32146
title: >-
  Idea: capture a Console answer into a Library note with the conversation as
  provenance
status: Done
assignee:
  - '@claude'
created_date: '2026-09-08 21:39'
updated_date: '2026-09-11 23:40'
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
- [x] #1 Design agreed with the user before implementation
- [x] #2 The created note records the conversation and message it came from
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reuse the existing Console note seam: a new overflow action `capture-note` ("Save answer as note") on completed ASSISTANT rows, beside TASK-31759's two span note actions.
2. Pure derivations in Chat/console_save_targets.py: first-line title (bounded by CONSOLE_SAVE_TITLE_MAX_CHARS) + provenance keywords (console, conversation:<id>, message:<id>).
3. Dispatch in UI/Console_Modules/message.py through handle_console_message_action (button-id prefix table + branch), writing through app.notes_scope_service.save_note under notes_user_id.
4. Receipt: reuse Widgets/confirmation_dialog.py with Open note -> NavigateToScreen(TAB_LIBRARY, {note_id}) (the existing Home deep-link contract).
5. Fix the sibling Save as... > Note owner-id bug it exposes (current_user is never set anywhere -> notes saved with a literal author id nothing sets; fix round 1 corrected this step's original claim that they were invisible -- see Implementation Notes).
6. RED->GREEN tests on the real action route in Tests/Chat/test_console_note_span_actions.py + pure helper units.
7. Live capture 235x52 and 100x30; guides in Docs/User_Guide/console/ and library/notes.md with stamps.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Console's message overflow menu gains **Capture as note** on a finished
assistant reply. It writes a Local Note whose title is the answer's own first
line (bounded by `CONSOLE_SAVE_TITLE_MAX_CHARS`), whose body is the answer
verbatim, and whose keywords are `console`, `conversation:<id>` and
`message:<id>` — AC#2, read back from the database in the live walk. No LLM
call, no clipboard. A "Saved to Notes" receipt offers **Open note**, which
posts the existing `LIBRARY_NAV_CONTEXT_NOTE_ID` deep link (the same contract
Home's resume-latest uses), landing in the Library note editor on that note.

Built by extending the seams that already existed rather than adding new ones:
one row in `_COMPLETED_ACTIONS`, one branch in the `handle_console_message_
action` dispatcher (same `console-note-actions` worker group as TASK-31759's
two span actions), two pure helpers beside the other Save-as derivations, and
`ConfirmationDialog` for the receipt — no new widget, no new controller
dependency, no schema change.

Decisions and trade-offs:
- **Label.** The design said "Save as note"; the live 24-cell More menu
  truncates past ~15 characters (it already cuts TASK-31759's two labels), so
  the label is **Capture as note**, which fits whole and does not read as a
  variant of the "Save as…" row above it.
- **Owner id.** `Save as… ▸ Note` wrote
  `user_id=getattr(app, "current_user", ...)`; `current_user` is set nowhere in
  the tree, so every note that path saved carried the literal "default_user"
  as its author id — the `client_id` sync attribution and optimistic locking
  read — and each write opened a second cached DB connection to the same
  file. (Fix round 1 withdrew the earlier claim that those notes were
  invisible in Library ▸ Notes: the notes table has no owner column and
  `list_notes` has no owner filter, so they were always listed.) All three
  Console note writers now share one `_console_notes_owner_id()`.
- **Temporary chats.** A note is a local write, and the ephemeral registry
  already blocks `save-as-note`, so `capture-note` has a registry row and is
  offered disabled with that reason rather than silently writing.
- **No run id.** The design allowed one "if the Console exposes one"; there is
  no per-message run id (only `change_review_run_id` on review rows), so the
  keyword is not invented.
- **No setup-modal guard of its own** — it rides the same dispatcher as
  copy/save-as/delete, which the blocking modal covers alike; guarding one of
  twenty sibling actions would be incoherent.
- **No new Inspector row**: the Inspector's static "Message actions" line names
  the primary row plus More…, which stays accurate.

Files: `tldw_chatbook/Chat/console_save_targets.py` (two pure helpers),
`tldw_chatbook/Chat/console_message_actions.py` (action row + gating),
`tldw_chatbook/Chat/console_ephemeral.py` (registry row),
`tldw_chatbook/UI/Console_Modules/message.py` (dispatch, handler, receipt,
owner-id fix), `Tests/Chat/test_console_save_targets.py`,
`Tests/Chat/test_console_message_actions.py`,
`Tests/Chat/test_console_note_span_actions.py`,
`Tests/UI/test_console_native_chat_flow.py`,
`Docs/User_Guide/console/chat-basics.md`, `Docs/User_Guide/library/notes.md`,
`Docs/security/production-diagnostic-inventory.json`.

Fix round 1 (review findings 1–7):
- The "invisible notes" root-cause claim was false and is withdrawn at every
  site (code comments, guide stamp, this file, test docstring); the code
  change stands for the author-id / second-connection reasons above.
- `capture-note` now also refuses at dispatch in a temporary chat (registry
  re-check, as the regenerate image/video branches do), pinned RED→GREEN on
  the real `handle_console_message_action` route.
- A reply opening with a code fence or a heading is titled by its first line
  of text (`console_answer_note_title` skips fence-only lines and drops
  leading `#`/`>`), pinned in `test_console_save_targets.py`.
- The save/notify tail is one helper, `_write_console_note` (keywords in,
  created note id out); capture and the TASK-31759 draft path both use it.
- `_capturable_assistant_answer` carries its own three conditions instead of
  aliasing the speech predicate; the dead `console-message-action-capture-
  note-` prefix entry is gone (production More rows post `console_action_id`).
- Riders filed: task-32512 (TASK-31759's `summarize-note` /
  `save-transcript-note` bypass the ephemeral registry) and task-32469 (the
  24-cell More menu truncates labels past ~15 characters with no ellipsis).
<!-- SECTION:NOTES:END -->
