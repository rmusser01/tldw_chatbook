---
id: TASK-38983
title: "File Notes: during a failed or conflicted save, Choose folder is offered but silently refused, and only typing retries the save"
status: To Do
created_date: 2026-11-06 01:42
labels:
- follow-up
- library
- file-notes
- ux
priority: medium
references:
- tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
- Tests/UI/test_library_file_notes_workspace.py
- Docs/User_Guide/library/file-notes.md
dependencies:
- TASK-38757
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down. Checked on dev 29592ecc9b and on the TASK-38757 branch, which merges first, 2026-11-05. Lines refer to that branch's `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`.

**Probe on dev (load 6.1, 120x40).**
- Steps: open a file, edit it, rename its folder away, let the autosave run, then a poll.
- The save state becomes `error`, and the status reads "Unavailable — the folder cannot be reached; your draft is preserved. Next: Choose folder."
- `set_root(<another valid folder>)` then returns False. The folder does not change, the action status stays empty, and the folder row still reads "Offline · Local folder: notes". The user is told nothing.

**Why Choose folder is refused.**
- The status resolver offers "Choose folder" for `unavailable`, and "Save Copy" for a conflict or a failed save (`:209-227`).
- `set_root` returns early at `if not await self.flush_pending_work(): return False` (`:6548-6549`), before it reports anything.
- `flush_pending_work` refuses whenever the save is in `conflict` or `error` (`:7111-7112`).
- `_change_root_with_deadline` reports nothing for a quick False.

**No Retry.**
- The only Retry copy belongs to Git (`:5646`, `:5707`, `:5734`), and there is no ctrl+s.
- `_start_autosave` runs only from `dirty` (`:6971-6979`), and only typing sets that state (`_editor_changed`, `:7193-7206`).
- In a conflict, typing does not retry either (`:7201`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The "Next:" action shown during a save error or conflict succeeds when taken, or the hint names a step that does
- [ ] #2 A refused folder change says why on the folder row instead of silently doing nothing; red before: the dev probe (False returned, folder row and action status unchanged)
- [ ] #3 A failed save offers an explicit Retry, as a control or a key, that re-runs the save without editing the text; a conflict keeps its Save Copy and Reload choices
- [ ] #4 Tests pin #1-#3 and are red on dev
<!-- AC:END -->
