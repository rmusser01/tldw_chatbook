---
id: TASK-32073
title: >-
  Library user-guide sweep: eleven documented claims contradicted by the live
  surface
status: Done
assignee: []
created_date: '2026-09-08 18:26'
updated_date: '2026-09-08 19:50'
labels:
  - library
  - docs
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Of 103 concrete claims checked, 11 are contradicted: the whole collections.md page, the Conversations detail (a transcript reader, not a preview), 'Show details' and the retry suffix on Import, the evidence-card keyboard flow, the Get-started compact rail rule, the F6 Reader heavy border, the Trash heading, the select-strip labels, the landing at compact widths, and the whitespace-title keep rule. Shipped but undocumented: Notes 'New / New folder / Add to folder / Move', the Prompts Info tab, the Chunking Lab strip, the captures browser, `ctrl+n` and `/ find note`. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 24.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the eleven contradicted claims is either fixed in the product or corrected in the guide, with a fresh 'Verified against' stamp
- [x] #2 The undocumented shipped controls are documented on their pages
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Every one of the eleven contradicted claims was re-tested against the live surface before being touched, because a critique's CAUSE attribution is unreliable in this repo -- and two of the eleven turned out to be critique errors, not product or doc errors. Recorded as such rather than quietly "fixed".

## The eleven, with verdicts

1. **collections.md (whole page)** -- CONTRADICTED. Rewritten under task-32057.
2. **Conversations detail is a transcript reader, not a preview** -- CONTRADICTED. Live: a "Conversation reader" pane with a Read/Info mode row, `Loaded <id> . 5 of 5 messages . complete.`, a "Find in complete transcript..." box, and per-message `user`/`assistant` blocks. The guide's "shows the preview (title, Messages: N, Updated: age)" row was wrong.
3. **Import "Show details"** -- PARTIAL, now stated. The row action ships, but only when the failed job carried an `error_detail`; infrastructure failures that stop the worker before it opens a file carry none, which is why the review saw a raw errno with no details row. Cross-referenced task-32054.
4. **Import retry suffix** -- CONTRADICTED. Ships as `. attempt N` (`library_ingest_state.py:494`, `retry_count + 1`); the guide's `. retry 1` is Home's Active work card, a different surface.
5. **Evidence-card keyboard flow** -- CONTRADICTED. `enter`/`o`/`u` are bound and advertised in the footer, and each gates on a focused `.library-rag-result-card`; nothing gives a card focus by keyboard, so the advertised flow needs a mouse. Keys unchanged, table now says which half works. Cross-referenced task-32053.
6. **Get-started compact rail rule** -- CONTRADICTED. Verified live on an EMPTY profile whose config.toml pre-existed: full rail, every count `(0)`, no Get started. `coerce_library_lifecycle` returns EXPANDED whenever the config was not created in the same run. Cross-referenced task-32059.
7. **F6 Reader heavy border** -- **HOLDS. The critique was wrong.** F6 onto the Reader content box repainted its frame from `+--+` to a heavy `box-drawing` frame in a plain-text capture (caps/11-f6-reader-stop.txt). Guide unchanged; the stamp records the re-test so this is not re-filed a third time.
8. **Trash heading** -- CONTRADICTED. `Local Trash . 1 item` live; `. N matching` while filtered; bare `Local Trash` when too narrow. Guide said `Trash (N)` in two places, both corrected.
9. **Select-strip labels at the 36-cell floor** -- **NOT REPRODUCED.** At 235x52 and at 100x30 with the Reader open, "0 selected", "Select all 11 shown", "Clear", "o Export", "o Review", "o Analyze", "o Delete" and "Done" all painted in full. The Items pane does not reach 36 cells at either width. Bulk-toolbar clipping below 110 columns is already task-15140; the page now makes no wider claim.
10. **Landing at compact widths** -- CONTRADICTED. At 100 columns the landing still paints beside the rail, counts line and all. "Compact" in the guide meant the below-64-column single stage; the text now says so. Cross-referenced task-32066.
11. **Whitespace-title keep rule** -- CONTRADICTED. The abandon-discard check is `not raw_title.strip()`, so a spaces-only title with an empty body and no keywords is still discarded -- the guide's "typing anything keeps it" was too strong.

## Undocumented shipped controls, now documented

- **Notes**: `New` and `New folder` toolbar actions; folder-selected `Rename` / `Move` / `Remove`; note-selected `Add to folder` / `Move note` / `Remove placement`; `Restore folder`. Each with its real disabled reason (sync-managed folder, sync-managed placement, stale branch, automatic Unfiled group).
- **Notes keys**: `ctrl+n new note` and `/ find note`, both advertised in the footer and both absent from the page, which claimed Enter-in-the-filter was "the only screen-specific key".
- **Prompts**: the editor's third mode button, **Info** -- persisted source / definition state, the draft-vs-saved reminder, and the Collections membership block.
- **Chunking Lab strip**: documented once on library.md (it is identical on every canvas) and cross-referenced from collections.md, skills.md and file-notes.md. Cross-referenced task-32064.
- The captures browser itself is task-32057's page.

## Approach

Every fix describes what ships TODAY and names the task that will change it, rather than documenting a future. Live verification used the group's two scratch profiles at 235x52 and 100x30 on socket `crit8-docs`; captures are in the group's caps/ directory. Import could not be exercised end to end (the host's POSIX semaphores are exhausted, so every local import fails at process-pool start), so claims 3 and 4 were verified against the shipping code paths and that limitation is recorded in the page's stamp.

## Files

`Docs/User_Guide/library.md`, `library/media-and-conversations.md`, `library/import-and-export.md`, `library/notes.md`, `library/search-and-rag.md`, `library/prompts.md`, `library/skills.md`, `library/file-notes.md`. No product code changed for this task.
<!-- SECTION:NOTES:END -->
